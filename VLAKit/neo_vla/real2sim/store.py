"""Workdir primitives: fingerprints, no-overwrite writes, the writer lock, runs, receipts, children.

Every stage run gets a fresh, sequence-numbered directory. It records the fingerprint of its inputs
in STARTED.json before any work happens and ends with a RECEIPT.json that is published atomically
and never replaced. A receipt lists the hashes of the files it vouches for; a receipt whose files no
longer match is invalid. A run that ends without a receipt is closed later with an INCOMPLETE
receipt, keeping its files as evidence, so an interrupted rerun can never fall back to an earlier
pass for the same inputs.
"""
from __future__ import annotations

import datetime
import hashlib
import json
import os
import re
import secrets
import signal
import socket
import stat
import subprocess
import threading
import time
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

RECEIPT = "RECEIPT.json"
STARTED = "STARTED.json"
RECEIPT_SCHEMA = "neo-vla-real2sim-receipt/1"
RUN_ID = re.compile(r"r(\d{6,})-(\d{8}T\d{6}Z)-([0-9a-f]{6})")
OUTPUT_LIMIT_BYTES = 64 << 20   # per captured stream of a child process
TERM_GRACE_S = 10.0
KILL_GRACE_S = 5.0
POLL_S = 0.05


class PathError(ValueError):
    """A path escapes its root, uses a symlink or names something unexpected."""


class LockedError(RuntimeError):
    """Another pipeline process holds the workdir lock."""


def utc_now() -> str:
    return datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds")


def canonical_sha256(value: Any) -> str:
    text = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(text.encode()).hexdigest()


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def file_record(path: str | Path) -> dict:
    return {"sha256": sha256_file(path), "size": os.path.getsize(path)}


def contained(root: str | Path, rel: str | Path) -> Path:
    """root/rel for a relative path without '..', absolute parts or symlinked components."""
    root = Path(root).resolve()
    rel = Path(rel)
    if rel.is_absolute() or not rel.parts or any(p in ("..", ".", "") for p in rel.parts):
        raise PathError(f"{rel}: expected a plain relative path inside {root}")
    current = root
    for part in rel.parts:
        current = current / part
        if current.is_symlink():
            raise PathError(f"{current}: symlinks are not followed")
    return current


def ensure_inside(path: str | Path, root: str | Path) -> Path:
    root = Path(root).resolve()
    path = Path(os.path.abspath(path))
    try:
        rel = path.relative_to(root)
    except ValueError as error:
        raise PathError(f"{path} is outside {root}") from error
    return contained(root, rel) if rel.parts else root


def write_new(path: Path, data: bytes) -> None:
    """Create a file that must not exist yet."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "xb") as handle:
        handle.write(data)
        handle.flush()
        os.fsync(handle.fileno())


def dumps(value: Any) -> bytes:
    return (json.dumps(value, indent=1, allow_nan=False) + "\n").encode()


def write_new_json(path: Path, value: Any) -> None:
    write_new(path, dumps(value))


def publish_json(path: Path, value: Any) -> None:
    """Atomic, no-overwrite publication: readers see either nothing or the complete file."""
    tmp = path.with_name(f".{path.name}.{secrets.token_hex(4)}.tmp")
    write_new(tmp, dumps(value))
    try:
        os.link(tmp, path)
    finally:
        tmp.unlink()


def read_json(path: str | Path) -> Any:
    return json.loads(Path(path).read_text())


def read_bounded(path: Path, limit: int) -> bytes:
    """Read a regular, non-symlinked file of at most `limit` bytes, checking the size before reading
    and never reading more than limit + 1 bytes, so a growing file cannot exhaust memory."""
    try:
        fd = os.open(path, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0))
    except OSError as error:
        raise PathError(f"{path}: cannot open without following symlinks ({error})") from error
    try:
        info = os.fstat(fd)
        if not stat.S_ISREG(info.st_mode):
            raise PathError(f"{path} is not a regular file")
        if info.st_size > limit:
            raise PathError(f"{path} has {info.st_size} bytes, above the {limit}-byte limit")
        chunks, total = [], 0
        while total <= limit:
            chunk = os.read(fd, min(1 << 20, limit + 1 - total))
            if not chunk:
                break
            chunks.append(chunk)
            total += len(chunk)
        if total > limit:
            raise PathError(f"{path} grew past the {limit}-byte limit while it was read")
        return b"".join(chunks)
    finally:
        os.close(fd)


def tree_records(root: Path, *, hashes: bool = True) -> dict:
    """{relative path: record} for every file below root; symlinks are refused."""
    out = {}
    for current, dirs, files in os.walk(root):
        dirs.sort()
        for name in sorted(dirs + files):
            path = Path(current) / name
            if path.is_symlink():
                raise PathError(f"{path}: symlinks are not followed")
        for name in sorted(files):
            path = Path(current) / name
            rel = path.relative_to(root).as_posix()
            out[rel] = file_record(path) if hashes else {"size": path.stat().st_size}
    return out


class Lock:
    """One writer per workdir. A stale lock is reported, never broken automatically."""

    def __init__(self, workdir: Path, command: str) -> None:
        self.path = Path(workdir) / "LOCK"
        self.info = {"pid": os.getpid(), "host": socket.gethostname(), "started_utc": utc_now(),
                     "command": command, "token": secrets.token_hex(8)}

    def __enter__(self) -> "Lock":
        self.path.parent.mkdir(parents=True, exist_ok=True)
        try:
            fd = os.open(self.path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o644)
        except FileExistsError as error:
            try:
                other = self.path.read_text()[:500]
            except OSError:
                other = "unreadable"
            raise LockedError(f"{self.path} is held: {other.strip()}. If that process is no longer "
                              f"running, delete the LOCK file and run the command again.") from error
        with os.fdopen(fd, "w") as handle:
            handle.write(json.dumps(self.info) + "\n")
        return self

    def __exit__(self, *exc: Any) -> None:
        try:
            if json.loads(self.path.read_text()).get("token") == self.info["token"]:
                self.path.unlink()
        except (OSError, ValueError):
            pass


# ----------------------------------------------------------------------------- runs and receipts
def run_seq(name: str) -> int:
    m = RUN_ID.fullmatch(name)
    return int(m.group(1)) if m else -1


def runs(stage_dir: Path) -> list:
    """Run directories of a stage, oldest first, ordered by their sequence number."""
    base = stage_dir / "runs"
    found = [p for p in base.iterdir() if p.is_dir() and RUN_ID.fullmatch(p.name)] if base.is_dir() else []
    return sorted(found, key=lambda p: (run_seq(p.name), p.name))


def run_dir(stage_dir: Path, run_id: str) -> Path:
    """The directory of a run named by its id; ids are never treated as paths of their own."""
    if not isinstance(run_id, str) or not RUN_ID.fullmatch(run_id):
        raise PathError(f"{run_id!r} is not a run id")
    return contained(stage_dir, Path("runs") / run_id)


def new_run(stage_dir: Path, inputs: Any = None) -> Path:
    """A fresh run directory numbered after every existing run of the stage. STARTED.json records
    the inputs fingerprint first, so an interrupted run still counts as the latest attempt."""
    base = stage_dir / "runs"
    base.mkdir(parents=True, exist_ok=True)
    seq = max([run_seq(p.name) for p in base.iterdir()] + [0]) + 1
    stamp = datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    while True:
        run = base / f"r{seq:06d}-{stamp}-{secrets.token_hex(3)}"
        try:
            run.mkdir()
            break
        except FileExistsError:
            seq += 1
    write_new_json(run / STARTED, {"schema": RECEIPT_SCHEMA, "stage": stage_dir.name, "run": run.name, "seq": seq,
                                   "inputs_fingerprint": None if inputs is None else canonical_sha256(inputs),
                                   "pid": os.getpid(), "host": socket.gethostname(), "started_utc": utc_now()})
    return run


def write_receipt(run: Path, *, stage: str, status: str, inputs: Any, outputs: Iterable[str],
                  reasons: Iterable[str] = (), not_established: Iterable[str] = (), fingerprint: str | None = None,
                  **extra: Any) -> dict:
    files = {rel: file_record(contained(run, rel)) for rel in sorted(set(outputs))}
    receipt = {**extra, "schema": RECEIPT_SCHEMA, "stage": stage, "run": run.name, "status": status,
               "inputs_fingerprint": fingerprint or canonical_sha256(inputs), "inputs": inputs, "outputs": files,
               "outputs_fingerprint": canonical_sha256(files), "reasons": list(reasons),
               "not_established": list(not_established), "finished_utc": utc_now()}
    publish_json(run / RECEIPT, receipt)
    return receipt


def close_incomplete(stage_dir: Path) -> list:
    """Give every run that ended without a receipt an INCOMPLETE receipt. Its files stay where they
    are as evidence and count for nothing. Call only while holding the workdir lock."""
    closed = []
    for run in runs(stage_dir):
        if (run / RECEIPT).exists():
            continue
        try:
            started = read_json(run / STARTED)
        except (OSError, ValueError):
            started = {}
        write_receipt(run, stage=stage_dir.name, status="INCOMPLETE", inputs=None, outputs=[],
                      fingerprint=started.get("inputs_fingerprint") or "unknown",
                      reasons=["the run ended without a receipt (interrupted or crashed); its files are kept as "
                               "evidence and are not used"], not_established=["anything"])
        closed.append(run.name)
    return closed


def load_receipt(run: Path) -> tuple:
    """(receipt, problems). Problems name outputs that are missing or changed after the receipt."""
    path = run / RECEIPT
    if not path.is_file():
        return None, ["no receipt (running or interrupted)"]
    try:
        receipt = read_json(path)
    except ValueError as error:
        return None, [f"receipt unreadable: {error}"]
    problems = []
    for rel, record in (receipt.get("outputs") or {}).items():
        try:
            if file_record(contained(run, rel)) != record:
                problems.append(f"{rel} changed after the receipt")
        except (OSError, PathError):
            problems.append(f"{rel} missing")
    return receipt, problems


def latest(stage_dir: Path, inputs: Any) -> tuple:
    """The newest run of a stage for these inputs: (run, receipt, problems), or (None, None, []).

    Newest means the highest sequence number, finished or not. A newer run without a receipt is
    reported as INCOMPLETE rather than letting an older pass stand in for it.
    """
    fingerprint = canonical_sha256(inputs)
    newest = None
    for run in runs(stage_dir):
        if _fingerprint(run) == fingerprint:
            newest = run
    if newest is None:
        return None, None, []
    receipt, problems = load_receipt(newest)   # outputs are hashed only for the run that counts
    if receipt is None:
        return newest, {"status": "INCOMPLETE", "reasons": problems}, problems
    return newest, receipt, problems


def _fingerprint(run: Path) -> str | None:
    """The inputs fingerprint a run belongs to: from its receipt, else from STARTED.json."""
    for name in (RECEIPT, STARTED):
        try:
            value = read_json(run / name).get("inputs_fingerprint")
        except (OSError, ValueError, AttributeError):
            continue
        if value is not None:
            return value
    return None


# ----------------------------------------------------------------------------- child processes
HAVE_WNOWAIT = hasattr(os, "waitid") and hasattr(os, "WNOWAIT")


def group_members(pgid: int) -> list | None:
    """Live (non-zombie) processes of a process group read from /proc, or None where /proc is absent."""
    proc = Path("/proc")
    if not (proc / "self" / "stat").is_file():
        return None
    members = []
    for entry in os.scandir(proc):
        if not entry.name.isdigit():
            continue
        try:
            text = Path(entry.path, "stat").read_text()
        except OSError:
            continue
        head, _, rest = text.rpartition(")")
        fields = rest.split()
        if len(fields) < 20 or not fields[2].lstrip("-").isdigit():
            continue
        if int(fields[2]) == pgid and fields[0] not in ("Z", "X"):
            members.append({"pid": int(entry.name), "state": fields[0], "start_ticks": int(fields[19]),
                            "comm": head.partition("(")[2][:64]})
    return members


def _cleanup_group(pgid: int, term_grace_s: float, kill_grace_s: float) -> dict:
    """Terminate what is left of the child's process group. Called only while the group leader is
    still unreaped (running or a zombie), so the group ID cannot belong to anyone else."""
    report: dict = {"group": pgid, "signals": [], "members": group_members(pgid), "survivors": None,
                    "observed": HAVE_WNOWAIT and Path("/proc/self/stat").is_file()}
    if report["members"] is not None and not report["members"]:
        report["survivors"] = []
        return report
    for sig, grace in ((signal.SIGTERM, term_grace_s), (signal.SIGKILL, kill_grace_s)):
        try:
            os.killpg(pgid, sig)
        except ProcessLookupError:
            report["survivors"] = []
            return report
        except PermissionError as error:   # a member we may not signal; reported, never retried wider
            report["error"] = repr(error)
            break
        report["signals"].append(sig.name)
        end = time.monotonic() + grace
        while time.monotonic() < end:
            alive = group_members(pgid)
            if alive is not None and not alive:
                report["survivors"] = []
                return report
            time.sleep(POLL_S)
    report["survivors"] = group_members(pgid)
    return report


class _Capture(threading.Thread):
    """Copies one output pipe to a file, keeping at most `limit` bytes; the rest is drained and dropped."""

    def __init__(self, pipe: Any, path: Path, limit: int) -> None:
        super().__init__(daemon=True)
        self.pipe, self.limit = pipe, limit
        self.handle = open(path, "xb")
        self.written, self.seen, self.limited = 0, 0, False

    def run(self) -> None:
        try:
            while True:
                chunk = self.pipe.read1(1 << 16)
                if not chunk:
                    break
                self.seen += len(chunk)
                room = self.limit - self.written
                if room > 0:
                    self.handle.write(chunk[:room])
                    self.written += min(room, len(chunk))
                if len(chunk) > room:
                    self.limited = True
        finally:
            self.handle.close()


def run_bounded(argv: Sequence[str], *, cwd: Path, env: Mapping[str, str] | None, timeout_s: float,
                stdout: Path, stderr: Path, stdin: Path | None = None, output_limit_bytes: int = OUTPUT_LIMIT_BYTES,
                term_grace_s: float = TERM_GRACE_S, kill_grace_s: float = KILL_GRACE_S) -> dict:
    """Run one child process (argument list, no shell) with a deadline and capped output.

    The child starts a new session, so it leads its own process group. Its exit is observed without
    reaping it (waitid with WNOWAIT), which keeps its PID, and therefore the group ID, reserved while
    the rest of the group is cleaned up: on Linux the remaining members are listed from /proc, sent
    SIGTERM, and after `term_grace_s` SIGKILL. This happens after a timeout, after an output stream
    reaches `output_limit_bytes`, and after a normal exit that left descendants behind. Only then is
    the child reaped; nothing is signalled afterwards. Where WNOWAIT is unavailable (macOS) the group
    is cleaned up only on timeout or output limit, before reaping, without listing its members.
    Processes that leave the group (setsid, setpgid) are outside this control on every platform.
    `stdin` names a file to feed the child; otherwise stdin is closed. Output files must not exist.
    """
    started = time.monotonic()
    stdin_handle = open(stdin, "rb") if stdin is not None else None
    try:
        proc = subprocess.Popen(list(argv), cwd=str(cwd), env=dict(env) if env is not None else None,
                                stdin=stdin_handle if stdin_handle is not None else subprocess.DEVNULL,
                                stdout=subprocess.PIPE, stderr=subprocess.PIPE, start_new_session=True)
    except OSError as error:
        if stdin_handle is not None:
            stdin_handle.close()
        for path in (stdout, stderr):
            write_new(path, b"")
        return {"returncode": None, "timed_out": False, "output_limit_reached": None, "launch_error": repr(error),
                "wall_s": 0.0, "cleanup": None}
    captures = {"stdout": _Capture(proc.stdout, stdout, output_limit_bytes),
                "stderr": _Capture(proc.stderr, stderr, output_limit_bytes)}
    for capture in captures.values():
        capture.start()
    timed_out, exited, reaped = False, False, False
    while True:
        if HAVE_WNOWAIT:
            try:
                exited = os.waitid(os.P_PID, proc.pid, os.WEXITED | os.WNOHANG | os.WNOWAIT) is not None
            except ChildProcessError:   # reaped by someone else: its group ID may already be reused
                exited = reaped = True
        else:
            exited = reaped = proc.poll() is not None   # poll() reaps the child
        limited = [name for name, c in captures.items() if c.limited]
        if exited or limited:
            break
        if time.monotonic() - started > timeout_s:
            timed_out = True
            break
        time.sleep(POLL_S)
    if not reaped:
        cleanup = _cleanup_group(proc.pid, term_grace_s, kill_grace_s)
    else:
        cleanup = {"group": proc.pid, "signals": [], "members": None, "survivors": None, "observed": False,
                   "note": "the child was already reaped, so its process group is not signalled"}
    try:
        returncode = proc.wait(timeout=kill_grace_s + 5.0)
    except subprocess.TimeoutExpired:
        returncode = None   # still alive after SIGKILL (for example blocked in the kernel); reported, not reaped
        cleanup["leader_not_reaped"] = True
    for capture in captures.values():
        capture.join(timeout=5.0)
        if not capture.is_alive():   # a pipe still held by an escaped process stays with its reader
            try:
                capture.pipe.close()
            except OSError:
                pass
    if stdin_handle is not None:
        stdin_handle.close()
    limited = [name for name, c in captures.items() if c.limited]
    return {"returncode": returncode, "timed_out": timed_out, "output_limit_reached": "+".join(limited) or None,
            "output_limit_bytes": output_limit_bytes,
            "stdout_bytes": captures["stdout"].seen, "stderr_bytes": captures["stderr"].seen,
            "capture_complete": not any(c.is_alive() for c in captures.values()),
            "wall_s": round(time.monotonic() - started, 3), "cleanup": cleanup}


def tail(path: Path, limit: int = 4000) -> str:
    try:
        with open(path, "rb") as handle:
            handle.seek(max(0, path.stat().st_size - limit))
            return handle.read().decode("utf-8", "replace")
    except OSError:
        return ""
