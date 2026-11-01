# Helpers shared by train.sh and the stage launchers. Sourced, never run.
#
# Reporting goes to stderr. The only machine setting read here besides the
# launch environment documented in README.md is NEO_VLA_PYTHON.

neo_python() {
    printf '%s' "${NEO_VLA_PYTHON:-python}"
}

# Three lines: the value, what was expected, what to change.
neo_die() {
    printf 'error: %s\n' "$1" >&2
    if [ "$#" -ge 2 ]; then printf '  expected %s\n' "$2" >&2; fi
    if [ "$#" -ge 3 ]; then printf '  fix %s\n' "$3" >&2; fi
    exit 2
}

neo_say() {
    printf 'neo: %s\n' "$1" >&2
}

neo_field() {
    printf '  %-10s %s\n' "$1" "$2" >&2
}

neo_next() {
    printf '\nneo: next\n  %s\n' "$1" >&2
}

# --- stage launchers (stage1.sh, stage2.sh, posttrain_real.sh) ---------------

# neo_stage_args "$@": sets NEO_PRINT_ARGV (1 for --print-argv) and NEO_EXTRA
# (hydra overrides after --, appended last so they win).
neo_stage_args() {
    NEO_PRINT_ARGV=0
    NEO_EXTRA=()
    while [ "$#" -gt 0 ]; do
        case "$1" in
            --print-argv) NEO_PRINT_ARGV=1; shift ;;
            --) shift; NEO_EXTRA=("$@"); break ;;
            -h|--help) sed -n '2,/^set -euo/p' "$0" | sed '$d; s/^# \{0,1\}//' >&2; exit 0 ;;
            *) neo_die "unknown argument $1" "--print-argv, --help, or -- HYDRA_OVERRIDE ..." \
                   "see bash $0 --help" ;;
        esac
    done
}

# neo_need NAME...: each variable must be set. Under --print-argv a missing one
# prints as <NAME> so the command can be inspected before any path exists.
neo_need() {
    local name
    for name in "$@"; do
        if [ -z "${!name:-}" ]; then
            [ "$NEO_PRINT_ARGV" = 1 ] || neo_die "$name is not set" "$name in the environment" \
                "export $name (README.md lists every input)"
            printf -v "$name" '<%s>' "$name"
        fi
    done
}

# neo_shape: the distributed settings of this node, from the environment.
neo_shape() {
    NNODES=${NNODES:-1}
    NODE_RANK=${NODE_RANK:-0}
    GPUS_PER_NODE=${GPUS_PER_NODE:-8}
    MASTER_PORT=${MASTER_PORT:-29500}
    if [ "$NNODES" != 1 ]; then neo_need MASTER_ADDR; fi
    NEO_WORLD=$(( NNODES * GPUS_PER_NODE ))
}

# neo_accumulation GLOBAL PER_DEVICE: the accumulation the world size leaves.
neo_accumulation() {
    local span=$(( $2 * NEO_WORLD ))
    if [ "$(( $1 % span ))" -ne 0 ]; then
        neo_die "global batch $1 over $2 per device x world $NEO_WORLD" \
            "a global batch the per-device batch and the world size divide exactly" \
            "change NNODES, GPUS_PER_NODE or PER_DEVICE_BATCH"
    fi
    printf '%s' $(( $1 / span ))
}

# neo_launch TRAIN_ARGS...: with NEO_ENV (an array of NAME=VALUE), prints the
# node's environment, the train.sh argv and the hydra list it composes
# (--print-argv), or exports the environment and runs train.sh.
neo_launch() {
    local word line="" seen=0
    local args=()
    if [ "$NEO_PRINT_ARGV" = 1 ]; then
        for word in ${NEO_ENV[@]+"${NEO_ENV[@]}"}; do line="$line${line:+ }$(printf '%q' "$word")"; done
        printf 'ENV %s\n' "$line"
        line=""
        for word in bash scripts/train.sh "$@"; do line="$line${line:+ }$(printf '%q' "$word")"; done
        printf 'ARGV %s\n' "$line"
        for word in "$@"; do
            if [ "$word" = "--" ] && [ "$seen" = 0 ]; then args+=(--print-hydra --); seen=1
            else args+=("$word"); fi
        done
        [ "$seen" = 1 ] || args+=(--print-hydra)
        bash "$NEO_REPO/scripts/train.sh" "${args[@]}"
        return
    fi
    for word in ${NEO_ENV[@]+"${NEO_ENV[@]}"}; do export "${word?}"; done
    cd "$NEO_REPO" && exec bash scripts/train.sh "$@"
}
