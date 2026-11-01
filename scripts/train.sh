#!/usr/bin/env bash
#
# Launch training on one node: the per-node helper scripts/stage1.sh,
# stage2.sh and posttrain_real.sh call. With --nnodes above one every node runs
# it with the same arguments and its own --node-rank; the values go to torchrun
# unchecked.

set -euo pipefail

here=$( cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P )
repo=$( cd "$here/.." && pwd -P )
# shellcheck source=lib.sh
. "$here/lib.sh"

usage() {
    cat >&2 <<'EOF'
usage: train.sh --root DIR --corpus NAME --run-name NAME --backbone DIR
                --arch NAME --mode NAME --data NAME --schedule NAME
                [--output-root DIR] [--nproc N] [--workers N] [--prefetch N]
                [--metadata-dir DIR] [--manifest-dir DIR]
                [--nnodes N] [--node-rank I] [--master-addr HOST]
                [--master-port PORT]
                [--global-batch N] [--per-device-batch N] [--accumulation N]
                [--print-hydra]
                [-- OVERRIDE ...]

  --root         the data root; becomes NEO_VLA_DATA_ROOT
  --corpus       which prepared manifest (and normaliser) to read
  --metadata-dir the folder holding the corpus's embodiments.json, relative
                 to --root; default metadata/<corpus>
  --run-name     names the run directory. Launching again with the same name
                 resumes from the newest checkpoint in it
  --backbone     the backbone checkpoint; becomes NEO_VLA_BACKBONE
  --arch         skip28 (config/arch/skip28.yaml)
  --mode         frozen_vlm or cotrain (config/mode)
  --data         the data group under config/data
  --schedule     the schedule group under config/schedule
  --output-root  where run directories go; becomes NEO_VLA_OUTPUT_ROOT
  --nproc        processes on this node. 1 launches python, more launch torchrun
  --workers      loader workers per process; becomes NEO_VLA_DATALOADER_WORKERS
  --prefetch     batches each loader worker reads ahead; becomes
                 NEO_VLA_DATALOADER_PREFETCH. Unset, torch's default of 2

Multi-node. Every node runs this with the same arguments but its own
--node-rank; the process world size is --nnodes times --nproc. The values are
handed to torch.distributed.run unchecked:

  --nnodes       nodes in this launch, default 1
  --node-rank    this node's index, 0 to --nnodes minus 1, default 0
  --master-addr  the rank zero node, as the other nodes reach it; required
                 above one node
  --master-port  the rendezvous port, default 29500

Batch. The schedule leaves gradient_accumulation_steps null and the engine
derives it from the world size. Naming the batch here states the numbers on
the launch line and in the log rather than leaving them implied:

  --global-batch     the samples an optimiser step sees, across every node
  --per-device-batch the samples one process sees at once
  --accumulation     the gradient accumulation steps. Given with the two
                     above, all three are stated as they are; without it the
                     accumulation is what the two and the world size leave,
                     and a global batch they do not divide exactly stops the
                     launch here.

--global-batch and --per-device-batch go together, both or neither.

--print-hydra prints the hydra list this launch would start train.py with,
one 'HYDRA <override>' line each, and exits 0: it exports nothing and starts
nothing. A --root that does not exist yet is named as given.

Anything after -- is passed to hydra unchanged: hydra takes the last
override of a key, so an override of data.manifest, data.train.normaliser,
data.corpus or the batch keys replaces the value this line states. The node of
rank zero makes --output-root/--run-name with its .hydra/{config,hydra,
overrides}.yaml and train.log before torchrun starts (see below); an override
of run_name or of hydra.run.dir, output_subdir, job.name or job_logging moves
those files where nothing made them first.

--data names the loader configuration and --corpus the manifest it reads;
data.corpus is set from --corpus as well, since every group spells its
manifest and normaliser from it.
EOF
    exit 2
}

root=""
corpus=""
run_name=""
backbone=""
arch=""
mode=""
data=""
schedule=""
output_root=""
nproc=1
workers=""
prefetch=""
metadata_dir=""
manifest_dir="manifests"
nnodes=1
node_rank=0
master_addr=""
master_port=29500
global_batch=""
per_device_batch=""
accumulation_given=""
overrides=()
print_hydra=0

while [ "$#" -gt 0 ]; do
    case "$1" in
        --root) root="${2:-}"; shift 2 ;;
        --corpus) corpus="${2:-}"; shift 2 ;;
        --run-name) run_name="${2:-}"; shift 2 ;;
        --backbone) backbone="${2:-}"; shift 2 ;;
        --arch) arch="${2:-}"; shift 2 ;;
        --mode) mode="${2:-}"; shift 2 ;;
        --data) data="${2:-}"; shift 2 ;;
        --schedule) schedule="${2:-}"; shift 2 ;;
        --output-root) output_root="${2:-}"; shift 2 ;;
        --nproc) nproc="${2:-}"; shift 2 ;;
        --workers) workers="${2:-}"; shift 2 ;;
        --prefetch) prefetch="${2:-}"; shift 2 ;;
        --metadata-dir) metadata_dir="${2:-}"; shift 2 ;;
        --manifest-dir) manifest_dir="${2:-}"; shift 2 ;;
        --nnodes) nnodes="${2:-}"; shift 2 ;;
        --node-rank) node_rank="${2:-}"; shift 2 ;;
        --master-addr) master_addr="${2:-}"; shift 2 ;;
        --master-port) master_port="${2:-}"; shift 2 ;;
        --global-batch) global_batch="${2:-}"; shift 2 ;;
        --per-device-batch) per_device_batch="${2:-}"; shift 2 ;;
        --accumulation) accumulation_given="${2:-}"; shift 2 ;;
        --print-hydra) print_hydra=1; shift ;;
        --) shift; overrides=("$@"); break ;;
        -h|--help) usage ;;
        *) neo_say "unknown argument $1"; usage ;;
    esac
done

for pair in root:"$root" corpus:"$corpus" run-name:"$run_name" \
            backbone:"$backbone" arch:"$arch" mode:"$mode" \
            data:"$data" schedule:"$schedule"; do
    if [ -z "${pair#*:}" ]; then
        neo_say "--${pair%%:*} is required"
        usage
    fi
done

world_size=$(( nnodes * nproc ))

# The batch plan, before anything opens a file.
accumulation=""
if [ -n "$global_batch" ] || [ -n "$per_device_batch" ]; then
    if [ -z "$global_batch" ] || [ -z "$per_device_batch" ]; then
        neo_die "--global-batch '$global_batch' and --per-device-batch '$per_device_batch'" \
            "both of them, so the accumulation is the only number derived" \
            "pass the one that is missing, or neither and let the schedule decide"
    fi
    if [ -n "$accumulation_given" ]; then
        accumulation=$accumulation_given
    else
        span=$(( per_device_batch * world_size ))
        if [ "$(( global_batch % span ))" -ne 0 ]; then
            neo_die "global batch $global_batch over $per_device_batch x $world_size processes" \
                "a global batch the per-device batch and the world size divide exactly" \
                "change --nnodes, --nproc or --per-device-batch, or state --accumulation"
        fi
        accumulation=$(( global_batch / span ))
    fi
elif [ -n "$accumulation_given" ]; then
    neo_die "--accumulation $accumulation_given without --global-batch and --per-device-batch" \
        "the three batch numbers stated together" \
        "pass --global-batch and --per-device-batch too, or drop --accumulation"
fi

if [ "$nnodes" != 1 ] && [ -z "$master_addr" ]; then
    neo_die "--nnodes $nnodes without --master-addr" \
        "the address of rank zero, as the other nodes reach it" \
        "--master-addr (MASTER_ADDR of the launch)"
fi

# The corpus root as its physical path when it exists (the manifest's paths are
# relative to it); otherwise as given, and the runner's own open is what fails.
if [ -d "$root" ]; then root=$( cd "$root" && pwd -P ); else root="${root%/}"; fi
# A corpus's stage records and normaliser sit under metadata/<corpus>.
if [ -z "$metadata_dir" ]; then metadata_dir="metadata/$corpus"; fi
if [ -z "$output_root" ]; then output_root="$root/outputs"; fi
manifest="$root/$manifest_dir/$corpus.json"

neo_say "train"
neo_field root "$root"
neo_field corpus "$corpus"
neo_field manifest "$manifest_dir/$corpus.json"
neo_field run "$run_name in $output_root"
neo_field choices "arch=$arch mode=$mode data=$data schedule=$schedule"
neo_field backbone "$backbone"
if [ "$nnodes" != 1 ]; then
    neo_field node "rank $node_rank of $nnodes, $nproc processes, world $world_size"
    neo_field master "$master_addr:$master_port"
else
    neo_field nproc "$nproc"
fi
if [ -n "$accumulation" ]; then
    neo_field batch "$global_batch = $per_device_batch x $accumulation x $world_size"
else
    neo_field batch "from the schedule, accumulation derived from world $world_size"
fi

# The normaliser of --corpus is the one the run reads, stated here rather than
# left to the data group's pattern, and data.corpus goes with it.
hydra=(run_name="$run_name" arch="$arch" mode="$mode" data="$data"
       schedule="$schedule" data.corpus="$corpus" data.manifest="$manifest"
       data.train.normaliser="$root/$metadata_dir/embodiments.json")
if [ -n "$accumulation" ]; then
    hydra+=(schedule.global_batch="$global_batch"
            schedule.per_device_batch="$per_device_batch"
            schedule.gradient_accumulation_steps="$accumulation")
fi
# A pretrain schedule keeps an EMA (decay 0.999) unless the launch states
# trainer.ema_decay; the schedule file cannot carry it, since the runner reads
# only its schedule keys from there. Finetunes are left as stated.
if [ "$schedule" = "pretrain" ]; then
    _ema_given=0
    for _o in ${overrides[@]+"${overrides[@]}"}; do
        case "$_o" in trainer.ema_decay=*) _ema_given=1 ;; esac
    done
    if [ "$_ema_given" -eq 0 ]; then
        hydra+=(trainer.ema_decay=0.999)
        neo_field ema "0.999 (pretrain default; pass trainer.ema_decay=null to opt out)"
    fi
fi

if [ "${#overrides[@]}" -gt 0 ]; then hydra+=("${overrides[@]}"); fi

if [ "$print_hydra" = 1 ]; then
    for item in "${hydra[@]}"; do printf 'HYDRA %s\n' "$item"; done
    exit 0
fi

export NEO_VLA_DATA_ROOT="$root"
export NEO_VLA_BACKBONE="$backbone"
export NEO_VLA_OUTPUT_ROOT="$output_root"
if [ -n "$workers" ]; then export NEO_VLA_DATALOADER_WORKERS="$workers"; fi
if [ -n "$prefetch" ]; then export NEO_VLA_DATALOADER_PREFETCH="$prefetch"; fi

# Machine settings, not run settings. Network interfaces stay as the caller
# selected them; the descriptor limit is raised because each loader worker
# holds shared-memory descriptors for its in-flight batches.
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
export NCCL_DEBUG="${NCCL_DEBUG:-WARN}"
export TORCH_NCCL_ASYNC_ERROR_HANDLING="${TORCH_NCCL_ASYNC_ERROR_HANDLING:-1}"
if [ "$(ulimit -n)" -lt 65536 ]; then ulimit -n 65536; fi

# The run directory's hydra files exist before any process starts. Every rank
# hydra starts creates <run>/.hydra/{config,hydra,overrides}.yaml and opens
# <run>/train.log; on some shared filesystems concurrent creates of one path
# fail with EEXIST, so the node of rank zero makes all four before torchrun,
# whose agents meet at their rendezvous before any worker starts. touch keeps
# the bytes of a file that is there (a resume).
if [ "$node_rank" -eq 0 ]; then
    run_dir="$output_root/$run_name"
    if ! { mkdir -p "$run_dir/.hydra" &&
           touch "$run_dir/.hydra/config.yaml" "$run_dir/.hydra/hydra.yaml" \
                 "$run_dir/.hydra/overrides.yaml" "$run_dir/train.log"; }; then
        neo_die "cannot create the hydra files of $run_dir" \
            "a writable run directory, made here before any rank starts" \
            "check --output-root and the space under it"
    fi
    neo_field hydra "$run_dir/.hydra/{config,hydra,overrides}.yaml and train.log made before any rank starts"
fi

neo_say "launching ${hydra[*]}"
if [ "$nnodes" != 1 ]; then
    ( cd "$repo" && "$(neo_python)" -m torch.distributed.run \
        --nnodes="$nnodes" --node_rank="$node_rank" --nproc_per_node="$nproc" \
        --master_addr="$master_addr" --master_port="$master_port" \
        train.py "${hydra[@]}" )
elif [ "$nproc" = "1" ]; then
    ( cd "$repo" && "$(neo_python)" train.py "${hydra[@]}" )
else
    ( cd "$repo" && "$(neo_python)" -m torch.distributed.run \
        --standalone --nproc_per_node="$nproc" train.py "${hydra[@]}" )
fi

neo_say "run directory $output_root/$run_name"
neo_next "scripts/train.sh --root $root --corpus $corpus --run-name $run_name --backbone $backbone --arch $arch --mode $mode --data $data --schedule $schedule   # same name resumes"
