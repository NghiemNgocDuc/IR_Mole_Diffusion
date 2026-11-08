#!/bin/bash
set -euo pipefail
STAGE="${1:?Expected spectrum or diffusion}"
shift
case "$STAGE" in spectrum|diffusion) ;; *) echo "Unknown stage: $STAGE" >&2; exit 2 ;; esac
module load conda/latest
eval "$(conda shell.bash hook)"
conda activate "${CONDA_ENV:-ir-diffusion}"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export PYTHONUNBUFFERED=1
python -c "import torch; assert torch.cuda.is_available(), 'No CUDA device: check allocation and PyTorch wheel'; print(torch.__version__, torch.cuda.get_device_name())"
EXTRA_ARGS=()
if [[ -n "${RESUME_CHECKPOINT:-}" ]]; then
    EXTRA_ARGS+=(--resume_from_checkpoint "$RESUME_CHECKPOINT")
fi
srun python -u "train_${STAGE}.py" --config "configs/${STAGE}.json" \
    --accelerator gpu "${EXTRA_ARGS[@]}" "$@"
