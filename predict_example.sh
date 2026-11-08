#!/bin/bash -l
#SBATCH --job-name=ir_picture
#SBATCH --output=logs/picture-%j.out
#SBATCH --error=logs/picture-%j.err
#SBATCH --time=01:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --gpus=1
#SBATCH --partition=gpu
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?Submit from the repository root}"
module load conda/latest
eval "$(conda shell.bash hook)"
conda activate "${CONDA_ENV:-ir-diffusion}"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
export PYTHONUNBUFFERED=1
python -c "import matplotlib, rdkit; import torch; assert torch.cuda.is_available(), 'GPU unavailable'"
PREDICTION_ROW="${PREDICTION_ROW:-$(python -c "import json; print(json.load(open('data/processed/splits.json'))['splits']['val'][0])")}"
PREDICTION_OUTPUT="predictions/example_${SLURM_JOB_ID:?}"
SPECTRUM_CHECKPOINT="${SPECTRUM_CHECKPOINT:-runs/spectrum/checkpoints/best.ckpt}"
DIFFUSION_CHECKPOINT="${DIFFUSION_CHECKPOINT:-runs/diffusion/checkpoints/best.ckpt}"
srun python -u predict_structure.py \
    --spectrum_checkpoint "$SPECTRUM_CHECKPOINT" \
    --diffusion_checkpoint "$DIFFUSION_CHECKPOINT" \
    --data data/processed/squirl.npz --row "$PREDICTION_ROW" --device cuda \
    --beam_size 5 --conformers 3 --n_steps 200 --output_dir "$PREDICTION_OUTPUT"
python -u plot_prediction.py --prediction_dir "$PREDICTION_OUTPUT"
if [[ -f make_viewer.py ]]; then
    python -u make_viewer.py --prediction_dir "$PREDICTION_OUTPUT"
    echo "Download $PREDICTION_OUTPUT/viewer.html and open it locally to rotate the molecule."
fi
echo "Open $PREDICTION_OUTPUT/result.png in the OnDemand file browser."
