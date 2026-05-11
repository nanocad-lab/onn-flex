#SBATCH --job-name=onn_allzeros_noquant
#SBATCH --partition=hpg-b200
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=4
#SBATCH --cpus-per-task=4
#SBATCH --mem=64gb
#SBATCH --time=80:00:00
#SBATCH --output=onn_allzeros_noquant_%j.out

set -euo pipefail

if [[ -n "${SLURM_SUBMIT_DIR:-}" ]]; then
  REPO_ROOT="${SLURM_SUBMIT_DIR}"
else
  SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
  REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
fi
cd "${REPO_ROOT}"

BASE_OUT="runs/onehots_quantparam/base_allzeros_noquant_clamp"

module load conda/25.7.0
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate onn-torch

export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}"

echo "Training baseline (all-zeros, clamp-only no quant)"
echo "  output: ${BASE_OUT}"

srun python - <<'PY'
