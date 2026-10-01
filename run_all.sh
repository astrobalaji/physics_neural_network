#!/usr/bin/env bash
# Rehearsal: runs every notebook top to bottom and saves the outputs inside them,
# so the plots are already there if anything fails live. Also caches the FNO model and data.
set -e
cd "$(dirname "$0")"
source .venv/bin/activate
for nb in 01_tensors_and_PINN.ipynb 02_fourier_neural_operator.ipynb 03_world_model_vjepa2.ipynb; do
  echo ">> running $nb"
  jupyter nbconvert --to notebook --execute --inplace --ExecutePreprocessor.timeout=1800 \
      --ExecutePreprocessor.kernel_name=siet-demo "$nb"
done
echo ">> all notebooks ran successfully"
