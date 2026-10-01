#!/usr/bin/env bash
# One-time setup for the SIET demo notebooks (macOS).
# Usage:  cd ~/Documents/SIET && bash setup_mac.sh
set -e
cd "$(dirname "$0")"
PY=${PYTHON:-python3}
echo ">> Using $($PY --version)"
$PY -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
python -m ipykernel install --user --name siet-demo --display-name "Python 3 (SIET demo)"
echo ">> Downloading Hugging Face models (about 2.5 GB, one time only)"
python prefetch_models.py
echo ">> Done. Start with:  source .venv/bin/activate && jupyter lab"
