#!/usr/bin/env bash
# Prepare data, run federated/centralized experiments, then launch the dashboard.
set -Eeuo pipefail

PROJECT_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$PROJECT_ROOT"

VENV_DIR="${VENV_DIR:-$PROJECT_ROOT/venv}"
PYTHON_BIN="${PYTHON_BIN:-python3}"

if [[ ! -x "$VENV_DIR/bin/python" ]]; then
    echo "== Creating project virtual environment =="
    if ! "$PYTHON_BIN" -m venv "$VENV_DIR"; then
        echo >&2 "ERROR: Could not create the virtual environment."
        echo >&2 "On Ubuntu/WSL, install venv support with:"
        echo >&2 "  sudo apt update && sudo apt install python3-venv"
        exit 1
    fi
fi

PYTHON="$VENV_DIR/bin/python"

echo "== Checking Python dependencies =="
"$PYTHON" -m pip install -r requirements.txt

mkdir -p results

echo
echo "== 1/4: Preparing hospital data partitions =="
"$PYTHON" data/prepare_data.py

echo
echo "== 2/4: Running federated training (FedAvg, no DP noise) =="
"$PYTHON" src/federated_simulation.py --rounds 15
cp results/fl_metrics.csv results/fl_metrics_no_dp.csv

echo
echo "== 3/4: Running federated training WITH differential privacy noise =="
"$PYTHON" src/federated_simulation.py --rounds 15 --dp-sigma 0.05
cp results/fl_metrics.csv results/fl_metrics_with_dp.csv
# Restore the no-DP metrics as the dashboard's default without retraining.
cp results/fl_metrics_no_dp.csv results/fl_metrics.csv

echo
echo "== 4/4: Running centralised baseline for comparison =="
"$PYTHON" src/baseline_centralized.py --epochs 15

echo
echo "Experiments complete. Starting the Streamlit dashboard."
echo "Open the Local URL printed below in your browser. Press Ctrl+C to stop."
exec "$PYTHON" -m streamlit run dashboard/dashboard.py --server.headless true
