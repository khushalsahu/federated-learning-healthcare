#!/usr/bin/env bash
# Runs the full pipeline end-to-end: data prep -> federated training ->
# centralised baseline -> (optional) DP comparison -> dashboard.
set -e

echo "== 1/4: Preparing hospital data partitions =="
python3 data/prepare_data.py

echo ""
echo "== 2/4: Running federated training (FedAvg, no DP noise) =="
python3 src/federated_simulation.py --rounds 15
cp results/fl_metrics.csv results/fl_metrics_no_dp.csv

echo ""
echo "== 3/4: Running federated training WITH differential privacy noise =="
python3 src/federated_simulation.py --rounds 15 --dp-sigma 0.05
cp results/fl_metrics.csv results/fl_metrics_with_dp.csv
# restore the plain run as the "main" results file the dashboard opens by default
python3 src/federated_simulation.py --rounds 15

echo ""
echo "== 4/4: Running centralised baseline for comparison =="
python3 src/baseline_centralized.py --epochs 15

echo ""
echo "All done. Launch the dashboard with:"
echo "    streamlit run dashboard/dashboard.py"
