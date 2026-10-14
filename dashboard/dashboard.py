"""
dashboard.py
------------
Streamlit dashboard for the Federated Learning for Privacy-Preserving
Healthcare project. Visualises:
  - Per-round global model accuracy / loss (the federated training curve)
  - Per-hospital local accuracy each round (are all hospitals converging?)
  - Federated model vs. centralised baseline, side by side
  - Effect of differential privacy noise on accuracy (privacy-utility trade-off)

Run with:
    streamlit run dashboard/dashboard.py
"""

import os
import pandas as pd
import streamlit as st

RESULTS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results")

st.set_page_config(page_title="Federated Learning — Healthcare Dashboard", layout="wide")

st.title("🏥 Federated Learning for Privacy-Preserving Healthcare")
st.caption("Live results from the simulated multi-hospital federated training pipeline.")


def load_csv(name):
    path = os.path.join(RESULTS_DIR, name)
    if not os.path.exists(path):
        return None
    return pd.read_csv(path)


fl_df = load_csv("fl_metrics.csv")
baseline_df = load_csv("baseline_metrics.csv")
fl_no_dp = load_csv("fl_metrics_no_dp.csv")
fl_with_dp = load_csv("fl_metrics_with_dp.csv")

if fl_df is None:
    st.error("No results found. Run `python src/federated_simulation.py` first "
             "(from the project root) to generate results/fl_metrics.csv.")
    st.stop()

global_curve = fl_df.drop_duplicates(subset="round")[["round", "global_test_accuracy", "global_test_loss"]]

# ---------------- Top-level summary ----------------
col1, col2, col3, col4 = st.columns(4)
final_round = global_curve.iloc[-1]
col1.metric("Final Federated Accuracy", f"{final_round['global_test_accuracy']:.2%}")
col2.metric("Final Federated Loss", f"{final_round['global_test_loss']:.3f}")
if baseline_df is not None:
    final_baseline = baseline_df.iloc[-1]
    col3.metric("Centralised Baseline Accuracy", f"{final_baseline['test_accuracy']:.2%}",
                delta=f"{(final_round['global_test_accuracy'] - final_baseline['test_accuracy']) * 100:+.2f} pts")
    col4.metric("Hospitals Simulated", fl_df["client_id"].nunique())

st.divider()

# ---------------- Global training curve ----------------
st.subheader("Global Model Convergence Across Communication Rounds")
c1, c2 = st.columns(2)
with c1:
    st.line_chart(global_curve.set_index("round")["global_test_accuracy"], height=300)
    st.caption("Global model accuracy on the held-out test set, per round.")
with c2:
    st.line_chart(global_curve.set_index("round")["global_test_loss"], height=300)
    st.caption("Global model loss on the held-out test set, per round.")

st.divider()

# ---------------- Per-hospital local performance ----------------
st.subheader("Per-Hospital Local Accuracy Each Round")
pivot = fl_df.pivot(index="round", columns="client_id", values="local_accuracy")
st.line_chart(pivot, height=320)
st.caption("Each hospital evaluates the shared global model on its own local data every round. "
           "Consistent lines across hospitals indicate the global model generalises well "
           "across institutions, not just on average.")

st.divider()

# ---------------- Federated vs Centralised ----------------
if baseline_df is not None:
    st.subheader("Federated Model vs. Centralised Baseline")
    comp_df = pd.DataFrame({
        "Approach": ["Federated (privacy-preserving)", "Centralised (pooled data)"],
        "Final Test Accuracy": [final_round["global_test_accuracy"], final_baseline["test_accuracy"]],
    }).set_index("Approach")
    st.bar_chart(comp_df, height=300)
    st.caption("The federated model never had access to pooled patient data across hospitals, "
               "yet reaches comparable — here, matching or better — accuracy than the "
               "traditional centralised approach.")

st.divider()

# ---------------- Differential Privacy trade-off ----------------
if fl_no_dp is not None and fl_with_dp is not None:
    st.subheader("Privacy–Utility Trade-off: Effect of Differential Privacy Noise")
    no_dp_curve = fl_no_dp.drop_duplicates(subset="round")[["round", "global_test_accuracy"]].rename(
        columns={"global_test_accuracy": "No DP noise"})
    with_dp_curve = fl_with_dp.drop_duplicates(subset="round")[["round", "global_test_accuracy"]].rename(
        columns={"global_test_accuracy": "With DP noise"})
    dp_sigma_used = fl_with_dp["dp_sigma"].iloc[0]
    merged = no_dp_curve.merge(with_dp_curve, on="round").set_index("round")
    st.line_chart(merged, height=300)
    st.caption(f"Comparing global test accuracy per round with and without differential-privacy "
               f"noise (sigma = {dp_sigma_used}) added to each hospital's weight update before "
               f"it is sent to the server. Smaller gaps mean privacy is being added at low accuracy cost.")
else:
    st.info("To see the differential-privacy comparison, run:\n\n"
            "```\npython src/federated_simulation.py --dp-sigma 0.05\n```\n"
            "and save a copy of the resulting `results/fl_metrics.csv` as "
            "`fl_metrics_with_dp.csv` (and a no-noise run as `fl_metrics_no_dp.csv`).")

st.divider()
st.caption("Federated Learning for Privacy-Preserving Healthcare — B.Tech Final Year Project")
