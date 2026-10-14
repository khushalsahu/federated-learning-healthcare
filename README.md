# Federated Learning for Privacy-Preserving Healthcare

A working prototype accompanying the project synopsis of the same name. Multiple
simulated hospitals collaboratively train a disease-prediction model using
**Federated Averaging (FedAvg)** — no hospital's raw patient data ever leaves
its own machine.

## What this demonstrates

- **Federated Learning pipeline** — global server + N hospital clients, built
  on [Flower](https://flower.ai)'s `NumPyClient` interface.
- **Privacy-preserving training** — each `HospitalClient` only ever reads its
  own CSV file. The server only ever receives model *weights*, never data.
- **Federated Averaging** — client weight updates are combined into a global
  model, weighted by each hospital's local dataset size.
- **Differential privacy option** — optional Gaussian noise added to client
  updates before they're sent, with a dashboard view of the resulting
  privacy–utility trade-off.
- **Baseline comparison** — a centralised model trained on pooled data, to
  measure what (if anything) is lost by not centralising patient data.
- **Dashboard** — Streamlit app visualising per-round convergence, per-hospital
  performance, and the federated-vs-centralised comparison.

## Project structure

```
fl_project/
├── data/
│   └── prepare_data.py       # loads dataset, creates hospital_A/B/C.csv + test.csv
├── src/
│   ├── model_utils.py        # build/get/set model weights, FedAvg averaging
│   ├── privacy.py            # differential-privacy noise mechanism
│   ├── fl_client.py          # HospitalClient (Flower NumPyClient)
│   ├── federated_simulation.py   # main FL training loop
│   └── baseline_centralized.py   # centralised baseline for comparison
├── dashboard/
│   └── dashboard.py          # Streamlit visualisation
├── results/                  # generated metrics + saved model weights (gitignored)
├── requirements.txt
├── run_all.sh                # runs the entire pipeline end-to-end
└── README.md
```

## Setup and one-command launch

From the project root in Linux, WSL, or another Bash-compatible environment, run:

```bash
./run_all.sh
```

The launcher will:
1. Create the local `venv/` virtual environment if it does not exist.
2. Install or verify the packages in `requirements.txt`.
3. Prepare the simulated hospital partitions and held-out test set.
4. Run federated training with and without the configured Gaussian noise, save both metric sets, and run the centralised baseline.
5. Launch the Streamlit dashboard and print its local URL. Press **Ctrl+C** in the terminal to stop it.

The first run requires internet access to install Python dependencies. On Ubuntu/WSL, if virtual-environment creation fails because `venv` support is missing, install it with:

```bash
sudo apt update
sudo apt install python3-venv
```

To activate the environment manually for development:

```bash
source venv/bin/activate
```

You can also run individual stages using the Python scripts directly; execute them from the project root so the expected `data/` and `results/` paths resolve correctly.

## Dataset

By default this uses scikit-learn's built-in **Breast Cancer Wisconsin**
dataset (569 patients, 30 diagnostic features, binary malignant/benign label)
since it ships with scikit-learn and needs no internet download — useful for
running this reliably on any machine, including a college lab PC.

To swap in the **Pima Indians Diabetes** or **UCI Heart Disease** dataset
(as named in the synopsis), download the CSV and edit `load_raw_dataset()` in
`data/prepare_data.py`:

```python
def load_raw_dataset():
    df = pd.read_csv("diabetes.csv")
    df = df.rename(columns={"Outcome": "target"})
    return df
```

Nothing else in the pipeline needs to change — partitioning, training, and
aggregation all work on any binary-classification tabular dataset with a
`target` column.

## How the model works (in plain terms)

The model is a logistic regression trained via `SGDClassifier` (scikit-learn).
Its parameters are just two NumPy arrays: `coef_` (feature weights) and
`intercept_` (bias). That's convenient for Federated Learning because
FedAvg only needs to know how to average lists of arrays — it doesn't care
what framework produced them. Each round:

1. Server sends the current `[coef_, intercept_]` to every hospital.
2. Each hospital calls `partial_fit()` on its own local data only.
3. Each hospital sends its updated `[coef_, intercept_]` back — optionally
   with Gaussian noise added first (`privacy.py`) for differential privacy.
4. Server computes a weighted average across all hospitals (`model_utils.average_weights`),
   weighted by how many patients each hospital trained on.
5. The averaged weights become the new global model for the next round.

## Note on the simulation mode

`federated_simulation.py` runs all hospitals in a single Python process for
simplicity and reproducibility — no networking, no extra infrastructure
required, so it runs identically on any machine. The `HospitalClient` class
is written against Flower's standard `NumPyClient` interface, so it is also
directly compatible with Flower's networked/distributed modes
(`flwr.simulation.start_simulation` or the `flwr run` CLI) if you want to
extend this into a genuinely distributed deployment — for example, one
Docker container per hospital communicating over a real network, optionally
orchestrated with Kubernetes and deployed on AWS. That extension is listed as
future scope in the project synopsis.

## Results (example run, 15 rounds, Breast Cancer dataset)

| Approach                          | Test Accuracy |
|------------------------------------|:-------------:|
| Federated (FedAvg, no DP)          | ~96.5%        |
| Federated (FedAvg, with DP noise σ=0.05) | ~96.5% |
| Centralised baseline (pooled data) | ~93.9%        |

Exact numbers will vary slightly by run since the SGD optimiser depends on
data order. Re-run `run_all.sh` to reproduce your own numbers for the report.

## Extending this project (future scope)

- **Non-IID partitioning** — currently hospitals get an IID (similarly
  distributed) split; real hospitals would have skewed class distributions.
  `data/prepare_data.py:partition_iid()` is the place to add a non-IID splitter.
- **Secure aggregation** — cryptographic aggregation so even the server cannot
  see any individual hospital's update, only the sum.
- **Containerisation** — wrap `federated_simulation.py`'s client loop into a
  small FastAPI service per hospital, run each in its own Docker container,
  and have them communicate over Flower's gRPC transport instead of in-process
  function calls.
- **Cloud deployment** — deploy the server and client containers on AWS
  (ECS/EC2), optionally orchestrated with Kubernetes.
