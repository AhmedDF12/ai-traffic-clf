# AI-Based Network Traffic Classification

Graduation project. This milestone reads captures, builds flows, fits scaling on the training split, and trains one model at a time. The models are logistic regression, a decision tree, a random forest, XGBoost, LightGBM, and CatBoost. Optuna searches the validation split only.

Application and service classification on ISCX VPN-nonVPN plus your own captures is the primary task. Benign-versus-malicious classification on CIC-IDS2017 is secondary and shares the same feature contract.

## Setup

The code targets Python 3.11 or newer. On this machine `python` is 3.10 and the `py -3.11` launcher points at a missing install, so the virtual environment in `.venv` was created with Python 3.14.

```powershell
.\.venv\Scripts\Activate.ps1
python -m pip install -U pip
pip install -e ".[dev]"
pytest
traffic-clf check --task service
traffic-clf check --task application
traffic-clf check --task security
```

XGBoost, LightGBM, CatBoost, and Optuna are installed for CPU training. Torch, SHAP, nfstream, and the API packages stay in the extras (`dl`, `boost`, `capture`, `viz`, `serve`) and are not required for this milestone. The `boost` extra also installs SHAP, which this step does not use.

`traffic-clf track-smoke` writes one local MLflow run to `mlruns/mlflow.db`. MLflow 3 no longer accepts a folder of run files, so the tracker is a SQLite file plus `mlruns/artifacts/`. Both are gitignored.

## What is guarded

- Ports and SNI cannot be turned on. They are flow keys or audit fields, not model features.
- Payloads are not stored. `privacy.strip_payload` must stay true.
- TLS metadata is off unless you later opt in for one named ablation.
- CIC labels are mapped through `configs/labels/`. An unknown label stops the load.
- ISCX labels come from the capture filename. `skype1.pcap` is rejected because chat, call, and file transfer are different classes.
- CIC-IDS2017 evaluation follows the day manifest in `configs/splits/cic_ids2017.yaml`. Monday is a false-positive check. Friday afternoon is test.
- File-level splits for ISCX use `assign_grouped_split`. A class with fewer than three capture files is rejected.

The checked-in timeouts match CICFlowMeter: active 120 seconds, idle 5 seconds, first 30 packets. For your own captures, pass `--overlay configs/presets/live.yaml` (idle 15 seconds). A free Colab session can use `--overlay configs/colab.yaml`. Colab is not required for this milestone, and its disk disappears when the runtime stops.

## Data you add later

Do not commit datasets. Put files here when you download them:

```text
data/raw/cic-ids2017/*.csv
data/raw/cic-ids2018/*.csv
data/raw/iscx/**/*.pcap
data/raw/custom/*.pcap
data/raw/custom/sessions.csv
```

CIC-IDS files are the official flow CSVs, not the raw PCAPs. Start with one day and `--max-rows` so a multi-gigabyte file is not loaded whole:

```powershell
traffic-clf load-cic data\raw\cic-ids2017\Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv --dataset cic_ids2017 --max-rows 1000
```

ISCX PCAPs are labeled with:

```powershell
traffic-clf map-iscx vpn_youtube_A.pcap email2b.pcap skype_chat1a.pcap
```

Copy `configs/examples/custom_sessions.csv` for self-capture. One host, one application at a time, and the session end must be after the start. Overlapping sessions on the same host are rejected.

## Label decisions

- The six service classes are streaming, voip, browsing, file transfer, chat, and gaming. Email and P2P are excluded from that head. Gaming exists only in self-capture.
- The ISCX mail captures are class `email`, because the traffic is SMTP/POP/IMAP rather than Gmail alone.
- Facebook and Hangouts video files are calls, so they are voip.
- CIC DoS tool names join `ddos`. Heartbleed, infiltration, XSS, and SQL injection join `malware` because those classes are tiny. The original string stays in `label_raw`.
- CSE-CIC-IDS2018 has no dedicated port-scan class. The official file also misspells infiltration as `Infilteration`. Both spellings map to `malware`.
- `Brute Force -XSS` is the web XSS attack, so it joins `malware`, not `brute_force`.

## Tests

```powershell
pytest
```

| Test module | What it locks |
| --- | --- |
| `tests/unit/test_config.py` | Task priority, port and payload guards, Colab and live overlays |
| `tests/unit/test_seeding.py` | Repeatable NumPy and Python draws |
| `tests/unit/test_tracking.py` | A local MLflow run stores the seed and `use_ports=false` |
| `tests/unit/test_labels.py` | ISCX filenames and CIC label collapse |
| `tests/unit/test_loaders.py` | CIC CSV dialects, identifier removal, custom sidecar rules |
| `tests/unit/test_splits.py` | Day split and file-level split |
| `tests/unit/test_flows.py` | PCAP to bidirectional flows, timeouts, and sidecar windows |
| `tests/unit/test_preprocess.py` | Train-only scaling, day split, and file split |
| `tests/unit/test_training.py` | Baselines fit on the training split and score the held-out files |
| `tests/unit/test_tuning.py` | Boosting models, and a search that scores validation rather than test |
| `tests/unit/test_cli.py` | The commands above |

## Flows and preprocessing

`extract-pcap` reads classic PCAP and single-section PCAPNG. Payload bytes are discarded after the header lengths are known. The first packet of a flow is the forward direction. Duration and inter-arrival times are microseconds, the same unit as CICFlowMeter. A zero-length flow gets a missing rate rather than infinity. Because a flow closes at the idle gap, each flow is one active period and its idle statistics are zero.

```powershell
traffic-clf extract-pcap --task service data\raw\iscx\vpn_youtube_A.pcap
traffic-clf prepare-pcap --task service --out data\processed\service data\raw\iscx\*.pcap
traffic-clf prepare-cic --max-rows 1000 --out data\processed\security data\raw\cic-ids2017\*.csv
```

`prepare-pcap` splits by capture file. A class with fewer than three files stops the run. Email and P2P flows are dropped from the service task and counted. `prepare-cic` uses the day manifest: Monday stays in `fp_check`, and the scaler is fit on the training days only. Protocol becomes `protocol_tcp`, `protocol_udp`, and `protocol_other`. It is not scaled as a number. Direction columns stay -1, 0, or 1.

The output directory contains `train_features.parquet`, `train_meta.parquet`, the same pair for validation, test, and `fp_check`, plus `preprocess.joblib`. Metadata still has addresses and ports. The feature file does not.

Self-captures need a sidecar. Flows that start outside its intervals are dropped, because a file also contains setup traffic.

```powershell
traffic-clf extract-pcap --task service --sidecar data\raw\custom\sessions.csv data\raw\custom\youtube_a.pcap
```

Use `--overlay configs/presets/live.yaml` when the idle timeout for your own captures should be 15 seconds. PCAPNG with multiple sections, and link types other than Ethernet, Linux cooked capture, and raw IP, are rejected.

## Baseline training

`train` reads a directory written by `prepare-pcap` or `prepare-cic`. It fits one of `logreg`, `decision_tree`, `random_forest`, `xgboost`, `lightgbm`, or `catboost` on the training features only. Baselines use balanced class weights. The boosting models use balanced sample weights instead. The validation split is reported and is not used to fit the model unless you run `tune`. A class listed in the task but missing from the training files is named in `classes_absent` and is left out of the macro average, so an ISCX-only service run is not punished for having no gaming captures.

```powershell
traffic-clf train --task service --model random_forest --data data\processed\service --out reports\service-rf
```

The output directory contains `model.joblib`, `metrics.json`, `confusion_test.csv`, and `report.txt`. The same scores are written to the local MLflow run. For the security task, Monday's `fp_check` file is scored as a false-positive rate: the share of benign flows predicted as an attack. That rate is not a model-selection score.

## Tuning

`tune` tries a small set of hyperparameters and keeps the one with the highest validation macro-F1. Each trial is fit on the training rows. The test split is scored once, after the winner is chosen. A random row fold is not used, because flows from the same capture would land on both sides of the score.

```powershell
traffic-clf tune --task service --model lightgbm --trials 20 --data data\processed\service --out reports\service-lgb
```

`trials.csv` lists every trial. Twenty trials is a short CPU search. Raise `--trials` when the prepared tables are the real ISCX or CIC files. An empty validation split stops the search. Boosting runs on CPU even when a Colab overlay asks for CUDA. That device is reserved for a later sequence model. LightGBM's leaf size is 1, so a small self-capture can still grow a split. XGBoost is trained on class indexes and the saved model predicts the class names.

## Google Colab

The full-run notebook is `notebooks/colab_full_run.ipynb`. It does not contain measured scores. A fresh Colab runtime starts empty, so the notebook clones the repository, installs it, and writes prepared tables and MLflow to Google Drive.

Open it from GitHub by substituting your account and repository:

```text
https://colab.research.google.com/github/ACCOUNT/REPOSITORY/blob/main/notebooks/colab_full_run.ipynb
```

Or upload the file to Drive and choose Open with Google Colab.

In the first code cell, set `GITHUB_REPO_URL` to `https://github.com/ACCOUNT/REPOSITORY.git`. In Colab, open the key icon and add secrets only if you need them:

| Secret | When |
| --- | --- |
| `GITHUB_TOKEN` | The repository is private. A public repository does not need it. |
| `KAGGLE_USERNAME` and `KAGGLE_KEY` | `DATA_SOURCE` is `kaggle`. Read the mirror's license before downloading it. |

The official CIC and ISCX files are behind a registration form. The notebook does not submit that form. Put the files on Drive, or set a direct URL you have checked yourself. Do not download the CIC PCAP days.

`configs/colab.yaml` points at `/content` and selects CUDA. The notebook writes `MyDrive/ai-traffic-clf/drive_overlay.yaml` instead, keeps boosting on CPU, and stores outputs under `/content/drive/MyDrive/ai-traffic-clf`. The Linux commands, after that overlay exists, are:

```bash
traffic-clf check --task service --root /content/ai-traffic-clf
traffic-clf prepare-cic --root /content/ai-traffic-clf --overlay /content/drive/MyDrive/ai-traffic-clf/drive_overlay.yaml --max-rows 20000 --out /content/drive/MyDrive/ai-traffic-clf/data/processed/security /content/drive/MyDrive/ai-traffic-clf/data/raw/cic-ids2017/*.csv
traffic-clf tune --root /content/ai-traffic-clf --task service --model lightgbm --trials 50 --overlay /content/drive/MyDrive/ai-traffic-clf/drive_overlay.yaml --data /content/drive/MyDrive/ai-traffic-clf/data/processed/service --out /content/drive/MyDrive/ai-traffic-clf/data/reports/service/lightgbm
```

Set `RUN_EXTRA = True` only after the main models exist. Idle-timeout ablation reads the captures again for 5, 15, 30, and 60 seconds. `RUN_NEURAL = True` stops, because the sequence networks are not in this repository. Finished output directories are skipped when `FORCE` is false.

## Not in this milestone

Neural nets, SHAP, ONNX, and the API are next. Outlier clipping is not applied: clipping byte rates on CIC-IDS would shrink DDoS flows toward the benign majority. SMOTE is not applied either. Synthetic flows can combine values no real flow ever had, so class weights and sample weights are the imbalance control until that comparison is run on purpose.
