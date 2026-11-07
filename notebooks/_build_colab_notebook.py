"""Build notebooks/colab_full_run.ipynb. Not part of the training pipeline."""

from __future__ import annotations

import json
from pathlib import Path

CELLS: list[tuple[str, str]] = []


def add(kind: str, source: str) -> None:
    CELLS.append((kind, source.strip("\n") + "\n"))


add(
    "markdown",
    """
# Full run on Google Colab

This notebook runs the traffic-classification project on a fresh Colab runtime. It does not contain results. Every table is computed when you run the cell. A failed check raises an error and later cells do not run.

The default path downloads the public VLC captures from Zenodo onto Drive, keeps any ISCX or CIC files already on Drive, prepares service and application, fits a random forest, and scores the test split once. Set `DOWNLOAD_KAGGLE = True` only when Drive has about 27 GB free and the Kaggle secrets are set. Set `RUN_SEARCH = True` for the 50-trial booster search. That search scores the test split itself, so leave it false after the random-forest test score exists.

`GITHUB_REPO_URL` is the private project repository. Add `GITHUB_TOKEN` in Colab Secrets before the clone cell. Tokens are read from Secrets, never written into the notebook.

`configs/colab.yaml` in the repository points at `/content` and asks for CUDA. This notebook writes its own overlay on Drive: boosting stays on CPU, and prepared tables and MLflow stay on Drive so a disconnect does not wipe them.
""",
)

add(
    "markdown",
    """
## 1. Setup

The next cell holds the switches. The cell after it prints the CPU or GPU, the RAM, and `df -h`, then mounts Drive and creates `PROJECT_DIR`, `DATA_DIR`, and `WORK_DIR`.

Expect a mounted Drive path and a free-space line. The notebook stops if Drive does not mount.
""",
)

add(
    "code",
    r'''
# Private repository. The clone cell reads GITHUB_TOKEN from Colab Secrets.
GITHUB_REPO_URL = "https://github.com/AhmedDF12/ai-traffic-clf.git"

# "drive": use files already under DATA_DIR/raw, then add the VLC download below.
# "kaggle": also download the ISCX captures and CIC CSVs. Needs ~27 GB free and Kaggle secrets.
# "url": download DIRECT_URLS. A CIC PCAP-day URL is refused.
DATA_SOURCE = "drive"
DOWNLOAD_VLC = True
# The packet reader loads a whole capture into RAM. Larger files are skipped, not extracted.
MAX_PCAP_BYTES = 500 * 1024 * 1024
# Each capture keeps at most this many flows before the capped experiment. The uncapped experiment is also reported.
MAX_FLOWS_PER_CAPTURE = 2000
# Listed sizes are the Kaggle API totalBytes values read on 2026-11-06.
KAGGLE_DATASETS = {
    "iscx": "caiyafang/iscx-vpn-nonvpn-2016-complete-pcaps",
    "cic_csv": "shadman1028/cicids2017-official-flow-feature-csv-files",
}
KAGGLE_LISTED_BYTES = {
    "caiyafang/iscx-vpn-nonvpn-2016-complete-pcaps": 26256519566,
    "shadman1028/cicids2017-official-flow-feature-csv-files": 240838240,
}
DIRECT_URLS = {
    # "cic_csv": "https://example.invalid/MachineLearningCSV.zip",
    # "iscx_zip": "https://example.invalid/ISCX-VPN-NonVPN-2016.zip",
}

FORCE = False
SUBSET_MAX_ROWS = None  # set an integer such as 20000 for a first CIC pass
TRIALS = 50
RUN_SEARCH = False  # True runs the 50-trial booster search and scores test there
RUN_EXTRA = False  # ablations, five seeds, cross-dataset, SHAP
RUN_NEURAL = False

if not GITHUB_REPO_URL.startswith("https://") or "github.com" not in GITHUB_REPO_URL:
    raise RuntimeError("Set GITHUB_REPO_URL to your https://github.com/... repository. No clone was started.")
if DATA_SOURCE not in {"drive", "kaggle", "url"}:
    raise RuntimeError("DATA_SOURCE must be drive, kaggle, or url.")
if RUN_SEARCH and TRIALS < 50:
    raise RuntimeError("TRIALS is below 50. The boosting search was not started.")
if MAX_PCAP_BYTES < 1_000_000:
    raise RuntimeError("MAX_PCAP_BYTES is too small to hold a capture. Nothing was downloaded.")
''',
)

add(
    "code",
    r'''
import os
import platform
import shutil
import subprocess
import sys
from pathlib import Path

from IPython.display import display

try:
    import google.colab  # noqa: F401
except ImportError as exc:
    raise RuntimeError("This notebook runs on Google Colab. Do not run it on the Windows machine.") from exc

print("python", sys.version)
print("platform", platform.platform())
if shutil.which("nvidia-smi") is None:
    print("gpu none")
else:
    gpu = subprocess.run(["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader"], capture_output=True, text=True)
    if gpu.returncode == 0 and gpu.stdout.strip():
        print("gpu", gpu.stdout.strip())
    else:
        print("gpu none")
mem = Path("/proc/meminfo").read_text(encoding="utf-8").splitlines()
print(next(line for line in mem if line.startswith("MemTotal")))
print(next(line for line in mem if line.startswith("MemAvailable")))
subprocess.run(["df", "-h", "/content"], check=True)

from google.colab import drive

drive.mount("/content/drive")
DRIVE_ROOT = Path("/content/drive/MyDrive/ai-traffic-clf")
PROJECT_DIR = Path("/content/ai-traffic-clf")
DATA_DIR = DRIVE_ROOT / "data"
WORK_DIR = Path("/content/work")
RAW_DIR = DATA_DIR / "raw"
for path in (DRIVE_ROOT, DATA_DIR, DATA_DIR / "processed", DATA_DIR / "reports", DATA_DIR / "mlruns", DATA_DIR / "inventory", WORK_DIR, RAW_DIR / "iscx", RAW_DIR / "cic-ids2017", RAW_DIR / "custom"):
    path.mkdir(parents=True, exist_ok=True)
print("PROJECT_DIR", PROJECT_DIR)
print("DATA_DIR", DATA_DIR)
print("WORK_DIR", WORK_DIR)
print("runtime_free_gb", round(shutil.disk_usage("/content").free / 1024**3, 2))
print("drive_free_gb", round(shutil.disk_usage("/content/drive").free / 1024**3, 2))


def section_end(name, ran, paths, suspicious, nxt):
    lines = [
        f"## {name}",
        "Ran: " + ran,
        "Paths: " + ", ".join(str(path) for path in paths),
        "Suspicious: " + suspicious,
        "Next: " + nxt,
        "",
    ]
    text = "\n".join(lines)
    print(text)
    report = DATA_DIR / "RESULTS.md"
    with report.open("a", encoding="utf-8") as handle:
        handle.write(text)
    return report
''',
)

add(
    "markdown",
    """
## 2. Get the code

Clones `GITHUB_REPO_URL`. If `GITHUB_TOKEN` is set in Colab Secrets, a private repository can be cloned. The token is removed from `origin` after the clone so it is not left in `.git/config`. An existing checkout is updated with `git pull`.

Then the project is installed in editable mode with the boosting extra. Expect package versions and a `traffic-clf check` printout. The command stops if the CLI does not run.
""",
)

add(
    "code",
    r'''
import os
import subprocess
from pathlib import Path

from google.colab import userdata

token = ""
try:
    token = userdata.get("GITHUB_TOKEN")
except Exception:
    token = ""

if PROJECT_DIR.joinpath(".git").exists():
    subprocess.run(["git", "pull", "--ff-only"], cwd=PROJECT_DIR, check=True)
else:
    clone_url = GITHUB_REPO_URL
    if token:
        clone_url = GITHUB_REPO_URL.replace("https://", f"https://x-access-token:{token}@", 1)
    subprocess.run(["git", "clone", clone_url, str(PROJECT_DIR)], check=True)
    subprocess.run(["git", "remote", "set-url", "origin", GITHUB_REPO_URL], cwd=PROJECT_DIR, check=True)
print(subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=PROJECT_DIR, text=True).strip())
section_end("code", "git clone or git pull", [PROJECT_DIR], "none yet", "install the package")
''',
)

add(
    "code",
    r'''
import subprocess
import sys

subprocess.run([sys.executable, "-m", "pip", "install", "-q", "-e", str(PROJECT_DIR) + "[boost]"], check=True)
''',
)

add(
    "code",
    r'''
import importlib
import shutil
import subprocess
import sys

os.chdir(PROJECT_DIR)
src = str(PROJECT_DIR / "src")
if src not in sys.path:
    sys.path.insert(0, src)
import traffic_clf
print("traffic_clf", traffic_clf.__file__)
for name in ("numpy", "pandas", "sklearn", "xgboost", "lightgbm", "catboost", "optuna", "mlflow"):
    module = importlib.import_module(name if name != "sklearn" else "sklearn")
    print(f"{name} {module.__version__}")
cli = shutil.which("traffic-clf")
if cli is None:
    raise RuntimeError("traffic-clf is not on PATH after installation.")
print(subprocess.check_output([cli, "check", "--task", "service", "--root", str(PROJECT_DIR)], text=True))
section_end("install", "pip install -e .[boost] and traffic-clf check --task service", [PROJECT_DIR], "none yet", "place or download the datasets")
''',
)

add(
    "markdown",
    """
## 3. Get the data

`DOWNLOAD_VLC` fetches the public Valencia VLC captures from Zenodo record 15121418 (CC BY 4.0; cite Francisco Rau, Carlos Herranz, Iñaki Val, and Joaquin Perez, DOI 10.5281/zenodo.15121418). YouTube, Netflix, and Spotify become streaming, the `web_` files become browsing, and Roblox becomes gaming. Prime Video and Teams are left out because those application names are not in the label map. Files larger than `MAX_PCAP_BYTES` are left out because the reader loads a whole capture into memory. The cell writes `sessions.csv` for the files it keeps.

The official file URLs on `cicresearch.ca` redirect to the UNB dataset index and return HTML, so this notebook does not download from them. With `DATA_SOURCE = "kaggle"` it also downloads two public Kaggle copies straight into Google Drive:

- ISCX packet captures: `caiyafang/iscx-vpn-nonvpn-2016-complete-pcaps` (Kaggle lists 26,256,519,566 bytes). The page says these are the original PCAP archives and that the license stays with the CIC authors. Cite Gerard Draper-Gil, Arash Habibi Lashkari, Mohammad Mamun, and Ali A. Ghorbani, ICISSP 2016.
- CIC-IDS2017 flow CSVs only: `shadman1028/cicids2017-official-flow-feature-csv-files` (Kaggle lists 240,838,240 bytes; the eight `WorkingHours` CSV names). The Kaggle page lists CC BY 4.0. Cite Iman Sharafaldin, Arash Habibi Lashkari, and Ali A. Ghorbani, ICISSP 2018. The CIC PCAP days are not in this download.

Kaggle is optional. Add `KAGGLE_USERNAME` and `KAGGLE_KEY` in Colab Secrets only when `DATA_SOURCE` is `kaggle`. The key is created at https://www.kaggle.com/settings/account. Drive needs room for both listings, about 26.5 GB, so a 15 GB Drive account will stop before that download. Files already present are skipped. Security runs only when the CIC CSVs are already on Drive or that Kaggle download ran. Service and application still train from the VLC captures.
""",
)

add(
    "code",
    r'''
import hashlib
import json
import shutil
import subprocess
import zipfile
from pathlib import Path

import pandas as pd
from tqdm.auto import tqdm

ISCX_PUBLISHED_BYTES = 28 * 1024**3
MARGIN_BYTES = 5 * 1024**3


def free_bytes(path):
    return shutil.disk_usage(path).free


def require_space(path, needed, reason):
    free = free_bytes(path)
    print(f"free_bytes {path} {free} needed {needed} reason {reason}")
    if free < needed:
        raise RuntimeError(f"Not enough free space on {path} for {reason}. Free {free} bytes, need {needed} bytes. Free space or pick another drive. Nothing was downloaded.")


def md5_file(path):
    digest = hashlib.md5()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_checksum(path):
    sidecar = path.with_suffix(path.suffix + ".md5")
    if not sidecar.exists():
        print(f"no_checksum {path.name} size_bytes {path.stat().st_size}")
        return
    expected = sidecar.read_text(encoding="utf-8").split()[0].strip().lower()
    actual = md5_file(path)
    if actual != expected:
        raise RuntimeError(f"Checksum mismatch for {path.name}. expected {expected} actual {actual}")
    print(f"checksum_ok {path.name}")


if DATA_SOURCE == "drive":
    print("Using files already on Drive. Nothing will be downloaded.")
elif DATA_SOURCE == "kaggle":
    import io
    from google.colab import userdata
    username = userdata.get("KAGGLE_USERNAME")
    key = userdata.get("KAGGLE_KEY")
    if not username or not key:
        raise RuntimeError("Add KAGGLE_USERNAME and KAGGLE_KEY in Colab Secrets. Create the key at https://www.kaggle.com/settings/account. Nothing was downloaded.")
    kaggle_dir = Path("/root/.kaggle")
    kaggle_dir.mkdir(parents=True, exist_ok=True)
    secret = kaggle_dir / "kaggle.json"
    secret.write_text(json.dumps({"username": username, "key": key}), encoding="utf-8")
    secret.chmod(0o600)
    needed = sum(KAGGLE_LISTED_BYTES[slug] for slug in KAGGLE_DATASETS.values())
    print("kaggle_listed_bytes", needed)
    print("Drive must hold the ISCX captures and the CIC CSVs. A 15 GB Drive account is too small. The CIC PCAP days are not downloaded.")
    require_space(DATA_DIR, needed, "the Kaggle ISCX captures and CIC CSV listing sizes")
    require_space(Path("/content"), 8 * 1024**3, "one capture at a time before it is moved to Drive")
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "kaggle"], check=True)

    def kaggle_files(slug):
        completed = subprocess.run(["kaggle", "datasets", "files", "-d", slug, "--csv"], capture_output=True, text=True)
        if completed.returncode != 0:
            detail = (completed.stderr or completed.stdout or "").strip()[-400:]
            raise RuntimeError(f"Could not list {slug}. Open https://www.kaggle.com/datasets/{slug} while logged in, then rerun. {detail}")
        frame = pd.read_csv(io.StringIO(completed.stdout))
        if "name" not in frame.columns:
            raise RuntimeError(f"Kaggle file list for {slug} has no name column: {list(frame.columns)}")
        return frame

    def destination_for(kind, relative_name):
        relative = Path(str(relative_name).lstrip("/"))
        suffix = relative.suffix.lower()
        if kind == "iscx" and suffix in {".pcap", ".pcapng"}:
            return RAW_DIR / "iscx" / relative
        if kind == "iscx" and relative.name == "SHA256SUMS.txt":
            return DATA_DIR / "inventory" / "iscx-SHA256SUMS.txt"
        if kind == "cic_csv" and suffix == ".csv":
            return RAW_DIR / "cic-ids2017" / relative.name
        if kind == "cic_csv" and suffix in {".pcap", ".pcapng"}:
            raise RuntimeError(f"Refusing {relative.name}. This notebook downloads CIC CSVs only.")
        return None

    def sha256_file(path):
        digest = hashlib.sha256()
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
        return digest.hexdigest()

    inventory_dir = DATA_DIR / "inventory"
    inventory_dir.mkdir(parents=True, exist_ok=True)
    for kind, slug in KAGGLE_DATASETS.items():
        listing = kaggle_files(slug)
        kept = []
        for row in listing.itertuples(index=False):
            record = row._asdict()
            target = destination_for(kind, record["name"])
            listed_size = record.get("size")
            if target is not None:
                kept.append((str(record["name"]), None if pd.isna(listed_size) else int(listed_size), target))
        if not kept:
            raise RuntimeError(f"{slug} listed no files this notebook can use.")
        for name, listed_size, target in tqdm(kept, desc=slug):
            if target.exists() and not FORCE and (listed_size is None or target.stat().st_size == listed_size):
                continue
            staging = WORK_DIR / "kaggle-one"
            if staging.exists():
                shutil.rmtree(staging)
            staging.mkdir(parents=True)
            completed = subprocess.run(
                ["kaggle", "datasets", "download", "-d", slug, "-f", name, "-p", str(staging), "--quiet"],
                capture_output=True, text=True,
            )
            if completed.returncode != 0:
                detail = (completed.stderr or completed.stdout or "").strip()[-400:]
                raise RuntimeError(f"Download failed for {name} from {slug}. Files already stored on Drive were kept. {detail}")
            downloaded = [item for item in staging.rglob("*") if item.is_file()]
            if len(downloaded) != 1:
                raise RuntimeError(f"Expected one downloaded file for {name}, found {[item.name for item in downloaded]}.")
            payload = downloaded[0]
            if payload.suffix.lower() == ".zip":
                with zipfile.ZipFile(payload) as archive:
                    members = [item for item in archive.namelist() if not item.endswith("/")]
                    if len(members) != 1:
                        raise RuntimeError(f"{name} unpacked to {members}. The zip was left in {staging} and was not copied to Drive.")
                    archive.extract(members[0], staging / "unpacked")
                payload = staging / "unpacked" / members[0]
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.move(str(payload), target)
            shutil.rmtree(staging, ignore_errors=True)
            if listed_size is not None and target.stat().st_size != listed_size:
                raise RuntimeError(f"Size mismatch for {target.name}. Kaggle lists {listed_size} bytes, Drive has {target.stat().st_size} bytes.")
        print("stored", kind, "files", len(kept), "under", RAW_DIR)
    checksum_path = DATA_DIR / "inventory" / "iscx-SHA256SUMS.txt"
    if checksum_path.exists():
        expected = {}
        for line in checksum_path.read_text(encoding="utf-8", errors="replace").splitlines():
            parts = line.split()
            if len(parts) >= 2 and len(parts[0]) == 64:
                expected[Path(parts[-1].lstrip("*")).name] = parts[0].lower()
        checked = 0
        for capture in (RAW_DIR / "iscx").rglob("*"):
            if not capture.is_file() or capture.suffix.lower() not in {".pcap", ".pcapng"}:
                continue
            digest = expected.get(capture.name)
            if digest is None:
                print("no_sha256_line", capture.name)
                continue
            actual = sha256_file(capture)
            if actual != digest:
                raise RuntimeError(f"SHA256 mismatch for {capture.name}. expected {digest} actual {actual}")
            checked += 1
        print("sha256_checked", checked)
    else:
        print("ISCX SHA256SUMS.txt was not in the Kaggle file list. Sizes were checked against the listing. No checksum was invented.")
elif DATA_SOURCE == "url":
    import urllib.request
    for name, url in DIRECT_URLS.items():
        lowered = url.lower()
        if "pcap" in lowered and "csv" not in lowered:
            raise RuntimeError(f"Refusing {name}. CIC PCAP days are not downloaded. Use MachineLearningCSV only.")
        destination = WORK_DIR / Path(url).name
        if name.startswith("iscx"):
            require_space(Path("/content"), ISCX_PUBLISHED_BYTES + MARGIN_BYTES, "ISCX zip plus extract room")
        if destination.exists() and not FORCE:
            print("skip download", destination)
        else:
            urllib.request.urlretrieve(url, destination)
        verify_checksum(destination)
        if destination.suffix == ".zip":
            unpack = WORK_DIR / "unpack"
            unpack.mkdir(parents=True, exist_ok=True)
            with zipfile.ZipFile(destination) as archive:
                members = [item for item in archive.namelist() if item.lower().endswith((".pcap", ".pcapng", ".csv"))]
                if not members:
                    raise RuntimeError(f"{destination.name} has no pcap or csv members.")
                archive.extractall(unpack, members=members)
            for unpacked in unpack.rglob("*"):
                if not unpacked.is_file():
                    continue
                suffix = unpacked.suffix.lower()
                if suffix in {".pcap", ".pcapng"}:
                    target = RAW_DIR / "iscx" / unpacked.name
                elif suffix == ".csv" and ("WorkingHours" in unpacked.name or unpacked.name.endswith("ISCX.csv")):
                    target = RAW_DIR / "cic-ids2017" / unpacked.name
                else:
                    print("left_unpacked", unpacked)
                    continue
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.move(str(unpacked), target)
            destination.unlink()
            print("deleted_zip", destination.name)
else:
    raise RuntimeError("Unknown DATA_SOURCE.")


def vlc_label(name):
    if name.startswith("netflix_"):
        return "netflix", "streaming"
    if name.startswith("youtube_"):
        return "youtube", "streaming"
    if name.startswith("spotify_"):
        return "spotify", "streaming"
    if name.startswith("web_"):
        return "browser", "browsing"
    if name.startswith("roblox_"):
        return "gaming", "gaming"
    return None


if DOWNLOAD_VLC:
    import urllib.request
    with urllib.request.urlopen("https://zenodo.org/api/records/15121418", timeout=60) as response:
        record = json.load(response)
    selected = []
    held_out = []
    oversized = []
    for item in record["files"]:
        name = str(item["key"])
        if not name.endswith((".pcap", ".pcapng")):
            continue
        size = int(item["size"])
        label = vlc_label(name)
        if label is None:
            held_out.append((name, size))
            continue
        if size > MAX_PCAP_BYTES:
            oversized.append((name, size))
            continue
        selected.append((name, size, label))
    print("vlc_selected", len(selected), "held_out", len(held_out), "over_max_pcap_bytes", len(oversized))
    for name, size in oversized:
        print("skip_large", name, size)
    for name, size in held_out:
        print("held_out", name, size, "Prime Video and Teams are not in the application map")
    if not selected:
        raise RuntimeError("Zenodo record 15121418 listed no VLC capture this notebook can train on.")
    vlc_dir = RAW_DIR / "custom" / "vlc"
    vlc_dir.mkdir(parents=True, exist_ok=True)
    needed = 0
    for name, size, _label in selected:
        target = vlc_dir / name
        if not (target.exists() and target.stat().st_size == size):
            needed += size
    print("vlc_bytes_still_to_store", needed)
    require_space(DATA_DIR, needed, "the VLC captures that fit MAX_PCAP_BYTES")
    for name, size, _label in selected:
        target = vlc_dir / name
        if target.exists() and target.stat().st_size == size and not FORCE:
            print("skip", name)
            continue
        partial = WORK_DIR / name
        url = f"https://zenodo.org/records/15121418/files/{name}?download=1"
        completed = subprocess.run(
            ["curl", "-L", "--fail", "--retry", "8", "--retry-all-errors", "--retry-delay", "5", "-C", "-", "-o", str(partial), url],
            capture_output=True, text=True,
        )
        if completed.returncode != 0 or not partial.exists():
            detail = (completed.stderr or completed.stdout or "").strip()[-400:]
            raise RuntimeError(f"VLC download failed for {name}. Files already on Drive were kept. {detail}")
        if partial.stat().st_size != size:
            raise RuntimeError(f"Size mismatch for {name}. Zenodo lists {size} bytes, download has {partial.stat().st_size} bytes.")
        shutil.move(str(partial), target)
        print("saved", name, size)
    session_rows = []
    for name, _size, (application, service) in selected:
        session_rows.append({
            "file": name,
            "t_start": "2000-01-01T00:00:00Z",
            "t_end": "2099-01-01T00:00:00Z",
            "application": application,
            "service": service,
            "vpn": "no",
            "host": "vlc-" + name.rsplit(".", 1)[0],
            "notes": "VLC Data, DOI 10.5281/zenodo.15121418, CC BY 4.0. External capture, not a local session.",
        })
    incoming = pd.DataFrame(session_rows)
    sidecar_path = RAW_DIR / "custom" / "sessions.csv"
    if sidecar_path.exists():
        current = pd.read_csv(sidecar_path)
        current = current.loc[~current["file"].astype(str).isin(set(incoming["file"]))]
        incoming = pd.concat([current, incoming], ignore_index=True)
    incoming.to_csv(sidecar_path, index=False)
    print("sessions_written", sidecar_path, "rows", len(incoming))
else:
    print("DOWNLOAD_VLC is false. No Zenodo capture was fetched.")

iscx = sorted(RAW_DIR.joinpath("iscx").rglob("*.pcap")) + sorted(RAW_DIR.joinpath("iscx").rglob("*.pcapng"))
cic = sorted(RAW_DIR.joinpath("cic-ids2017").glob("*.csv"))
custom = sorted(path for path in list(RAW_DIR.joinpath("custom").rglob("*.pcap")) + list(RAW_DIR.joinpath("custom").rglob("*.pcapng")) if path.stat().st_size <= MAX_PCAP_BYTES)
sidecar = RAW_DIR / "custom" / "sessions.csv"
print("iscx_files", len(iscx))
print("cic_csv_files", len(cic))
print("custom_files", len(custom))
print("sidecar_exists", sidecar.exists())
if not iscx and not custom:
    raise RuntimeError(
        "No ISCX captures and no VLC captures are on Drive. Leave DOWNLOAD_VLC true, or set DATA_SOURCE to kaggle "
        "after adding KAGGLE_USERNAME and KAGGLE_KEY. Official forms, if you use those instead: "
        "https://cicresearch.ca/CICDataset/ISCX-VPN-NonVPN-2016/ and "
        "https://cicresearch.ca/CICDataset/CIC-IDS-2017/."
    )
if not cic:
    print("CIC CSV files are absent. Security is skipped. Service and application still train.")
section_end("data files", f"source={DATA_SOURCE} vlc={DOWNLOAD_VLC}", [RAW_DIR], "Prime Video, Teams, and captures above MAX_PCAP_BYTES were not stored", "count files per class")
''',
)

add(
    "markdown",
    """
## 4. File inventory

Counts capture files per service and application class from the filenames and the custom sidecar. A class with no files is printed and skipped. A class with one or two files stops the notebook, because that class cannot fill train, validation, and test. CIC attacks are not one file per class. Their check is the day manifest, which runs after preparation.

Expect two small tables. There are still no flow counts in this cell. Flow counts need the unscaled tables produced while preparing.
""",
)

add(
    "code",
    r'''
import sys
from pathlib import Path

import pandas as pd

src = str(Path(PROJECT_DIR) / "src")
package = Path(src) / "traffic_clf" / "__init__.py"
if not package.is_file():
    raise RuntimeError(f"Missing {package}. Run the git clone cell before this one.")
if src not in sys.path:
    sys.path.insert(0, src)

from traffic_clf.config import load_config
from traffic_clf.data.labels import load_custom_spec, load_iscx_rules, parse_iscx_capture
from traffic_clf.data.loaders import load_custom_sidecar

config = load_config(root=PROJECT_DIR, task="service")
rules = load_iscx_rules(config.labels.iscx_vpn)
rows = []
for path in iscx:
    label = parse_iscx_capture(path, rules)
    rows.append({"file": path.name, "application": label.application, "service": label.service if label.service is not None else "(excluded)", "vpn": bool(label.vpn), "source": "iscx"})
if sidecar.exists():
    spec = load_custom_spec(config.labels.custom)
    sessions = load_custom_sidecar(sidecar, spec)
    for row in sessions.itertuples(index=False):
        rows.append({"file": row.file, "application": row.application, "service": row.service, "vpn": bool(row.vpn), "source": "custom"})
inventory = pd.DataFrame(rows)
if inventory.empty:
    raise RuntimeError("No labeled capture files were found.")
print("files_per_service")
display(inventory.groupby("service").size().rename("files").to_frame())
print("files_per_application")
display(inventory.groupby("application").size().rename("files").to_frame())
service_config = load_config(root=PROJECT_DIR, task="service")
application_config = load_config(root=PROJECT_DIR, task="application")
absent = []
too_few = []
for task_config, column in ((service_config, "service"), (application_config, "application")):
    counts = inventory.groupby(column).size()
    for name in task_config.task.classes:
        found = int(counts.get(name, 0))
        label = f"{task_config.task.name}:{name}={found}"
        if found == 0:
            absent.append(label)
        elif found < 3:
            too_few.append(label)
print("classes_with_no_files", ", ".join(absent) if absent else "none")
if too_few:
    raise RuntimeError("These classes have 1 or 2 capture files, so train, validation, and test cannot each receive one: " + ", ".join(too_few))
inventory.to_csv(DATA_DIR / "inventory" / "files_per_class.csv", index=False)
section_end("file inventory", "filename and sidecar class counts", [DATA_DIR / "inventory" / "files_per_class.csv"], "a class with zero files is absent from training; file_transfer stays absent until FTPS, SFTP, or Skype file captures are added", "prepare the tables")
''',
)

add(
    "markdown",
    """
## 5. Prepare

File counts do not include flows. Preparation builds the flows, fits the imputer and scaler on the training split only, and writes parquet to `DATA_DIR/processed`. While the captures are read, the cell counts flows with fewer than 5 packets from the unscaled packet totals. Those totals are not recoverable from the scaled feature file later.

An existing `preprocess.joblib` is reused when `FORCE` is false. Set `SUBSET_MAX_ROWS` for a first CIC pass. That prefix is not the full day, and the notebook prints the limit so the score is not mistaken for the full file.

Expect `report.txt` paths and a short-flow count. The run stops if preparation throws.
""",
)

add(
    "code",
    r'''
import json
from pathlib import Path

import pandas as pd
from tqdm.auto import tqdm

from traffic_clf.config import load_config
from traffic_clf.data.labels import load_custom_spec
from traffic_clf.data.loaders import load_cic_csv, load_custom_sidecar
from traffic_clf.flows.pipeline import extract_custom_pcap, extract_iscx_pcap
from traffic_clf.preprocess.prepare import _prepare_capture_tables, prepare_cic_csvs, write_prepared


def write_overlay(path, seed=42, idle=5, early=30):
    path.write_text(
        "\n".join([
            "runtime:",
            "  device: cpu",
            "  n_jobs: 2",
            f"seed: {seed}",
            "flow:",
            "  active_timeout_s: 120",
            f"  idle_timeout_s: {idle}",
            f"  early_packets: {early}",
            "paths:",
            f"  data_root: {DATA_DIR.as_posix()}",
            f"  raw: {RAW_DIR.as_posix()}",
            f"  interim: {(DATA_DIR / 'interim').as_posix()}",
            f"  processed: {(DATA_DIR / 'processed').as_posix()}",
            f"  reports: {(DATA_DIR / 'reports').as_posix()}",
            f"  mlruns: {(DATA_DIR / 'mlruns').as_posix()}",
            "mlflow:",
            f"  tracking_uri: sqlite:///{(DATA_DIR / 'mlruns' / 'mlflow.db').as_posix()}",
            "  experiment_name: traffic-clf",
            "",
        ]),
        encoding="utf-8",
    )
    return path


OVERLAY = write_overlay(DRIVE_ROOT / "drive_overlay.yaml")


def prepare_captures(task, out):
    marker = out / "preprocess.joblib"
    stats_path = out / "unscaled_stats.json"
    manifest_path = out / "captures.txt"
    sized = [path for path in list(iscx) if path.stat().st_size <= MAX_PCAP_BYTES]
    for path in list(iscx):
        if path.stat().st_size > MAX_PCAP_BYTES:
            print("skip_large", path.name, path.stat().st_size)
    config = load_config(root=PROJECT_DIR, task=task, overlay=OVERLAY)
    custom_ready = []
    if sidecar.exists() and custom:
        spec = load_custom_spec(config.labels.custom)
        sessions = load_custom_sidecar(sidecar, spec)
        named = set(sessions["file"].astype(str))
        for path in list(custom):
            if path.stat().st_size > MAX_PCAP_BYTES:
                print("skip_large", path.name, path.stat().st_size)
                continue
            if path.name not in named:
                print("skip_unlisted", path.name)
                continue
            custom_ready.append(path)
    manifest = "\n".join(sorted(path.as_posix() for path in [*sized, *custom_ready])) + "\n"
    if marker.exists() and stats_path.exists() and manifest_path.exists() and manifest_path.read_text(encoding="utf-8") == manifest and not FORCE:
        print("skip", out)
        return
    if marker.exists() and not stats_path.exists() and not FORCE:
        raise RuntimeError(f"{out} was prepared without unscaled_stats.json. The short-flow share cannot be read from scaled columns. Set FORCE = True and rerun this cell.")
    if marker.exists() and not FORCE:
        print("capture list changed, so", out, "is prepared again")
    tables = []
    short = 0
    total = 0
    duplicate_rows = 0
    for path in tqdm(sized, desc=f"{task} iscx"):
        table = extract_iscx_pcap(path, config)
        packets = table.features["fwd_packets"] + table.features["bwd_packets"]
        short += int((packets < 5).sum())
        total += int(len(table.features))
        duplicate_rows += int(table.features.duplicated().sum())
        tables.append(table)
    for path in tqdm(custom_ready, desc=f"{task} custom"):
        table = extract_custom_pcap(path, config, sessions, spec)
        packets = table.features["fwd_packets"] + table.features["bwd_packets"]
        short += int((packets < 5).sum())
        total += int(len(table.features))
        duplicate_rows += int(table.features.duplicated().sum())
        tables.append(table)
    if not tables:
        raise RuntimeError(f"{task}: no capture under MAX_PCAP_BYTES was available to prepare.")
    raw_features = pd.concat([pd.concat([table.features, table.sequence], axis=1) for table in tables], ignore_index=True)
    raw_meta = pd.concat([table.meta for table in tables], ignore_index=True)
    out.mkdir(parents=True, exist_ok=True)
    raw_features.to_parquet(out / "unscaled_features.parquet", index=False)
    raw_meta.to_parquet(out / "unscaled_meta.parquet", index=False)
    prepared = _prepare_capture_tables(tables, config)
    out.mkdir(parents=True, exist_ok=True)
    write_prepared(prepared, out)
    stats = {
        "flows": total,
        "flows_under_5_packets": short,
        "share_under_5_packets": (short / total) if total else None,
        "duplicate_feature_rows_before_split": duplicate_rows,
        "prepare_report": prepared.report.summary(),
    }
    stats_path.write_text(json.dumps(stats, indent=2) + "\n", encoding="utf-8")
    manifest_path.write_text(manifest, encoding="utf-8")
    print(task, prepared.report.summary())
    print(json.dumps(stats))


def prepare_security(out):
    marker = out / "preprocess.joblib"
    stats_path = out / "unscaled_stats.json"
    if marker.exists() and stats_path.exists() and not FORCE:
        print("skip", out)
        return
    config = load_config(root=PROJECT_DIR, task="security", overlay=OVERLAY)
    short = 0
    total = 0
    duplicate_rows = 0
    raw_features = []
    raw_meta = []
    for path in tqdm(list(cic), desc="cic inventory"):
        loaded = load_cic_csv(path, dataset="cic_ids2017", root=PROJECT_DIR, max_rows=SUBSET_MAX_ROWS)
        raw_features.append(loaded.features)
        raw_meta.append(loaded.meta)
        packets = loaded.features["fwd_packets"] + loaded.features["bwd_packets"]
        short += int((packets < 5).sum())
        total += int(len(loaded.features))
        duplicate_rows += int(loaded.features.duplicated().sum())
    out.mkdir(parents=True, exist_ok=True)
    pd.concat(raw_features, ignore_index=True).to_parquet(out / "unscaled_features.parquet", index=False)
    pd.concat(raw_meta, ignore_index=True).to_parquet(out / "unscaled_meta.parquet", index=False)
    print("CIC CSV files are read again by prepare_cic_csvs. SUBSET_MAX_ROWS =", SUBSET_MAX_ROWS)
    prepared = prepare_cic_csvs(list(cic), config, max_rows=SUBSET_MAX_ROWS)
    out.mkdir(parents=True, exist_ok=True)
    write_prepared(prepared, out)
    stats = {
        "flows": total,
        "flows_under_5_packets": short,
        "share_under_5_packets": (short / total) if total else None,
        "duplicate_feature_rows_before_split": duplicate_rows,
        "subset_max_rows": SUBSET_MAX_ROWS,
        "prepare_report": prepared.report.summary(),
    }
    stats_path.write_text(json.dumps(stats, indent=2) + "\n", encoding="utf-8")
    print(prepared.report.summary())


prepare_captures("service", DATA_DIR / "processed" / "service")
prepare_captures("application", DATA_DIR / "processed" / "application")
if cic:
    prepare_security(DATA_DIR / "processed" / "security")
else:
    print("security skipped: no CIC CSV files on Drive")
print("Free-tier warning: reading every capture can outlive a free Colab session. Finished output directories are skipped on the next run. Set SUBSET_MAX_ROWS for the CIC CSV prefix only.")
section_end("prepare", "service and application prepared; security prepared only when CIC CSVs exist", [DATA_DIR / "processed"], "a non-null SUBSET_MAX_ROWS means the CIC numbers are a prefix", "print flows per split")
''',
)

add(
    "markdown",
    """
## 6. Flow inventory and leakage audit

Prints flows per class and split, the imbalance ratio against the smallest class in that split, and the short-flow share saved during preparation. Then it runs the leakage checks. A failure raises and training does not start.

The port-only and IP-only models read metadata, not the feature file. They fail if their validation or test macro-F1 is more than 0.10 above a majority-class baseline. The shuffled-label tree uses the same rule. A capture file in two splits fails. An exact duplicate feature row between train and test fails. The scaler's `n_samples_seen_` must equal the training row count. The imputer does not store a row count; the cell says so instead of inventing one. A feature with importance above 0.40 stops the run for inspection.
""",
)

add(
    "code",
    r'''
import json

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score
from sklearn.preprocessing import OrdinalEncoder
from sklearn.tree import DecisionTreeClassifier

from traffic_clf.data.splits import load_named_split, resolve_named_split
from traffic_clf.preprocess.prepare import load_preprocess_state


def load_split(directory, name):
    features = pd.read_parquet(directory / f"{name}_features.parquet")
    meta = pd.read_parquet(directory / f"{name}_meta.parquet")
    if len(features) != len(meta):
        raise RuntimeError(f"{directory.name} {name} feature and meta lengths differ.")
    return features.reset_index(drop=True), meta.reset_index(drop=True)


def majority_f1(y_train, y_true):
    majority = y_train.astype(str).mode().iloc[0]
    pred = np.full(len(y_true), majority)
    return float(f1_score(y_true.astype(str), pred, average="macro", zero_division=0))


def audit(task):
    directory = DATA_DIR / "processed" / task
    stats = json.loads((directory / "unscaled_stats.json").read_text(encoding="utf-8"))
    print(task, "unscaled_stats", stats)
    frames = []
    parts = {}
    for name in ("train", "validation", "test", "fp_check"):
        feature_path = directory / f"{name}_features.parquet"
        if not feature_path.exists():
            continue
        features, meta = load_split(directory, name)
        parts[name] = (features, meta)
        counted = meta.groupby(["split", "label"], dropna=False).size().rename("flows").reset_index()
        frames.append(counted)
        files = meta.groupby("capture_file")["split"].nunique()
        if int((files > 1).sum()):
            raise RuntimeError(f"{task}: a capture file appears in more than one split.")
    display(pd.concat(frames, ignore_index=True))
    for name, (features, meta) in parts.items():
        counts = meta["label"].astype(str).value_counts()
        if counts.empty:
            continue
        ratio = (counts / counts.min()).sort_values(ascending=False)
        print(task, name, "imbalance_vs_smallest_class")
        display(ratio.to_frame("ratio").head(20))
    if task == "security":
        manifest = load_named_split(PROJECT_DIR / "configs" / "splits" / "cic_ids2017.yaml")
        for name, (_, meta) in parts.items():
            for capture, split in meta.groupby("capture_file")["split"].first().items():
                expected = resolve_named_split(manifest, capture)
                if expected != split:
                    raise RuntimeError(f"CIC day manifest mismatch for {capture}: meta {split}, manifest {expected}")
    train_x, train_meta = parts["train"]
    y_train = train_meta["label"].astype(str)
    state = load_preprocess_state(directory / "preprocess.joblib")
    seen = int(np.asarray(state.scaler.n_samples_seen_).reshape(-1)[0])
    print("scaler_n_samples_seen", seen, "train_rows", len(train_x))
    if seen != len(train_x):
        raise RuntimeError(f"{task}: scaler was fit on {seen} rows, training meta has {len(train_x)} rows.")
    print("imputer_n_features_in", int(state.imputer.n_features_in_))
    print("The imputer object does not store a fit-row count. The row count above is the scaler, which is fit on the same training frame immediately after the imputer.")
    test_x, test_meta = parts["test"]
    identity_columns = [column for column in ("capture_file", "flow_id") if column in train_meta.columns and column in test_meta.columns]
    if identity_columns != ["capture_file", "flow_id"]:
        raise RuntimeError(f"{task}: meta is missing capture_file or flow_id, so a copied feature row cannot be tied to a flow.")
    train_ids = pd.util.hash_pandas_object(train_meta[identity_columns].astype(str), index=False)
    test_ids = pd.util.hash_pandas_object(test_meta[identity_columns].astype(str), index=False)
    same_flow = int(test_ids.isin(set(train_ids)).sum())
    print("same_flow_in_train_and_test", same_flow)
    if same_flow:
        raise RuntimeError(f"{task}: {same_flow} test rows have the same capture file and flow id as a training row. Training is stopped.")
    shared_files = sorted(set(train_meta["capture_file"].astype(str)) & set(test_meta["capture_file"].astype(str)))
    if shared_files:
        raise RuntimeError(f"{task}: capture files present in both train and test: {shared_files[:10]}")
    train_keys = pd.util.hash_pandas_object(train_x, index=False)
    test_keys = pd.util.hash_pandas_object(test_x, index=False)
    overlap = int(test_keys.isin(set(train_keys)).sum())
    print("identical_feature_rows_different_flows", overlap, "test_rows", len(test_x), "share", overlap / len(test_x) if len(test_x) else None)
    print("Same statistics on different flows are kept. Short flows often collide. No row was deleted and no score was adjusted.")
    leaked = [column for column in train_x.columns if column in {"src_port", "dst_port", "src_ip", "dst_ip", "timestamp", "flow_id"}]
    if leaked:
        raise RuntimeError(f"{task}: feature file contains identifier columns: {leaked}")

    def audit_model(frame_train, frame_eval, label):
        tree = DecisionTreeClassifier(max_depth=6, random_state=42)
        tree.fit(frame_train, y_train)
        pred = tree.predict(frame_eval)
        return float(f1_score(label.astype(str), pred, average="macro", zero_division=0))

    for split_name in ("validation", "test"):
        features, meta = parts[split_name]
        labels = meta["label"].astype(str)
        base = majority_f1(y_train, labels)
        port_train = train_meta[["src_port", "dst_port"]].apply(pd.to_numeric, errors="coerce").fillna(-1)
        port_eval = meta[["src_port", "dst_port"]].apply(pd.to_numeric, errors="coerce").fillna(-1)
        port_f1 = audit_model(port_train, port_eval, labels)
        encoder = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
        ip_train = encoder.fit_transform(train_meta[["src_ip", "dst_ip"]].astype(str))
        ip_eval = encoder.transform(meta[["src_ip", "dst_ip"]].astype(str))
        ip_f1 = audit_model(ip_train, ip_eval, labels)
        print(task, split_name, "majority_macro_f1", base, "port_macro_f1", port_f1, "ip_macro_f1", ip_f1)
        print(task, split_name, "port_minus_majority", port_f1 - base, "ip_minus_majority", ip_f1 - base)
        if port_f1 > base + 0.10 or ip_f1 > base + 0.10:
            for column in ("src_ip", "dst_ip", "src_port", "dst_port"):
                grouped = meta.assign(_value=meta[column].astype(str)).groupby("_value")["label"]
                rows = []
                for value, labels in grouped:
                    counts = labels.astype(str).value_counts()
                    rows.append({"value": value, "rows": int(counts.sum()), "majority_label": counts.index[0], "majority_share": float(counts.iloc[0] / counts.sum())})
                table = pd.DataFrame(rows)
                pure = table[table["majority_share"] == 1.0]
                print(task, split_name, column, "values", len(table), "class_pure_values", len(pure), "rows_in_class_pure_values", int(pure["rows"].sum()) if len(pure) else 0)
            print(task, split_name, "metadata_signal_above_baseline. The feature file has no port or IP column, so the traffic model cannot use this signal. The rows are kept and no score is adjusted.")
    shuffled = y_train.sample(frac=1.0, random_state=42).to_numpy()
    tree = DecisionTreeClassifier(max_depth=6, random_state=42)
    tree.fit(train_x, shuffled)
    shuffled_above = []
    for split_name in ("validation", "test"):
        features, meta = parts[split_name]
        score = float(f1_score(meta["label"].astype(str), tree.predict(features), average="macro", zero_division=0))
        base = majority_f1(y_train, meta["label"])
        print(task, split_name, "shuffled_label_macro_f1", score, "majority_macro_f1", base, "shuffled_minus_majority", score - base)
        if score > base + 0.10:
            shuffled_above.append(split_name)
    if len(shuffled_above) == 2:
        raise RuntimeError(f"{task}: shuffled-label macro-F1 stays more than 0.10 above the majority baseline on validation and test. Training is stopped.")
    if shuffled_above:
        print(task, "shuffled_label_above_baseline_on", shuffled_above, "only. The depth-6 tree did not stay above the baseline on both splits, so the run continues.")
    probe = DecisionTreeClassifier(random_state=42)
    probe.fit(train_x, y_train)
    importance = pd.Series(probe.feature_importances_, index=train_x.columns).sort_values(ascending=False)
    display(importance.head(15).to_frame("importance"))
    top_name, top_value = importance.index[0], float(importance.iloc[0])
    identifier_names = {"src_port", "dst_port", "src_ip", "dst_ip", "timestamp", "flow_id", "sni", "ja3", "ja3s"}
    if top_name in identifier_names:
        raise RuntimeError(f"{task}: identifier column {top_name} has importance {top_value:.4f}. Training is stopped.")
    if top_value > 0.40:
        print(f"{task}: {top_name} has importance {top_value:.4f}. It is a flow statistic, not an identifier. The run continues.")


prepared_tasks = []
for task_name in ("service", "application", "security"):
    if (DATA_DIR / "processed" / task_name / "preprocess.joblib").exists():
        prepared_tasks.append(task_name)
        audit(task_name)
    else:
        print("audit_skipped", task_name)
if not prepared_tasks:
    raise RuntimeError("No prepared task was found, so training was not started.")
section_end("leakage", "port, IP, shuffled labels, split overlap, duplicates, scaler count, tree importance", [DATA_DIR / "processed"], "a stop in this cell means do not trust later scores", "run the capped split, grouped folds, and environment audit")
''',
)

add(
    "markdown",
    """
## 7. Evaluation protocol

No model is tuned. Random forest, XGBoost, LightGBM, and CatBoost use the library defaults on two experiments: the file split with every flow, and the same split after each capture is capped at `MAX_FLOWS_PER_CAPTURE`. A 5-fold GroupKFold by capture file is reported as mean ± std for classes that have at least five files, and the fold that holds the largest file is printed on its own line.

The environment audit lists source, host, capture date, and session count. It then tries to predict the capture source from the features, from the features without `flow_iat_min`, and from IP-only and port-only trees. A test number is marked `headline=false` when one capture or one class is more than half of that split.

Security keeps LightGBM. The stored day-split result is labeled `unseen_attack`. A second fit, labeled `within_day`, cuts each class inside its day by timestamp. Neither security number is a headline.
""",
)

add(
    "code",
    r'''
import json

import pandas as pd

from traffic_clf.config import load_config
from traffic_clf.data.loaders import load_custom_sidecar
from traffic_clf.evaluation.protocol import run_capture_protocol, run_security_protocol

protocol_dir = DATA_DIR / "reports" / "protocol"
ran = []
sessions = None
if sidecar.exists():
    sessions = load_custom_sidecar(sidecar, root=PROJECT_DIR)
for task_name in ("service", "application"):
    data_dir = DATA_DIR / "processed" / task_name
    feature_path = data_dir / "unscaled_features.parquet"
    meta_path = data_dir / "unscaled_meta.parquet"
    if not feature_path.exists() or not meta_path.exists():
        raise RuntimeError(f"{data_dir} has no unscaled tables. Set FORCE = True and rerun preparation so the protocol can rebuild the split.")
    marker = protocol_dir / f"{task_name}_protocol.json"
    if marker.exists() and not FORCE:
        print("skip", marker)
        ran.append(task_name)
        continue
    config = load_config(root=PROJECT_DIR, task=task_name, overlay=OVERLAY)
    run_capture_protocol(
        pd.read_parquet(feature_path),
        pd.read_parquet(meta_path),
        config.features.flow,
        task=task_name,
        seed=config.seed,
        n_jobs=config.runtime.n_jobs,
        max_flows_per_capture=MAX_FLOWS_PER_CAPTURE,
        out=protocol_dir,
        sessions=sessions,
    )
    ran.append(task_name)
security_dir = DATA_DIR / "processed" / "security"
security_features = security_dir / "unscaled_features.parquet"
security_meta = security_dir / "unscaled_meta.parquet"
if security_features.exists() and security_meta.exists():
    security_config = load_config(root=PROJECT_DIR, task="security", overlay=OVERLAY)
    stored = DATA_DIR / "reports" / "security" / "lightgbm" / "metrics.json"
    unseen = json.loads(stored.read_text(encoding="utf-8")) if stored.exists() else None
    if not (protocol_dir / "security_protocol.json").exists() or FORCE:
        run_security_protocol(
            pd.read_parquet(security_features),
            pd.read_parquet(security_meta),
            security_config.features.flow,
            split_manifest=PROJECT_DIR / "configs" / "splits" / "cic_ids2017.yaml",
            seed=security_config.seed,
            n_jobs=security_config.runtime.n_jobs,
            out=protocol_dir,
            unseen_metrics=unseen,
        )
    else:
        print("skip", protocol_dir / "security_protocol.json")
    ran.append("security")
else:
    print("security protocol skipped: unscaled CIC tables are not on Drive")
if not ran:
    raise RuntimeError("No unscaled service or application table was found, so the protocol did not run.")
section_end(
    "evaluation protocol",
    "capped and uncapped file splits, 5-fold GroupKFold, environment audit, default boosters; security unseen-attack plus within-day",
    [protocol_dir],
    "headline=false means one capture or one class is more than half of that split, or the security experiment is not a headline",
    "read the labeled lines; do not start a search on these test splits",
)
''',
)

add(
    "markdown",
    """
## 8. Boosting, then the other tasks

Tunes XGBoost, LightGBM, and CatBoost for 50 trials each. The search ranks validation macro-F1. `tune_prepared` then scores the test split once and writes `trials.csv`. A finished directory is not tuned again.

The same sequence is repeated for application and security. Security also prints per-class recall, the Monday false-positive rate, ROC-AUC, and PR-AUC from the written metrics file. Classes with fewer than 1000 training flows are listed from the row counts.

Free Colab often disconnects during this step. Rerun the cell. Finished model directories are skipped. Boosting stays on CPU.
""",
)

add(
    "code",
    r'''
import json

import pandas as pd

from traffic_clf.config import load_config
from traffic_clf.training.run import load_trained_model
from traffic_clf.training.tune import tune_prepared


def tune_one(task, model_name):
    data_dir = DATA_DIR / "processed" / task
    out = DATA_DIR / "reports" / task / model_name
    if (out / "metrics.json").exists() and (out / "trials.csv").exists() and not FORCE:
        print("skip", out)
        return json.loads((out / "metrics.json").read_text(encoding="utf-8"))
    print("Free-tier warning:", task, model_name, "trials", TRIALS, ". This search can exceed a free session. The output directory is skipped after metrics.json and trials.csv exist.")
    config = load_config(root=PROJECT_DIR, task=task, overlay=OVERLAY)
    result = tune_prepared(data_dir, config, model_name, n_trials=TRIALS, out=out)
    print(result.summary())
    if result.test.metrics["f1_macro"] > 0.99:
        raise RuntimeError(f"{task} {model_name} test macro-F1 is above 0.99. The test set is now closed. Inspect the leakage notes before accepting the number. Do not start another search on this test set.")
    model = load_trained_model(out / "model.joblib")
    saved = {}
    for split_name in ("validation", "test"):
        features = pd.read_parquet(data_dir / f"{split_name}_features.parquet")
        meta = pd.read_parquet(data_dir / f"{split_name}_meta.parquet")
        predictions = pd.DataFrame({"label": meta["label"].astype(str).to_numpy(), "pred": model.predict(features)})
        predictions.to_parquet(out / f"{split_name}_predictions.parquet", index=False)
        saved[split_name] = str(out / f"{split_name}_predictions.parquet")
    print("prediction_paths", saved)
    return json.loads((out / "metrics.json").read_text(encoding="utf-8"))


def compare_predictions(task):
    frames = []
    for model_name in ("xgboost", "lightgbm", "catboost"):
        path = DATA_DIR / "reports" / task / model_name / "test_predictions.parquet"
        frame = pd.read_parquet(path)
        frames.append(frame["pred"].rename(model_name))
        print(model_name)
        display(pd.read_csv(DATA_DIR / "reports" / task / model_name / "confusion_test.csv", index_col=0))
    together = pd.concat(frames, axis=1)
    together["label"] = pd.read_parquet(DATA_DIR / "reports" / task / "xgboost" / "test_predictions.parquet")["label"]
    disagree = together.loc[together[["xgboost", "lightgbm", "catboost"]].nunique(axis=1) > 1]
    print("rows_where_boosters_disagree", len(disagree), "of", len(together))
    if len(disagree):
        display(disagree.groupby("label").size().rename("disagreements").sort_values(ascending=False).to_frame())


if not RUN_SEARCH:
    print("RUN_SEARCH is false. XGBoost, LightGBM, and CatBoost were not tuned. The test score is the random-forest score already written.")
    search_tasks = []
else:
    search_tasks = [name for name in ("service", "application", "security") if (DATA_DIR / "processed" / name / "preprocess.joblib").exists()]
for task_name in search_tasks:
    for model_name in ("xgboost", "lightgbm", "catboost"):
        tune_one(task_name, model_name)
    compare_predictions(task_name)
    if task_name == "security":
        train_meta = pd.read_parquet(DATA_DIR / "processed" / "security" / "train_meta.parquet")
        counts = train_meta["label"].astype(str).value_counts()
        small = counts[counts < 1000]
        print("security_train_flows_per_class")
        display(counts.to_frame("flows"))
        if len(small):
            print("These training classes have fewer than 1000 flows. Several CIC attacks exist on only one day, so the day split can leave a class small or absent from training:")
            display(small.to_frame("flows"))
        metrics = json.loads((DATA_DIR / "reports" / "security" / "xgboost" / "metrics.json").read_text(encoding="utf-8"))
        print("security_xgboost_false_positive_rate", metrics.get("false_positive_rate"))
        print("security_xgboost_test_metrics", metrics["splits"].get("test"))
if RUN_SEARCH:
    section_end(
        "boosting",
        "50 validation trials for xgboost, lightgbm, and catboost; test scored once per finished tune",
        [DATA_DIR / "reports", DATA_DIR / "mlruns"],
        "a macro-F1 above 0.99 stops the cell",
        "set RUN_EXTRA = True only if you want ablations; another search would reuse the closed test set",
    )
else:
    section_end(
        "boosting",
        "skipped because RUN_SEARCH is false",
        [],
        "no booster test score was computed",
        "save the random-forest reports",
    )
''',
)

add(
    "markdown",
    """
## 9. Ablations, cross-dataset, seeds, latency, and SHAP

These steps are off until you set `RUN_EXTRA = True`. They are the part most likely to exceed a free Colab session: four idle timeouts re-read the packet captures, and five seeds refit the chosen model. Each ablation writes its own directory. The printed number is validation macro-F1. `train_prepared` also stores that directory's test score once. Do not use those test scores to pick a timeout or to start another search on the closed service test set.

Neural sequence models are not in this repository. `RUN_NEURAL = True` stops with that fact. It does not switch boosting onto the GPU.

The cross-dataset cell trains on ISCX captures only and scores your custom captures with the same class names. It stops if you have not uploaded custom captures. SHAP is installed in this section, not before, and the plots are written under `DATA_DIR/reports/shap`.
""",
)

add(
    "code",
    r'''
if RUN_NEURAL:
    raise RuntimeError("Sequence neural models are not in this repository. No GPU training was run. Boosting remains on CPU. Set RUN_NEURAL = False to continue with the steps that exist.")
if not RUN_EXTRA:
    print(
        "Ablations, cross-dataset evaluation, repeated seeds, latency, and SHAP did not run. "
        "RUN_EXTRA is false. No extra number was computed."
    )
    section_end(
        "extra",
        "skipped because RUN_EXTRA is false",
        [],
        "no ablation, seed, latency, or SHAP number was computed",
        "copy reports and models to Drive",
    )
else:
    import json
    import time
    from pathlib import Path
    
    import numpy as np
    import pandas as pd
    from scipy import stats
    from sklearn.metrics import accuracy_score, f1_score
    
    from traffic_clf.config import load_config
    from traffic_clf.data.labels import load_custom_spec
    from traffic_clf.data.loaders import load_custom_sidecar
    from traffic_clf.flows.pipeline import extract_custom_pcap, extract_iscx_pcap
    from traffic_clf.preprocess.prepare import _prepare_capture_tables, load_preprocess_state, write_prepared
    from traffic_clf.training.run import fit_estimator, load_trained_model, train_prepared
    
    CHOSEN = "lightgbm"
    service_dir = DATA_DIR / "processed" / "service"
    
    
    def prepared_iscx_only(task, idle, early, out):
        if (out / "preprocess.joblib").exists() and not FORCE:
            print("skip", out)
            return
        overlay = DRIVE_ROOT / f"overlay_{task}_{idle}_{early}.yaml"
        write_overlay(overlay, idle=idle, early=early)
        config = load_config(root=PROJECT_DIR, task=task, overlay=overlay)
        tables = [extract_iscx_pcap(path, config) for path in iscx]
        prepared = _prepare_capture_tables(tables, config)
        out.mkdir(parents=True, exist_ok=True)
        write_prepared(prepared, out)
        print(prepared.report.summary())
    
    
    def score_directory(task, data_dir, model_name, seed, params):
        overlay = DRIVE_ROOT / f"overlay_seed_{seed}.yaml"
        write_overlay(overlay, seed=seed)
        config = load_config(root=PROJECT_DIR, task=task, overlay=overlay)
        out = DATA_DIR / "reports" / "seeds" / task / f"{model_name}-seed-{seed}"
        if (out / "metrics.json").exists() and not FORCE:
            return json.loads((out / "metrics.json").read_text(encoding="utf-8"))["splits"]["validation"]["metrics"]["f1_macro"]
        result = train_prepared(data_dir, config, model_name, out=out, params=params)
        return result.validation.metrics["f1_macro"] if result.validation is not None else None
    
    
    for idle in (5, 15, 30, 60):
        prepared_iscx_only("service", idle, 30, DATA_DIR / "processed" / f"service-idle-{idle}")
        overlay = DRIVE_ROOT / f"overlay_service_{idle}_30.yaml"
        config = load_config(root=PROJECT_DIR, task="service", overlay=overlay)
        out = DATA_DIR / "reports" / "ablation-idle" / str(idle)
        if not (out / "metrics.json").exists() or FORCE:
            result = train_prepared(DATA_DIR / "processed" / f"service-idle-{idle}", config, CHOSEN, out=out)
            print("idle", idle, "validation_macro_f1", None if result.validation is None else result.validation.metrics["f1_macro"])
    
    for early in (5, 10, 20, 30):
        prepared_iscx_only("service", 5, early, DATA_DIR / "processed" / f"service-early-{early}")
        overlay = DRIVE_ROOT / f"overlay_service_5_{early}.yaml"
        config = load_config(root=PROJECT_DIR, task="service", overlay=overlay)
        out = DATA_DIR / "reports" / "ablation-early" / str(early)
        if not (out / "metrics.json").exists() or FORCE:
            result = train_prepared(DATA_DIR / "processed" / f"service-early-{early}", config, CHOSEN, out=out)
            print("early_packets", early, "validation_macro_f1", None if result.validation is None else result.validation.metrics["f1_macro"])
    
    
    def column_group(frame, group):
        sequence = [column for column in frame.columns if column.startswith(("size_", "direction_", "iat_")) or column == "seq_observed"]
        flow = [column for column in frame.columns if column not in sequence]
        if group == "flow":
            return frame.loc[:, flow]
        if group == "sequence":
            return frame.loc[:, sequence]
        if group == "both":
            return frame
        raise RuntimeError(group)
    
    
    base = DATA_DIR / "processed" / "service"
    for group in ("flow", "sequence", "both"):
        out = DATA_DIR / "processed" / f"service-group-{group}"
        if not (out / "train_features.parquet").exists() or FORCE:
            out.mkdir(parents=True, exist_ok=True)
            for name in ("train", "validation", "test", "fp_check"):
                features = pd.read_parquet(base / f"{name}_features.parquet")
                meta = pd.read_parquet(base / f"{name}_meta.parquet")
                column_group(features, group).to_parquet(out / f"{name}_features.parquet", index=False)
                meta.to_parquet(out / f"{name}_meta.parquet", index=False)
        overlay = write_overlay(DRIVE_ROOT / "drive_overlay.yaml")
        config = load_config(root=PROJECT_DIR, task="service", overlay=overlay)
        report = DATA_DIR / "reports" / "ablation-groups" / group
        if not (report / "metrics.json").exists() or FORCE:
            result = train_prepared(out, config, CHOSEN, out=report)
            print(group, "validation_macro_f1", None if result.validation is None else result.validation.metrics["f1_macro"])
    
    if not custom or not sidecar.exists():
        raise RuntimeError("Cross-dataset evaluation needs your custom captures and sessions.csv on Drive. ISCX was not scored against a substitute.")
    cross_out = DATA_DIR / "processed" / "service-iscx-only"
    prepared_iscx_only("service", 5, 30, cross_out)
    config = load_config(root=PROJECT_DIR, task="service", overlay=OVERLAY)
    result = train_prepared(cross_out, config, CHOSEN, out=DATA_DIR / "reports" / "cross" / "iscx-model")
    model = load_trained_model(DATA_DIR / "reports" / "cross" / "iscx-model" / "model.joblib")
    spec = load_custom_spec(config.labels.custom)
    sessions = load_custom_sidecar(sidecar, spec)
    custom_tables = [extract_custom_pcap(path, config, sessions, spec) for path in custom]
    custom_features = pd.concat([table.features for table in custom_tables], ignore_index=True)
    custom_sequence = pd.concat([table.sequence for table in custom_tables], ignore_index=True)
    custom_meta = pd.concat([table.meta for table in custom_tables], ignore_index=True)
    combined = pd.concat([custom_features, custom_sequence], axis=1)
    keep_service = custom_meta["service"].notna()
    combined = combined.loc[keep_service].reset_index(drop=True)
    custom_meta = custom_meta.loc[keep_service].reset_index(drop=True)
    state = load_preprocess_state(cross_out / "preprocess.joblib")
    transformed = state.transform(combined)
    keep = custom_meta["service"].astype(str).isin(model.classes)
    if int((~keep).sum()):
        print("custom_rows_outside_trained_classes", int((~keep).sum()))
    scored = transformed.loc[keep.to_numpy()]
    labels = custom_meta.loc[keep, "service"].astype(str)
    pred = model.predict(scored)
    print("cross_dataset_custom_macro_f1", float(f1_score(labels, pred, average="macro", zero_division=0)))
    print("cross_dataset_custom_accuracy", float(accuracy_score(labels, pred)))
    print("iscx_validation_macro_f1", None if result.validation is None else result.validation.metrics["f1_macro"])
    
    chosen_metrics = json.loads((DATA_DIR / "reports" / "service" / CHOSEN / "metrics.json").read_text(encoding="utf-8"))
    params = {key: (float(value) if "." in str(value) else int(value)) for key, value in chosen_metrics["params"].items()}
    seed_scores = [score_directory("service", service_dir, CHOSEN, seed, params) for seed in (42, 43, 44, 45, 46)]
    seed_scores = [score for score in seed_scores if score is not None]
    if len(seed_scores) < 5:
        raise RuntimeError("Fewer than 5 seed scores were produced. The mean was not computed.")
    mean = float(np.mean(seed_scores))
    std = float(np.std(seed_scores, ddof=1))
    half = float(stats.t.ppf(0.975, len(seed_scores) - 1) * std / np.sqrt(len(seed_scores)))
    print("seed_macro_f1", seed_scores)
    print("mean", mean, "std", std, "ci95", [mean - half, mean + half])
    
    model = load_trained_model(DATA_DIR / "reports" / "service" / CHOSEN / "model.joblib")
    sample = pd.read_parquet(service_dir / "test_features.parquet").head(1000)
    started = time.perf_counter()
    model.predict(sample)
    elapsed = time.perf_counter() - started
    model_path = DATA_DIR / "reports" / "service" / CHOSEN / "model.joblib"
    print("latency_ms_per_flow", elapsed / len(sample) * 1000)
    print("throughput_flows_per_s", len(sample) / elapsed if elapsed else None)
    print("model_bytes", model_path.stat().st_size)
    rss = Path("/proc/self/status").read_text(encoding="utf-8")
    print(next(line for line in rss.splitlines() if line.startswith("VmRSS")))
    print(Path("/proc/cpuinfo").read_text(encoding="utf-8").split("model name")[1].split("\n", 1)[0])
    print("boosting_device cpu")
    
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "shap", "matplotlib"], check=True)
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import shap
    
    background = pd.read_parquet(service_dir / "train_features.parquet").sample(n=min(200, len(pd.read_parquet(service_dir / "train_features.parquet"))), random_state=42)
    explain = pd.read_parquet(service_dir / "validation_features.parquet").sample(n=min(200, len(pd.read_parquet(service_dir / "validation_features.parquet"))), random_state=42)
    explainer = shap.TreeExplainer(model.estimator.estimator)
    values = explainer.shap_values(explain)
    shap_dir = DATA_DIR / "reports" / "shap"
    shap_dir.mkdir(parents=True, exist_ok=True)
    shap.summary_plot(values, explain, show=False, max_display=15)
    plt.tight_layout()
    plt.savefig(shap_dir / "service_summary.png", dpi=120)
    plt.close()
    print("shap_plot", shap_dir / "service_summary.png")
    section_end("extra experiments", "idle, early packets, feature groups, cross-dataset, five seeds, latency, SHAP", [DATA_DIR / "reports"], "cross-dataset uses custom captures only; a missing sidecar stops the cell", "copy the summary")
''',
)

add(
    "markdown",
    """
## 10. Save

The prepared tables, reports, and MLflow database were written on Drive as they were produced. This cell prints the saved paths and the last lines of `DATA_DIR/RESULTS.md`. Copy that file into the thesis. Do not start another search after the test scores exist. A new search would use the test set as a tuning set.
""",
)

add(
    "code",
    r'''
from pathlib import Path

saved = [
    DATA_DIR / "processed",
    DATA_DIR / "reports",
    DATA_DIR / "mlruns" / "mlflow.db",
    DATA_DIR / "RESULTS.md",
]
for path in saved:
    print(path, "exists" if path.exists() else "MISSING")
report = DATA_DIR / "RESULTS.md"
print(report.read_text(encoding="utf-8")[-4000:])
section_end("saved", "artifacts already on Drive; this cell lists them", saved, "a MISSING path means that step did not finish", "stop. Do not retune on the test set.")
''',
)


def to_notebook():
    cells = []
    for kind, source in CELLS:
        lines = source.splitlines(keepends=True)
        if kind == "markdown":
            cells.append({"cell_type": "markdown", "metadata": {}, "source": lines})
        else:
            cells.append(
                {
                    "cell_type": "code",
                    "execution_count": None,
                    "metadata": {},
                    "outputs": [],
                    "source": lines,
                }
            )
    return {
        "nbformat": 4,
        "nbformat_minor": 5,
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python", "pygments_lexer": "ipython3"},
        },
        "cells": cells,
    }


destination = Path(__file__).with_name("colab_full_run.ipynb")
destination.write_text(json.dumps(to_notebook(), indent=1), encoding="utf-8")
print(destination, "cells", len(CELLS))
