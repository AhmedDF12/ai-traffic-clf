# Results

Numbers below were measured on this machine. No model was trained. No flow count is filled in, because the captures and CSVs are not on disk.

## Step 0 — Data inventory (blocked)

Checked on 2026-11-06.

| Check | Result |
| --- | --- |
| `D:\Projects\AI - P1\data` | Path does not exist |
| ISCX PCAPs under the project | 0 |
| CIC-IDS2017 CSVs under the project | 0 |
| Custom PCAPs and `sessions.csv` | 0 |
| `D:\Projects` | `AI - P1` and `Image Project- OCR` only |
| Downloads filenames matching pcap, ISCX, CIC-IDS, MachineLearning, sessions.csv | none (`ProtonVPN_v4.3.13_x64.exe` is an installer, not the dataset) |

Free space measured with `Win32_LogicalDisk`:

| Drive | Free | Size |
| --- | --- | --- |
| C: | 11.01 GB | 930.46 GB |
| D: | 34.72 GB | 97.66 GB |
| F: | 48.31 GB | 833.86 GB |

Published sizes from the dataset pages, not measured here:

| Archive | Published size | Source |
| --- | --- | --- |
| ISCXVPN2016 captures | 28 GB | https://www.unb.ca/cic/datasets/vpn.html |
| CIC-IDS2017 Monday–Friday PCAPs | 11.0 + 11 + 13 + 7.8 + 8.3 GB | https://www.unb.ca/cic/datasets/ids-2017.html |

A request for `MachineLearningCSV.zip` on the old CIC host returned HTTP 301 and then HTTP 302 to the UNB dataset index. The file itself was not downloaded. The CIC form at `https://cicresearch.ca/CICDataset/CIC-IDS-2017/` asks for an email, an organization, and a job title. The ISCX form is `https://cicresearch.ca/CICDataset/ISCX-VPN-NonVPN-2016/`.

Flows per class, imbalance, short-flow share, and duplicate-flow rate were not computed. Every class has 0 capture files, which is under the 3-file minimum, so the inventory stops here.

Classes were not dropped or merged in a run. The label maps still say email and P2P are excluded from the six service classes, and the CIC tool names collapse into `ddos` or `malware`. That collapse has not been applied to any real file yet.
