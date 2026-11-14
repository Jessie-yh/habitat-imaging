# cervical-habitat


parametric maps, normalises every map inside the ROI, clusters voxels with
K-means, and quantifies the resulting sub-regions (volume, parameter
distribution, spatial interaction).

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/python-3.9%2B-blue.svg)](pyproject.toml)

---

## What it does

```
 NIfTI ROI ─┐
            ├─► grid alignment (ROI space) ─► resample 1×1×3 mm³
 DCE 4-D ───┘                                      │
   or                                  ROI-guided 1–99 % clip + min–max
 pre-computed maps ────────────────────────────────┤
                                                    ▼
                              pooled voxel features (all patients)
                                                    │
                                    K-means++  (DCE k=3 / IVIM k=4)
                                                    │
                    ┌───────────────────────────────┼──────────────────────┐
                    ▼                               ▼                      ▼
             habitat label map              volume & fraction      MSI matrix (22 feats)
```

Two study configurations ship with the repository:

| Configuration         | Features                    | Clusters  | Config file                      |
| --------------------- | --------------------------- | --------- | -------------------------------- |
| DCE semi-quantitative | wash-in rate, wash-out rate | **k = 3** | `configs/dce_rates_k3.yaml`      |
| IVIM                  | D, D\*, f                   | **k = 4** | `configs/ivim_d_dstar_f_k4.yaml` |

Both also run the four-criterion cluster-number evaluation (elbow, silhouette,
Calinski–Harabasz, Davies–Bouldin, k = 2–10) and report the majority-vote
suggestion, so the fixed *k* can be checked against the data.
