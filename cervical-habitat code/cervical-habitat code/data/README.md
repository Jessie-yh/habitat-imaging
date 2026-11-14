# data/

Create one sub-directory per patient under `data/raw/` and put the NIfTI files
inside it:

```
data/raw/
├── C001/
│   ├── roi.nii.gz          # binary tumour mask, 1 = tumour
│   ├── dce.nii.gz          # 4-D DCE series (x, y, z, time)   [DCE mode]
│   ├── ivim_D.nii.gz       # D and D* in mm²/s                [IVIM mode]
│   ├── ivim_Dstar.nii.gz
│   └── ivim_f.nii.gz       # perfusion fraction, 0–1
└── C002/...
```

* The directory name becomes `case_id` in every output table.
* File names are configurable through `{case}` placeholders in the YAML
  (`dataset.roi_pattern`, `dataset.maps[].pattern`), so an existing export
  layout can be used as-is.
* All series of a patient must already be co-registered; DICOM conversion,
  ROI delineation and registration are out of scope.

Requirements and a `dcm2niix` example: [../docs/data_layout.md](../docs/data_layout.md).
