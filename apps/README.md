# Streamlit application

[English](README.md) | [简体中文](README.zh-CN.md)

Upload an NPY volume [N,H,W] or single-frame grayscale DICOM files from one series. Select a slice to view its MRI, ROI, segmentation, and entropy overlay, and download the ROI mask as NPY.

```powershell
python -m pip install -e ".[app]"
streamlit run apps/streamlit_app.py
```

Select the ensemble JSON in the sidebar. Models are loaded and cached through `knee_mri.engine.predictor.EnsemblePredictor`. Clear the cache or restart the application after changing weights. Uploads are not automatically saved to disk.

The page downloads ROI masks at model resolution. Use `knee-mri predict` for full-volume segmentation on the input image grid. Actual segmentation weights and, if configured, localization weights are required.
