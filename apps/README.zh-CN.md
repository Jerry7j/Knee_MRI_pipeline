# Streamlit 应用

[English](README.md) | [简体中文](README.zh-CN.md)

上传 NPY [N,H,W] 或同一序列的单帧灰度 DICOM，选择切片后显示 MRI、ROI、分割和熵叠加，并下载所选 ROI 的 NPY 掩膜。

```powershell
python -m pip install -e ".[app]"
streamlit run apps/streamlit_app.py
```

侧边栏指定 ensemble JSON，模型通过 knee_mri.engine.predictor.EnsemblePredictor 加载并缓存。修改权重后需要清理缓存或重启应用，上传数据不会自动写入磁盘。

页面下载模型分辨率的 ROI 掩膜；完整体的图像数组尺寸分割通过 knee-mri predict 生成。运行需要真实分割权重和可选定位权重。
