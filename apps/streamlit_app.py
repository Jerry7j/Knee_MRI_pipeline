"""Run with: streamlit run apps/streamlit_app.py"""
from pathlib import Path
import io
import numpy as np
import streamlit as st
from knee_mri.engine.predictor import EnsemblePredictor
from knee_mri.data.inputs import load_dicom_series, validate_volume
from knee_mri.visualization.plots import PALETTE, CLASS_NAMES, entropy_map

PROJECT_ROOT = Path(__file__).resolve().parents[1]


@st.cache_resource
def load_predictor(config):
    """Create and cache the ensemble predictor used by the application.

    Args:
        config: Ensemble JSON path, also used as a Streamlit resource cache key.

    Returns:
        A loaded EnsemblePredictor. Clear the cache or restart after changing weights.
    """
    return EnsemblePredictor(config)


def main():
    """Render MRI upload, slice selection, predictions, and ROI mask downloads.

    Notes:
        Accepts one NPY volume or a single DICOM series without automatically
        saving uploads. Displays results or input, dependency, and model errors
        in the Streamlit page.
    """
    st.set_page_config(page_title="KneeXNet-2.5D", layout="wide")
    st.title("Knee MRI cartilage and meniscus segmentation")
    config = st.sidebar.text_input("Ensemble configuration", str(PROJECT_ROOT / "configs" / "ensemble.json"))
    files = st.file_uploader("Upload an NPY volume [N,H,W] or one DICOM series", type=["npy", "dcm"], accept_multiple_files=True)
    if not files:
        return
    try:
        if len(files) == 1 and files[0].name.lower().endswith(".npy"):
            volume = validate_volume(np.load(files[0], allow_pickle=False))
        elif all(f.name.lower().endswith(".dcm") for f in files):
            volume = load_dicom_series(files)
        else:
            raise ValueError("Select one NPY file or files from one DICOM series")
        index = st.slider("Slice index", 0, len(volume) - 1, len(volume) // 2) if len(volume) > 1 else 0
        with st.spinner("Loading models and running segmentation"):
            result = load_predictor(config).predict_slice(volume, index)
        columns = st.columns(4)
        columns[0].image(volume[index], caption="Original slice", clamp=True)
        columns[1].image(result["image"], caption="Knee ROI", clamp=True)
        columns[2].image(PALETTE[result["prediction"]], caption="Segmentation")
        import matplotlib
        heat = matplotlib.colormaps["jet"](entropy_map(result["probability"]) / np.log(5))[..., :3]
        overlay = 0.6 * result["image"][..., None] + 0.4 * heat
        columns[3].image(overlay, caption="Entropy overlay", clamp=True)
        st.caption(" | ".join(f"{i}: {name}" for i, name in enumerate(CLASS_NAMES) if i))
        st.caption(f"ROI xyxy: {result['bbox_xyxy']}")
        buffer = io.BytesIO()
        np.save(buffer, result["prediction"])
        st.download_button("Download ROI segmentation (.npy)", buffer.getvalue(), file_name=f"slice_{index:04d}_roi.npy")
    except (ValueError, FileNotFoundError, RuntimeError, ImportError) as error:
        st.error(str(error))


if __name__ == "__main__":
    main()
