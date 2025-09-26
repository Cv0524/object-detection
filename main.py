from __future__ import annotations
import pathlib
pathlib.PosixPath = pathlib.WindowsPath
from typing import Union
from pathlib import Path
import cv2
import numpy as np
import pandas as pd
import streamlit as st
from streamlit_option_menu import option_menu
import torch

from main_pipeline import (
    load_yolov5_custom,
    detect_image,
    run_webcam_detection_streamlit,
    set_confidence,  # make sure this exists in pipeline.py
)

st.set_page_config(
    page_title="Custom Object Detection using YOLOv5",
    page_icon="👁‍🗨",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.title("👁‍🗨 Streamlit Object Detection (.pt)")

MODEL_PATH: Path = Path(
    r"best_updated_datav3.pt"
)

with st.sidebar:
    selected = option_menu(
        "Input Choice",
        ["File Upload", "Real Time Camera"],
        icons=["file", "camera"],
        menu_icon="cast",
        default_index=0,
    )

conf = st.slider("Confidence threshold", 0.05, 0.95, 0.50, 0.05)

device = "cuda" if torch.cuda.is_available() else "cpu"
st.caption(f"Device: {device}. Torch {torch.__version__}.")

@st.cache_resource(show_spinner=True)
def get_model(weights_path: Union[str, Path], force_reload: bool = False):
    return load_yolov5_custom(weights_path, force_reload=force_reload)

_model = None
def ensure_model():
    global _model
    if _model is None:
        with st.spinner("Loading YOLOv5 model (.pt)..."):
            _model = get_model(MODEL_PATH, force_reload=False)
    if _model is not None:
        set_confidence(_model, conf)
    return _model

if selected == "File Upload":
    input_image = st.file_uploader(
        "Upload Image",
        type=["png", "jpg", "jpeg", "bmp", "tiff"],
        accept_multiple_files=False,
    )
    if input_image is not None:
        file_bytes = np.asarray(bytearray(input_image.read()), dtype=np.uint8)
        opencv_image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
        if opencv_image is None:
            st.error("Failed to decode the uploaded image.")
        else:
            m = ensure_model()
            detections, annotated_bgr = detect_image(opencv_image, model=m)
            if annotated_bgr is not None:
                st.image(annotated_bgr, channels="BGR", caption="Detections")
                st.write(f"Found {len(detections)} objects")
                if len(detections):
                    st.dataframe(pd.DataFrame(detections))
                ok, buf = cv2.imencode(".jpg", annotated_bgr)
                if ok:
                    st.download_button("Download result.jpg", buf.tobytes(), "result.jpg", "image/jpeg")

if selected == "Real Time Camera":
    m = ensure_model()
    run_webcam_detection_streamlit(
        model=m,
        camera_index=0,
        conf_threshold=conf,
        save_video=False
    )
