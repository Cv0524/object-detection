from __future__ import annotations
import pathlib
pathlib.PosixPath = pathlib.WindowsPath
import os
import time
from typing import Union
import cv2
import numpy as np
import torch
import streamlit as st

Device = "cuda" if torch.cuda.is_available() else "cpu"

@st.cache_resource(show_spinner=True)
def load_yolov5_custom(weights_path: Union[str, os.PathLike], force_reload: bool = False):
    try:
        # Optional: torch.hub.set_dir / TORCH_HOME if needed
        model = torch.hub.load(
            "ultralytics/yolov5",
            "custom",
            path=str(weights_path),
            force_reload=force_reload,
        )
        try:
            if Device == "cuda":
                model.to("cuda")
        except Exception:
            pass
        return model
    except Exception as e:
        st.error(f"Failed to load YOLOv5 model from '{weights_path}': {e}")
        return None

def set_confidence(model, conf: float):
    try:
        model.conf = float(conf)
    except Exception:
        pass

def ensure_bgr(image_input: Union[str, os.PathLike, np.ndarray]) -> np.ndarray:
    if isinstance(image_input, (str, os.PathLike)):
        img = cv2.imread(str(image_input), cv2.IMREAD_COLOR)
        if img is None:
            raise ValueError(f"Could not load image from path: {image_input}")
        return img
    if isinstance(image_input, np.ndarray):
        if image_input.ndim == 2:
            return cv2.cvtColor(image_input, cv2.COLOR_GRAY2BGR)
        if image_input.ndim == 3 and image_input.shape[2] == 3:
            return image_input
        raise ValueError("Unsupported ndarray; expected HxWx3 BGR or HxW grayscale.")
    raise TypeError("image_input must be a path (str/PathLike) or a NumPy ndarray.")

def detect_image(image_input, model):
    if model is None:
        st.error("Model not loaded.")
        return [], None
    try:
        image_bgr = ensure_bgr(image_input)
        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        results = model(image_rgb)
        detections = results.pandas().xyxy[0].to_dict("records")
        annotated_rgb = results.render()[0]
        annotated_bgr = cv2.cvtColor(annotated_rgb, cv2.COLOR_RGB2BGR)
        return detections, annotated_bgr
    except Exception as e:
        st.error(f"Detection failed: {e}")
        return [], None

def run_webcam_detection_streamlit(
    model,
    camera_index: int = 0,
    conf_threshold: float = 0.50,
    save_video: bool = False,
    output_path: Union[str, os.PathLike] = "camera_detection.mp4",
):
    if model is None:
        st.error("Model could not be loaded.")
        return

    set_confidence(model, conf_threshold)

    if "camera_running" not in st.session_state:
        st.session_state.camera_running = True

    frame_placeholder = st.empty()
    info_placeholder = st.empty()

    cap = None
    writer = None
    frame_count = 0
    total_inference_time = 0.0
    start_time = time.time()

    if st.session_state.camera_running:
        try:
            cap = cv2.VideoCapture(camera_index)
            if not cap.isOpened():
                st.error(f"Cannot open camera index {camera_index}")
                st.session_state.camera_running = False
                return

            cap.set(cv2.CAP_PROP_FRAME_WIDTH, 800)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 500)
            cap.set(cv2.CAP_PROP_FPS, 30)

            if save_video:
                fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                writer = cv2.VideoWriter(str(output_path), fourcc, 30, (800, 500))

            while st.session_state.camera_running:
                ret, frame = cap.read()
                if not ret:
                    st.warning("Failed to capture frame from camera.")
                    break

                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                t0 = time.time()
                results = model(frame_rgb)
                dt = time.time() - t0

                annotated = results.render()[0]
                annotated_bgr = cv2.cvtColor(annotated, cv2.COLOR_RGB2BGR)

                frame_count += 1
                total_inference_time += dt
                elapsed = time.time() - start_time
                avg_fps = frame_count / elapsed if elapsed > 0 else 0.0

                cv2.putText(annotated_bgr, f"FPS: {avg_fps:.1f}", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                cv2.putText(annotated_bgr, f"Inference: {dt*1000:.1f}ms", (10, 60),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

                frame_placeholder.image(annotated_bgr, channels="BGR")

                if save_video and writer:
                    writer.write(cv2.resize(annotated_bgr, (800, 500)))

                if frame_count % 30 == 0:
                    dets = results.pandas().xyxy[0].to_dict('records')
                    info_text = f"Frame {frame_count}: Found {len(dets)} objects\n"
                    for det in dets:
                        name = det.get('name', 'obj')
                        conf_v = det.get('confidence', 0.0)
                        info_text += f" - {name} : {conf_v:.2f}\n"
                    info_placeholder.text(info_text)

                time.sleep(0.001)

        finally:
            if cap is not None:
                cap.release()
            if writer is not None:
                writer.release()
            cv2.destroyAllWindows()

            if frame_count > 0:
                avg_fps = frame_count / max(1e-6, (time.time() - start_time))
                avg_inf_ms = (total_inference_time / frame_count) * 1000
                st.write(
                    f"Session ended. Processed {frame_count} frames. "
                    f"Average FPS: {avg_fps:.2f}, Average inference time: {avg_inf_ms:.1f} ms"
                )
    else:
        st.info("Camera detection stopped.")
