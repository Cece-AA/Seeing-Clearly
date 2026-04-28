from pathlib import Path
import subprocess
import sys
import threading

import streamlit as st
from streamlit_webrtc import VideoProcessorBase, WebRtcMode, webrtc_streamer
import av

from seeing_clearly.core import (
    analyze_frame,
    build_inference_transform,
    build_model,
    get_device,
    load_checkpoint,
    load_face_cascade,
    resolve_model_path,
    TemporalEmotionSmoother,
)


st.set_page_config(
    page_title="Seeing Clearly",
    page_icon=":eyes:",
    layout="wide",
)

RTC_CONFIGURATION = {
    "iceServers": [
        {"urls": ["stun:stun.l.google.com:19302"]},
        {"urls": ["stun:stun1.l.google.com:19302"]},
    ]
}

st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@400;500;700&family=IBM+Plex+Sans:wght@400;500;600&display=swap');

    :root {
        --bg: #f4efe6;
        --ink: #162027;
        --card: #fffaf0;
        --accent: #46dab0;
        --accent-deep: #13795b;
        --warm: #f6b26b;
        --line: rgba(22, 32, 39, 0.12);
    }

    .stApp {
        background:
            radial-gradient(circle at top left, rgba(246, 178, 107, 0.28), transparent 34%),
            radial-gradient(circle at top right, rgba(70, 218, 176, 0.26), transparent 30%),
            linear-gradient(180deg, #f7f3eb 0%, #f1eadf 100%);
        color: var(--ink);
        font-family: "IBM Plex Sans", sans-serif;
    }

    h1, h2, h3 {
        font-family: "Space Grotesk", sans-serif;
        letter-spacing: -0.02em;
    }

    .hero {
        padding: 1.4rem 1.5rem;
        border: 1px solid var(--line);
        border-radius: 24px;
        background: linear-gradient(135deg, rgba(255,250,240,0.96), rgba(255,246,228,0.86));
        box-shadow: 0 18px 50px rgba(22, 32, 39, 0.08);
        margin-bottom: 1rem;
    }

    .metric-card {
        padding: 1rem 1.1rem;
        border-radius: 20px;
        background: rgba(255, 250, 240, 0.96);
        border: 1px solid var(--line);
        box-shadow: 0 12px 32px rgba(22, 32, 39, 0.06);
        margin-bottom: 0.8rem;
    }

    .metric-card h4 {
        margin: 0 0 0.2rem 0;
        font-family: "Space Grotesk", sans-serif;
    }

    .metric-card p {
        margin: 0.12rem 0;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource
def load_resources():
    device = get_device()
    model_path = resolve_model_path()
    checkpoint = load_checkpoint(model_path, device=device)
    model = build_model(model_path=model_path, device=device)
    face_cascade = load_face_cascade()
    infer_tf = build_inference_transform(checkpoint["architecture"])
    return model, face_cascade, infer_tf, device, model_path.name, checkpoint["architecture"]


def render_detection_cards(detections):
    if not detections:
        st.warning("No face is currently detected. Try better lighting or face the camera more directly.")
        return

    st.subheader("Emotion Guidance")
    cols = st.columns(min(3, len(detections)))
    for idx, detection in enumerate(detections):
        with cols[idx % len(cols)]:
            st.markdown(
                f"""
                <div class="metric-card">
                    <h4>Face {idx + 1}: {detection["emotion"].title()}</h4>
                    <p><strong>Confidence:</strong> {detection["confidence"]:.2%}</p>
                    <p><strong>Suggested response:</strong> {detection["prompt"]}</p>
                </div>
                """,
                unsafe_allow_html=True,
            )


class EmotionVideoProcessor(VideoProcessorBase):
    def __init__(self):
        self.model, self.face_cascade, self.infer_tf, self.device, _, _ = load_resources()
        self._lock = threading.Lock()
        self._detections = []
        self.smoother = TemporalEmotionSmoother()

    def recv(self, frame):
        frame_bgr = frame.to_ndarray(format="bgr24")
        annotated_bgr, detections = analyze_frame(
            frame_bgr,
            self.model,
            self.face_cascade,
            self.infer_tf,
            device=self.device,
            smoother=self.smoother,
        )
        with self._lock:
            self._detections = detections
        return av.VideoFrame.from_ndarray(annotated_bgr, format="bgr24")

    def get_detections(self):
        with self._lock:
            return list(self._detections)


def run_analysis(max_samples):
    command = [
        sys.executable,
        "analysis/emotion_model_analysis.py",
        "--output-dir",
        "analysis_outputs",
    ]
    if max_samples:
        command.extend(["--max-samples", str(max_samples)])

    completed = subprocess.run(
        command,
        capture_output=True,
        text=True,
        cwd=Path(__file__).resolve().parent,
        check=False,
    )
    return completed


def render_analysis_outputs():
    output_dir = Path("analysis_outputs")
    summary_path = output_dir / "analysis_summary.md"
    confusion_path = output_dir / "confusion_matrix.png"
    saliency_dir = output_dir / "saliency_maps"

    if summary_path.exists():
        st.subheader("Model Summary")
        st.markdown(summary_path.read_text())

    if confusion_path.exists():
        st.subheader("Confusion Matrix")
        st.image(str(confusion_path), use_container_width=True)

    if saliency_dir.exists():
        saliency_images = sorted(saliency_dir.glob("*.png"))
        if saliency_images:
            st.subheader("Saliency Maps")
            for image_path in saliency_images:
                st.image(str(image_path), caption=image_path.stem.replace("_", " ").title(), use_container_width=True)


st.markdown(
    """
    <div class="hero">
        <h1>Seeing Clearly</h1>
        <p>Browser-based emotion recognition with assistive response prompts, plus a built-in evaluation dashboard for class-level performance and saliency maps.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

tab_detect, tab_analysis = st.tabs(["Emotion Detection", "Model Analysis"])

with tab_detect:
    _, _, _, _, model_filename, architecture = load_resources()
    status_left, status_right = st.columns(2)
    with status_left:
        st.caption(f"Loaded model: `{model_filename}`")
    with status_right:
        st.caption(f"Architecture: `{architecture}`")

    st.subheader("Live Camera Feed")
    st.write(
        "Start the webcam below to run real-time emotion detection in the browser. "
        "The video stream is annotated live with the predicted emotion and suggested response prompt."
    )

    ctx = webrtc_streamer(
        key="emotion-live-stream",
        mode=WebRtcMode.SENDRECV,
        video_processor_factory=EmotionVideoProcessor,
        media_stream_constraints={"video": True, "audio": False},
        async_processing=True,
        rtc_configuration=RTC_CONFIGURATION,
    )

    if ctx.video_processor:
        current_detections = ctx.video_processor.get_detections()
        render_detection_cards(current_detections)
    else:
        st.info("Click `START` to begin the live camera stream.")
        st.caption("If camera access fails on a hosted link, allow browser camera permissions and refresh once.")

with tab_analysis:
    st.subheader("Generate Evaluation Artifacts")
    st.write(
        "Run the saved analysis pipeline from the browser to produce per-class accuracy, confusion matrices, "
        "and saliency maps tied back to assistive-use strengths and limitations."
    )

    max_samples = st.number_input(
        "Optional sample limit for a faster run",
        min_value=0,
        value=0,
        step=100,
        help="Set this above 0 for a quicker partial analysis.",
    )

    if st.button("Run Model Analysis", use_container_width=True):
        with st.spinner("Running evaluation and generating visualizations..."):
            completed = run_analysis(max_samples if max_samples > 0 else None)

        if completed.returncode == 0:
            st.success("Analysis complete.")
        else:
            st.error("The analysis run did not finish successfully.")
            if completed.stderr:
                st.code(completed.stderr)

    render_analysis_outputs()
