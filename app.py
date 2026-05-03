from collections import deque
from pathlib import Path
import threading

import av
import cv2
import numpy as np
import streamlit as st
from streamlit_webrtc import VideoProcessorBase, WebRtcMode, webrtc_streamer
import torch
import torch.nn as nn
from torchvision import models, transforms


CLASS_NAMES = ["angry", "disgust", "fear", "happy", "neutral", "sad", "surprise"]
EMOTION_PROMPTS = {
    "angry": "Ask if something frustrating happened.",
    "disgust": "Ask if something is making them uncomfortable.",
    "fear": "Ask if they want reassurance or support.",
    "happy": "Say you're glad to see them in such a great mood.",
    "neutral": "Start with a calm and friendly check-in.",
    "sad": "Ask if everything is okay.",
    "surprise": "Ask what just caught their attention.",
}
MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]
MODEL_PATH = Path("models/clean_baseline_transfer.pth")
FALLBACK_MODEL_PATH = Path("models/fer_best_model.pth")


st.set_page_config(page_title="Seeing Clearly Demo", page_icon=":eyes:", layout="wide")

st.markdown(
    """
    <style>
    :root {
      --paper: #fbfaf7;
      --ink: #1d252c;
      --muted: #63717b;
      --line: #d9ded8;
      --accent: #157a5d;
      --panel: #ffffff;
    }
    .stApp {
      background: var(--paper);
      color: var(--ink);
    }
    .hero {
      padding: 1.1rem 1.2rem;
      border: 1px solid var(--line);
      background: var(--panel);
      margin-bottom: 1rem;
    }
    .hero h1 {
      margin: 0 0 .35rem 0;
      font-size: 2.1rem;
    }
    .hero p {
      margin: 0;
      color: var(--muted);
      max-width: 820px;
    }
    .detection-card {
      padding: 1rem;
      border: 1px solid var(--line);
      background: var(--panel);
      margin-bottom: .75rem;
    }
    .detection-card h3 {
      margin: 0 0 .35rem 0;
      color: var(--accent);
    }
    .note {
      color: var(--muted);
      font-size: .92rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def create_resnet18(num_classes):
    model = models.resnet18(weights=None)
    num_features = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Dropout(0.5),
        nn.Linear(num_features, num_classes),
    )
    return model


def resolve_model_path():
    if MODEL_PATH.exists():
        return MODEL_PATH
    return FALLBACK_MODEL_PATH


@st.cache_resource
def load_resources():
    device = get_device()
    model_path = resolve_model_path()
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
        class_names = checkpoint.get("class_names", CLASS_NAMES)
        state_dict = checkpoint["state_dict"]
    else:
        class_names = CLASS_NAMES
        state_dict = checkpoint

    model = create_resnet18(num_classes=len(class_names))
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()

    face_cascade = cv2.CascadeClassifier(
        cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    )
    if face_cascade.empty():
        raise RuntimeError("Failed to load OpenCV face detector.")

    transform = transforms.Compose(
        [
            transforms.ToPILImage(),
            transforms.Grayscale(num_output_channels=3),
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=MEAN, std=STD),
        ]
    )
    return model, class_names, face_cascade, transform, device, model_path.name


class TemporalSmoother:
    def __init__(self, history_size=6):
        self.history = deque(maxlen=history_size)

    def update(self, probabilities):
        self.history.append(probabilities)
        return np.mean(np.stack(self.history), axis=0)


def preprocess_face(face_bgr):
    gray = cv2.cvtColor(face_bgr, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    normalized = clahe.apply(gray)
    return cv2.cvtColor(normalized, cv2.COLOR_GRAY2BGR)


def predict_face(face_bgr, model, class_names, transform, device, smoother):
    processed = preprocess_face(face_bgr)
    image = transform(processed).unsqueeze(0).to(device)
    with torch.no_grad():
        logits = model(image)
        probs = torch.softmax(logits, dim=1).squeeze(0).detach().cpu().numpy()

    if smoother is not None:
        probs = smoother.update(probs)

    pred = int(np.argmax(probs))
    emotion = class_names[pred]
    confidence = float(probs[pred])
    return emotion, confidence


def draw_label(frame, lines, x, y):
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.55
    thickness = 2
    padding = 8
    line_gap = 8
    sizes = [cv2.getTextSize(line, font, font_scale, thickness)[0] for line in lines]
    width = max(size[0] for size in sizes)
    height = max(size[1] for size in sizes)
    block_height = len(lines) * height + (len(lines) - 1) * line_gap + padding * 2
    top = max(y - block_height - 10, 0)
    right = min(x + width + padding * 2, frame.shape[1] - 1)
    bottom = top + block_height

    cv2.rectangle(frame, (x, top), (right, bottom), (223, 242, 234), -1)
    text_y = top + padding + height
    for line in lines:
        cv2.putText(frame, line, (x + padding, text_y), font, font_scale, (29, 37, 44), thickness)
        text_y += height + line_gap


def analyze_frame(frame_bgr, model, class_names, face_cascade, transform, device, smoother):
    detections = []
    annotated = frame_bgr.copy()
    gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=5, minSize=(50, 50))

    for x, y, w, h in faces:
        pad_x = int(w * 0.22)
        pad_y = int(h * 0.22)
        y1 = max(y - pad_y, 0)
        y2 = min(y + h + pad_y, frame_bgr.shape[0])
        x1 = max(x - pad_x, 0)
        x2 = min(x + w + pad_x, frame_bgr.shape[1])
        face = frame_bgr[y1:y2, x1:x2]
        emotion, confidence = predict_face(face, model, class_names, transform, device, smoother)
        prompt = EMOTION_PROMPTS.get(emotion, "Treat this prediction as a soft cue.")

        detections.append({"emotion": emotion, "confidence": confidence, "prompt": prompt})
        cv2.rectangle(annotated, (x, y), (x + w, y + h), (21, 122, 93), 2)
        draw_label(annotated, [f"{emotion} ({confidence:.2f})", prompt], x, y)

    return annotated, detections


class EmotionVideoProcessor(VideoProcessorBase):
    def __init__(self):
        self.model, self.class_names, self.face_cascade, self.transform, self.device, _ = load_resources()
        self.smoother = TemporalSmoother()
        self._lock = threading.Lock()
        self._detections = []

    def recv(self, frame):
        frame_bgr = frame.to_ndarray(format="bgr24")
        annotated, detections = analyze_frame(
            frame_bgr,
            self.model,
            self.class_names,
            self.face_cascade,
            self.transform,
            self.device,
            self.smoother,
        )
        with self._lock:
            self._detections = detections
        return av.VideoFrame.from_ndarray(annotated, format="bgr24")

    def get_detections(self):
        with self._lock:
            return list(self._detections)


def render_detection_cards(detections):
    if not detections:
        st.info("No face is currently detected. Try facing the camera with steady lighting.")
        return

    for idx, detection in enumerate(detections, start=1):
        st.markdown(
            f"""
            <div class="detection-card">
              <h3>Face {idx}: {detection["emotion"].title()}</h3>
              <p><strong>Confidence:</strong> {detection["confidence"]:.1%}</p>
              <p><strong>Suggested prompt:</strong> {detection["prompt"]}</p>
            </div>
            """,
            unsafe_allow_html=True,
        )


st.markdown(
    """
    <div class="hero">
      <h1>Seeing Clearly Webcam Prototype</h1>
      <p>
        A local browser demo for the final project. The prediction is a FER-2013
        expression label, not a direct measurement of someone's internal emotion.
      </p>
    </div>
    """,
    unsafe_allow_html=True,
)

model, class_names, _, _, device, model_filename = load_resources()
left, right = st.columns([2, 1])

with left:
    st.subheader("Live camera")
    ctx = webrtc_streamer(
        key="seeing-clearly-webcam",
        mode=WebRtcMode.SENDRECV,
        video_processor_factory=EmotionVideoProcessor,
        media_stream_constraints={"video": True, "audio": False},
        async_processing=True,
        rtc_configuration={"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]},
    )

with right:
    st.subheader("Model")
    st.write(f"Checkpoint: `{model_filename}`")
    st.write(f"Device: `{device}`")
    st.write("Classes: " + ", ".join(class_names))
    st.markdown(
        '<p class="note">The labels are shown as soft cues. Low confidence or ambiguous expressions should not be treated as reliable emotional judgments.</p>',
        unsafe_allow_html=True,
    )
    st.subheader("Current detections")
    if ctx.video_processor:
        render_detection_cards(ctx.video_processor.get_detections())
    else:
        st.info("Click START to begin the webcam stream.")
