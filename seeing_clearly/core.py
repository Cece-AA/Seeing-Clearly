from pathlib import Path
from collections import deque

import cv2
import numpy as np
import torch
import torch.nn as nn
from torchvision import models, transforms
try:
    from facenet_pytorch import InceptionResnetV1
except ImportError:  # pragma: no cover - optional dependency
    InceptionResnetV1 = None


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
DEFAULT_MODEL_PATH = Path("models/fer_best_model.pth")
LEGACY_MODEL_PATH = Path("models/fer_resnet18.pth")


class FaceEmotionModel(nn.Module):
    def __init__(self, num_classes, pretrained=False):
        super().__init__()
        if InceptionResnetV1 is None:
            raise ImportError("facenet-pytorch is required for inceptionresnetv1 models.")
        pretrained_source = "vggface2" if pretrained else None
        self.backbone = InceptionResnetV1(pretrained=pretrained_source, classify=False)
        self.dropout = nn.Dropout(0.4)
        self.classifier = nn.Linear(512, num_classes)

    def forward(self, x):
        embedding = self.backbone(x)
        embedding = self.dropout(embedding)
        return self.classifier(embedding)


def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def resolve_model_path(model_path=None):
    if model_path is not None:
        return Path(model_path)
    if DEFAULT_MODEL_PATH.exists():
        return DEFAULT_MODEL_PATH
    return LEGACY_MODEL_PATH


def create_model(architecture="resnet18", num_classes=None, pretrained=False):
    num_classes = num_classes or len(CLASS_NAMES)

    if architecture == "resnet18":
        weights = models.ResNet18_Weights.DEFAULT if pretrained else None
        model = models.resnet18(weights=weights)
        num_features = model.fc.in_features
        model.fc = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(num_features, num_classes)
        )
        return model

    if architecture == "efficientnet_b0":
        weights = models.EfficientNet_B0_Weights.DEFAULT if pretrained else None
        model = models.efficientnet_b0(weights=weights)
        num_features = model.classifier[1].in_features
        model.classifier = nn.Sequential(
            nn.Dropout(0.4),
            nn.Linear(num_features, num_classes)
        )
        return model

    if architecture == "efficientnet_b2":
        weights = models.EfficientNet_B2_Weights.DEFAULT if pretrained else None
        model = models.efficientnet_b2(weights=weights)
        num_features = model.classifier[1].in_features
        model.classifier = nn.Sequential(
            nn.Dropout(0.4),
            nn.Linear(num_features, num_classes)
        )
        return model

    if architecture == "inceptionresnetv1":
        return FaceEmotionModel(num_classes=num_classes, pretrained=pretrained)

    raise ValueError(f"Unsupported architecture: {architecture}")


def load_checkpoint(model_path, device=None):
    device = device or get_device()
    checkpoint = torch.load(model_path, map_location=device)

    if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
        class_names = checkpoint.get("class_names", CLASS_NAMES)
        architecture = checkpoint.get("architecture", "resnet18")
        state_dict = checkpoint["state_dict"]
    else:
        class_names = CLASS_NAMES
        architecture = "resnet18"
        state_dict = checkpoint

    return {
        "architecture": architecture,
        "class_names": class_names,
        "state_dict": state_dict,
    }


def build_model(model_path=DEFAULT_MODEL_PATH, device=None):
    device = device or get_device()
    model_path = resolve_model_path(model_path)
    checkpoint = load_checkpoint(model_path, device=device)
    model = create_model(
        architecture=checkpoint["architecture"],
        num_classes=len(checkpoint["class_names"]),
        pretrained=False,
    )

    model.load_state_dict(checkpoint["state_dict"])
    model.to(device)
    model.eval()
    return model


def get_input_size(architecture):
    if architecture == "inceptionresnetv1":
        return 160
    if architecture == "efficientnet_b2":
        return 260
    return 224


def build_inference_transform(architecture="resnet18"):
    image_size = get_input_size(architecture)
    return transforms.Compose([
        transforms.ToPILImage(),
        transforms.Grayscale(num_output_channels=3),
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=MEAN, std=STD)
    ])


def build_dataset_transform(architecture="resnet18"):
    image_size = get_input_size(architecture)
    return transforms.Compose([
        transforms.Grayscale(num_output_channels=3),
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=MEAN, std=STD)
    ])


def load_face_cascade():
    cascade = cv2.CascadeClassifier(
        cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    )
    if cascade.empty():
        raise RuntimeError("Failed to load OpenCV face detector.")
    return cascade


def detect_faces(frame_bgr, face_cascade):
    gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    return face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.2,
        minNeighbors=5,
        minSize=(50, 50)
    )


def expand_box(box, frame_shape, padding_ratio=0.22):
    x, y, w, h = box
    pad_x = int(w * padding_ratio)
    pad_y = int(h * padding_ratio)
    max_h, max_w = frame_shape[:2]

    x1 = max(x - pad_x, 0)
    y1 = max(y - pad_y, 0)
    x2 = min(x + w + pad_x, max_w)
    y2 = min(y + h + pad_y, max_h)
    return x1, y1, x2, y2


def preprocess_face(face_bgr):
    gray = cv2.cvtColor(face_bgr, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    normalized = clahe.apply(gray)
    return cv2.cvtColor(normalized, cv2.COLOR_GRAY2BGR)


def predict_emotion(face_bgr, model, infer_tf, device=None):
    device = device or get_device()
    processed_face = preprocess_face(face_bgr)
    image = infer_tf(processed_face).unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = model(image)
        probs = torch.softmax(outputs, dim=1)
        pred = int(torch.argmax(probs).item())
        confidence = float(torch.max(probs).item())
        probabilities = probs.squeeze(0).detach().cpu().numpy()

    emotion = CLASS_NAMES[pred]
    return {
        "emotion": emotion,
        "confidence": confidence,
        "prompt": EMOTION_PROMPTS[emotion],
        "class_index": pred,
        "probabilities": probabilities,
    }


class TemporalEmotionSmoother:
    def __init__(self, distance_threshold=120, max_missing=12, history_size=6):
        self.distance_threshold = distance_threshold
        self.max_missing = max_missing
        self.history_size = history_size
        self.tracks = {}
        self.next_track_id = 0

    def _center(self, box):
        x, y, w, h = box
        return np.array([x + w / 2, y + h / 2], dtype=float)

    def _best_track(self, box):
        center = self._center(box)
        best_track_id = None
        best_distance = None

        for track_id, track in self.tracks.items():
            distance = np.linalg.norm(center - track["center"])
            if distance <= self.distance_threshold and (best_distance is None or distance < best_distance):
                best_track_id = track_id
                best_distance = distance

        return best_track_id, center

    def _create_track(self, center, probabilities):
        track_id = self.next_track_id
        self.next_track_id += 1
        self.tracks[track_id] = {
            "center": center,
            "prob_history": deque([probabilities], maxlen=self.history_size),
            "missing": 0,
        }
        return track_id

    def update(self, box, probabilities):
        track_id, center = self._best_track(box)
        if track_id is None:
            track_id = self._create_track(center, probabilities)
        else:
            track = self.tracks[track_id]
            track["center"] = center
            track["prob_history"].append(probabilities)
            track["missing"] = 0

        current_track = self.tracks[track_id]
        smoothed = np.mean(np.stack(current_track["prob_history"]), axis=0)
        return track_id, smoothed

    def tick(self, matched_track_ids):
        stale_track_ids = []
        for track_id, track in self.tracks.items():
            if track_id in matched_track_ids:
                continue
            track["missing"] += 1
            if track["missing"] > self.max_missing:
                stale_track_ids.append(track_id)

        for track_id in stale_track_ids:
            del self.tracks[track_id]


def draw_label_block(frame, lines, x, y, background=(70, 218, 176), text_color=(16, 24, 32)):
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.55
    thickness = 2
    line_gap = 8
    padding = 8

    sizes = [cv2.getTextSize(line, font, font_scale, thickness)[0] for line in lines]
    text_width = max(width for width, _ in sizes)
    line_height = max(height for _, height in sizes)
    block_height = len(lines) * line_height + (len(lines) - 1) * line_gap + padding * 2
    top = max(y - block_height - 10, 0)
    bottom = top + block_height
    right = min(x + text_width + padding * 2, frame.shape[1] - 1)

    cv2.rectangle(frame, (x, top), (right, bottom), background, -1)

    text_y = top + padding + line_height
    for line in lines:
        cv2.putText(
            frame,
            line,
            (x + padding, text_y),
            font,
            font_scale,
            text_color,
            thickness
        )
        text_y += line_height + line_gap


def analyze_frame(frame_bgr, model, face_cascade, infer_tf, device=None, smoother=None):
    device = device or get_device()
    detections = []
    annotated = frame_bgr.copy()
    matched_track_ids = set()

    for x, y, w, h in detect_faces(frame_bgr, face_cascade):
        x1, y1, x2, y2 = expand_box((x, y, w, h), frame_bgr.shape)
        face = frame_bgr[y1:y2, x1:x2]
        result = predict_emotion(face, model, infer_tf, device=device)

        if smoother is not None:
            track_id, smoothed_probs = smoother.update((x, y, w, h), result["probabilities"])
            matched_track_ids.add(track_id)
            smoothed_class = int(np.argmax(smoothed_probs))
            result["class_index"] = smoothed_class
            result["emotion"] = CLASS_NAMES[smoothed_class]
            result["confidence"] = float(smoothed_probs[smoothed_class])
            result["prompt"] = EMOTION_PROMPTS[result["emotion"]]
            result["probabilities"] = smoothed_probs

        result.update({"box": (int(x), int(y), int(w), int(h))})
        detections.append(result)

        cv2.rectangle(annotated, (x, y), (x + w, y + h), (70, 218, 176), 2)
        label = f"{result['emotion']} ({result['confidence']:.2f})"
        draw_label_block(annotated, [label, result["prompt"]], x, y)

    if smoother is not None:
        smoother.tick(matched_track_ids)

    return annotated, detections


def bgr_to_rgb(image):
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


def rgb_to_bgr(image):
    return cv2.cvtColor(image, cv2.COLOR_RGB2BGR)


def unnormalize_image(image_tensor):
    image = image_tensor.detach().cpu().clone().numpy().transpose(1, 2, 0)
    image = image * np.array(STD) + np.array(MEAN)
    return np.clip(image, 0, 1)
