from pathlib import Path
import sys

import cv2

sys.path.append(str(Path(__file__).resolve().parents[1]))

from seeing_clearly.core import (
    TemporalEmotionSmoother,
    analyze_frame,
    build_inference_transform,
    build_model,
    get_device,
    load_checkpoint,
    load_face_cascade,
    resolve_model_path,
)


device = get_device()
model_path = resolve_model_path()

if not model_path.exists():
    print(f"Error: Model file not found at {model_path}")
    raise SystemExit(1)

model = build_model(model_path=model_path, device=device)
checkpoint = load_checkpoint(model_path, device=device)
face_cascade = load_face_cascade()
infer_tf = build_inference_transform(checkpoint["architecture"])
smoother = TemporalEmotionSmoother()

# Start webcam capture
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Could not access the webcam.")
    exit()

print("Press Q to quit")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame, _ = analyze_frame(frame, model, face_cascade, infer_tf, device=device, smoother=smoother)

    # Display the result
    cv2.imshow("Emotion Detector", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
