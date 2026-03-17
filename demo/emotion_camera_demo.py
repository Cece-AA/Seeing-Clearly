import cv2
import torch
import torch.nn as nn
import os
from torchvision import models, transforms
from PIL import Image

# Setup device and class names from training
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
class_names = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']

# Path to model file in the models subfolder
# Adjust this path based on where you are running the script from
model_path = os.path.join("models", "fer_resnet18.pth")

# Initialize the ResNet18 architecture
model = models.resnet18()
num_ftrs = model.fc.in_features

# This part MUST match the training notebook's Sequential block
model.fc = nn.Sequential(
    nn.Dropout(0.5),
    nn.Linear(num_ftrs, len(class_names))
)

# Load the saved state dictionary
if os.path.exists(model_path):
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
else:
    print(f"Error: Model file not found at {model_path}")
    exit()

# Face detector
face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

# Inference transforms matching the notebook evaluation settings
infer_tf = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Grayscale(num_output_channels=3),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

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

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces in the frame
    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.2,
        minNeighbors=5,
        minSize=(50, 50)
    )

    for (x, y, w, h) in faces:
        # Crop the face for the model
        face = frame[y:y+h, x:x+w]

        # Preprocess the crop
        img = infer_tf(face).unsqueeze(0).to(device)

        with torch.no_grad():
            outputs = model(img)
            probs = torch.softmax(outputs, dim=1)
            pred = torch.argmax(probs).item()
            conf = torch.max(probs).item()

        label = f"{class_names[pred]} ({conf:.2f})"

        # Draw the rectangle and label on the original frame
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
        cv2.putText(
            frame,
            label,
            (x, y-10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 0),
            2
        )

    # Display the result
    cv2.imshow("Emotion Detector", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()