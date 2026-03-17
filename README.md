# Seeing Clearly: Real-Time Emotion Recognition

Seeing Clearly is a real-time facial expression recognition system built using a fine-tuned ResNet-18 model trained on the FER-2013 dataset. The system detects faces from a webcam and predicts emotional expressions to assist users with emotion recognition.

The model identifies seven emotional states:

- Angry
- Disgust
- Fear
- Happy
- Neutral
- Sad
- Surprise

The application uses PyTorch for deep learning and OpenCV for real-time face detection and webcam integration.

---

## Repository Structure

Seeing-Clearly/

README.md  
requirements.txt  

models/  
    fer_resnet18.pth  

demo/  
    emotion_camera_demo.py  

notebooks/  
    Project_Demo.ipynb  

- emotion_camera_demo.py — runs the real-time emotion recognition demo  
- Project_Demo.ipynb — notebook containing the full model training pipeline  
- fer_resnet18.pth — trained model weights  

---

## Installation

Clone the repository:

git clone https://github.com/yourusername/Seeing-Clearly.git  
cd Seeing-Clearly

Install the required packages:

pip install -r requirements.txt

---

## Running the Real-Time Demo

Once dependencies and the model file are installed, run:

python demo/emotion_camera_demo.py

The application will:

1. Open your webcam
2. Detect faces in the video stream
3. Predict facial emotions in real time
4. Display the predicted emotion and confidence above the detected face

Press **Q** to exit the application.

---

## Training the Model

If you want to retrain the model from scratch, the full training pipeline is available in:

notebooks/Project_Demo.ipynb

This notebook includes:

- FER-2013 dataset loading
- preprocessing and augmentation
- ResNet-18 fine-tuning
- model evaluation
- model export for the live demo

---

## Requirements

- Python 3.9+
- PyTorch
- torchvision
- OpenCV
- NumPy
- Matplotlib
- scikit-learn

All dependencies are listed in requirements.txt.

---

## Hardware Requirements

The demo requires a device with an accessible webcam.  
Performance may vary depending on CPU/GPU availability.
