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
app.py

models/  
    fer_resnet18.pth  

demo/  
    emotion_camera_demo.py  

analysis/
    emotion_model_analysis.py

seeing_clearly/
    core.py

notebooks/  
    Project_Demo.ipynb  

- app.py — launches the Streamlit web app
- emotion_camera_demo.py — runs the real-time emotion recognition demo  
- emotion_model_analysis.py — generates per-class accuracy, confusion matrices, saliency maps, and a short analysis summary
- core.py — shared model loading, face detection, inference, and prompt utilities
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

To launch the browser-based app, run:

```bash
streamlit run app.py
```

The web app includes:

1. A live browser webcam stream for emotion detection
2. Real-time emotion predictions with assistive response prompts overlaid on the video
3. A built-in analysis dashboard that can generate confusion matrices and saliency maps

---

## Hosting the Web App

This project is prepared for sharing through Streamlit Community Cloud.

1. Push this repository to GitHub.
2. Go to [Streamlit Community Cloud](https://share.streamlit.io/).
3. Create a new app from your GitHub repo.
4. Set the main file path to `app.py`.
5. Deploy and share the resulting `streamlit.app` link.

The repo already includes:

- `.streamlit/config.toml` for hosted Streamlit settings
- `runtime.txt` to pin the Python version
- `requirements.txt` pinned for deployment stability
- `DEPLOY.md` with the short deployment checklist

Once deployed, the hosted app will provide the same live browser camera workflow as the local version.

Important:
Choose Python `3.11` in Streamlit Community Cloud advanced settings when you deploy. If you already deployed with another Python version, delete the app and redeploy it with Python 3.11.

---

## Running the Desktop Camera Demo

Once dependencies and the model file are installed, run:

python demo/emotion_camera_demo.py

The application will:

1. Open your webcam
2. Detect faces in the video stream
3. Predict facial emotions in real time
4. Display the predicted emotion, confidence, and an emotion-based response prompt above the detected face

Example prompts include:

- Sad: "Ask if everything is okay."
- Happy: "Say you're glad to see them in such a great mood."
- Fear: "Ask if they want reassurance or support."
- Surprise: "Ask what just caught their attention."

Press **Q** to exit the application.

---

## Running Model Analysis

To generate a deeper evaluation of the trained model, run:

```bash
python analysis/emotion_model_analysis.py
```

This script:

1. Evaluates the model on the FER-2013 test split
2. Reports per-class accuracy instead of only overall accuracy
3. Saves a confusion matrix to `analysis_outputs/confusion_matrix.png`
4. Saves saliency-map overlays for each class to `analysis_outputs/saliency_maps/`
5. Writes a short summary to `analysis_outputs/analysis_summary.md` that highlights stronger and weaker assistive-use cases

If you want a quicker smoke test, you can limit the sample count:

```bash
python analysis/emotion_model_analysis.py --max-samples 200
```

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
