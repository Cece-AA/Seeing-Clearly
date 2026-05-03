# Seeing Clearly

Seeing Clearly is an Applied Machine Learning final project about the reliability of facial expression recognition in an assistive webcam setting. The project uses transfer learning with a ResNet-18 model trained on FER-2013, then studies where that model is reliable, where it fails, and how uncertainty-aware interface choices can make the system safer to interpret.

The final submission is organized around two reproducible artifacts:

- `index.html`: the local technical blog post submitted to CourseWorks.
- `notebooks/Seeing_Clearly_Final_Reproducibility.ipynb`: the notebook used to fit or load the model and regenerate the figures used in the blog post.

The older Streamlit app, desktop webcam demo, and helper scripts are preserved on the `legacy-current-webapp` branch. This branch is intentionally focused on the final blog post, one reproducible notebook, the saved model checkpoint, and generated figures.

## Project Question

Can a transfer-learned FER-2013 classifier provide useful assistive facial-expression cues in a live webcam interface, and how should confidence, class imbalance, and temporal smoothing affect when the system speaks confidently versus hedges?

## Repository Structure

- `notebooks/Seeing_Clearly_Final_Reproducibility.ipynb` - final notebook for model fitting/loading, evaluation, calibration-style confidence analysis, and figure generation.
- `models/fer_best_model.pth` - current checkpoint used by the final analysis.
- `models/fer_best_model.json` - metadata and training history for the current checkpoint.
- `assets/figures/` - generated figures referenced by the final blog post.
- `app.py` - optional local Streamlit webcam prototype.

## Setup

```bash
python3 -m pip install -r requirements.txt
```

The notebook downloads FER-2013 through `kagglehub` unless you provide a local dataset path. The expected dataset layout is:

```text
fer2013/
  train/
    angry/
    disgust/
    fear/
    happy/
    neutral/
    sad/
    surprise/
  test/
    ...
```

## Reproducing Figures

Open and run:

```bash
jupyter notebook notebooks/Seeing_Clearly_Final_Reproducibility.ipynb
```

The notebook writes blog-ready figures to `assets/figures/`, including:

- FER-2013 sample images
- training and validation history
- normalized confusion matrix
- per-class accuracy
- confidence threshold tradeoff
- calibration-style reliability diagram

## Optional Webcam Prototype

To run the local webcam demo:

```bash
streamlit run app.py
```

The demo uses the selected clean checkpoint when available and displays FER-2013 expression labels as soft assistive cues.
