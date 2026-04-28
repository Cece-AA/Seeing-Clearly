# Deploy Seeing Clearly

## Streamlit Community Cloud

1. Push this repository to GitHub.
2. Go to https://share.streamlit.io/
3. Create a new app.
4. Choose this repository and branch.
5. Open "Advanced settings."
6. Set the Python version to `3.11`.
7. Set the main file path to `app.py`.
8. Deploy.

## Notes

- The app expects a trained checkpoint in `models/fer_best_model.pth`.
- The deployed app uses WebRTC for live camera streaming.
- If the camera does not start right away, allow browser camera permissions and refresh once.
- `requirements.txt` is pinned for deployment stability on Python 3.11.
- If you already deployed the app with Python 3.14, delete that app and redeploy with Python 3.11 because Python can't be changed in place on Streamlit Community Cloud.
