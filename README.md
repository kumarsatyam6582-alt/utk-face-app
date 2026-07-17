# UTK Face Age & Gender Predictor

A Streamlit app that detects a face in an uploaded photo and predicts age and gender using a trained Keras model.

## Project structure

```
your-repo/
├── app.py
├── requirements.txt
├── .gitignore
└── model/
    └── age_gender_model.keras
```

## 1. Get the model file into the repo

Your model is currently on Google Drive at
`/content/drive/MyDrive/age_gender_project/age_gender_model.keras`.

Download it from Colab first:

```python
from google.colab import files
files.download('/content/drive/MyDrive/age_gender_project/age_gender_model.keras')
```

Then place it at `model/age_gender_model.keras` in this project folder.

**Check the file size** — GitHub blocks files over 100 MB on a normal push.

- **Under 100 MB:** just commit it normally.
- **Over 100 MB:** use [Git LFS](https://git-lfs.com/):
  ```bash
  git lfs install
  git lfs track "*.keras"
  git add .gitattributes
  ```
  Streamlit Community Cloud does support Git LFS files.
- **Alternative for large models:** don't commit the model at all. Host it somewhere (Hugging Face Hub, a GitHub Release asset, S3, etc.) and download it at app startup in `load_my_model()` with `urllib.request.urlretrieve(...)` before calling `tf.keras.models.load_model`.

## 2. Push to GitHub

```bash
cd your-repo
git init
git add .
git commit -m "Initial commit: age/gender predictor app"
git branch -M main
git remote add origin https://github.com/<your-username>/<your-repo>.git
git push -u origin main
```

## 3. Deploy on Streamlit Community Cloud

1. Go to https://share.streamlit.io and sign in with GitHub.
2. Click **New app**.
3. Select your repo, branch (`main`), and set **Main file path** to `app.py`.
4. Click **Deploy**.

The first build will take a few minutes (installing TensorFlow). Once it's up, you'll get a public URL like `https://<your-app>.streamlit.app`.

## Notes on the changes made to app.py

- Removed the `%%writefile` line — that's a Colab notebook magic command, not valid Python; it must be gone for a real `.py` file.
- Replaced the hardcoded Google Drive path with a path relative to the script (`model/age_gender_model.keras`), so it works both locally and when deployed.
- `requirements.txt` uses `opencv-python-headless` instead of `opencv-python` — the regular package needs system GUI libraries that aren't available on Streamlit Cloud's servers and will fail to install.
- `requirements.txt` uses `tensorflow-cpu` since Streamlit Cloud's free tier has no GPU; this installs faster and uses less memory. Swap back to `tensorflow` if you need GPU-specific ops.
