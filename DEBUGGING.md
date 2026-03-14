# Pneumonia XAI – Debugging Checklist

## Flow

```
React (port 3002) → Upload X-ray
       ↓
Flask /api/predict (port 5000)
       ↓
CNN prediction → JSON { prediction, probability, all_probabilities }
       ↓
Frontend shows result; user can request Grad-CAM / LIME via /api/explain/*
```

## If prediction doesn’t show

1. **F12 → Network tab**  
   Upload an image and check:

   - Does **`/api/predict`** appear?
   - **Method**: must be **POST** (not GET).
   - **Request**: body should be **Form Data** with key **`file`** (the image file).
   - **Status**:
     - **200** → OK; check **Response** for `prediction`, `probability`, `all_probabilities`.
     - **400** → Bad request (e.g. no file, wrong field name, or invalid file type).
     - **404** → Route not found (wrong URL or proxy not hitting Flask).
     - **500** → Backend error (model not loaded, or exception in `/api/predict`).

2. **Backend**
   - Flask must be running: `python app/app.py` (or from `app` dir) so it serves on **port 5000**.
   - Model must exist: `models/pneumonia_model.h5` or `pneumonia_model_finetuned.h5`.
   - Console should show: `Loading model from ...` and `All models loaded successfully!`.

3. **Frontend**
   - React dev server on **port 3002** (or whatever you use); `package.json` has `"proxy": "http://localhost:5000"` so `/api/*` goes to Flask.
   - Open the app at **http://localhost:3002** (not http://localhost:5000) so the proxy is used.

## XAI (Grad-CAM / LIME) not loading

- Same idea: in Network tab check **`/api/explain/gradcam`** or **`/api/explain/lime`**.
- They are **POST** with **Form Data** key **`file`** (image).
- If **200**: response has `original_image`, `heatmap`, `overlay` (Grad-CAM) or `lime_explanation` (LIME).
- If **500**: often “Grad-CAM not initialized” or “LIME explainer not initialized” → model or XAI init failed on backend (check Flask console).

## Quick backend test (no frontend)

```bash
# From project root
curl -X POST -F "file=@path/to/chest_xray.png" http://localhost:5000/api/predict
```

Expected: JSON with `prediction`, `probability`, `all_probabilities`.
