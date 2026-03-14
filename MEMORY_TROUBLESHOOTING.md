# Memory / OOM Troubleshooting (Pneumonia XAI)

If you see **OOM (Out of Memory)** or **"The paging file is too small for this operation to complete"** when running `python app/app.py`, use the steps below.

## 1. Increase Windows paging file (virtual memory)

This often fixes "paging file is too small" and can help with OOM.

1. Press **Win + R**, type `sysdm.cpl`, Enter.
2. Open the **Advanced** tab → **Performance** → **Settings**.
3. **Advanced** tab → **Virtual memory** → **Change**.
4. Uncheck **Automatically manage paging file size for all drives**.
5. Select your system drive (e.g. **C:**) → **Custom size**:
   - **Initial size:** 4096 MB (or 1.5× your RAM if you have 8 GB).
   - **Maximum size:** 8192 MB or higher (e.g. 1.5× RAM, max 16 GB).
6. **Set** → **OK** → restart the PC if prompted.

## 2. Free RAM before starting the backend

- Close browsers (many tabs), IDEs, and other heavy apps.
- Restart the PC if the system has been on for a long time.

## 3. Backend already tuned for low memory

In `app/app.py` we:

- Set **TF_NUM_INTEROP_THREADS=1** and **TF_NUM_INTRAOP_THREADS=2** to limit TensorFlow threads.
- Limit **intra_op** and **inter_op** parallelism to reduce peak RAM.
- Run Flask with **use_reloader=False** so the model is loaded only once (no reloader process).

## 4. Optional: reduce TensorFlow memory further

Before starting the app you can set:

```powershell
$env:TF_ENABLE_ONEDNN_OPTS = "0"
$env:TF_NUM_INTRAOP_THREADS = "1"
python app/app.py
```

## 5. If it still fails

- **Train a smaller model:** In `models/train_model.py` use `MobileNetV2` instead of `ResNet50` (smaller and uses less memory).
- **Use a machine with more RAM** or run the backend on a server and use the frontend from your PC.

## Quick reference

| Error | Action |
|------|--------|
| OOM when allocating tensor | Increase paging file, close other apps, try step 4. |
| Paging file too small | Increase paging file (step 1), restart PC. |
| Crash on Flask “Restarting with stat” | Fixed by `use_reloader=False`; no code change needed. |
