Place your checkpoint files here:

  weights/dfine/best_stg1.pth
  weights/rfdetr/checkpoint_best_regular.pth

The registry auto-discovers any folder containing a manifest.json.
To add a new checkpoint:
  1. Create a new subfolder, e.g.  weights/dfine_v2/
  2. Drop the .pth file inside it
  3. Copy and edit manifest.json (update weights_file, num_classes, class_names, display_name)
  4. Restart uvicorn (or call POST /api/models/{model_id}/load from the UI)
