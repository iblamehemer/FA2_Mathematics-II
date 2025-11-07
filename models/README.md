# Models Directory

This directory is intended to hold pre‑trained model weights for the PPE
compliance application.

By default, the Streamlit app looks for a file named `best.pt` in this
directory.  After training a YOLOv8 model using `train.py`, copy the
generated `runs/detect/<run_name>/weights/best.pt` file into this folder and
rename it to `best.pt`.  You can also choose to upload a different
`*.pt` file via the Streamlit sidebar at runtime.