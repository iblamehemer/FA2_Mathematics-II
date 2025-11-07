import io
import os
from pathlib import Path
import time
import numpy as np
import pandas as pd
from PIL import Image
import streamlit as st
import plotly.express as px

# 🔐 try to import YOLO + cv2 safely
YOLO_AVAILABLE = True
try:
    from ultralytics import YOLO
    import cv2
except Exception:
    YOLO_AVAILABLE = False

# 👉 configure page (do this once)
st.set_page_config(page_title="PPE Compliance Monitor", page_icon="🦺", layout="wide")

st.title("🦺 Computer Vision PPE Compliance Dashboard")
st.caption("Upload an image or use your webcam to detect PPE and classify worker compliance (Green/Yellow/Red).")

# ❗ if ultralytics/opencv didn’t import, show demo instead of dying
if not YOLO_AVAILABLE:
    st.warning(
        "Ultralytics / OpenCV could not be imported on Streamlit Cloud.\n"
        "This is a demo view of the app. To run full detection, run locally:\n"
        "`pip install -r requirements.txt` and then `streamlit run app.py`."
    )
    demo_df = pd.DataFrame([
        {"image_index": 1, "person_id": 1, "status": "GREEN", "present_items": "helmet, vest", "missing_items": "", "violations": 0, "num_ppe_detected": 2},
        {"image_index": 1, "person_id": 2, "status": "RED", "present_items": "", "missing_items": "All PPE", "violations": 1, "num_ppe_detected": 0},
    ])
    st.subheader("Sample Compliance Output")
    st.dataframe(demo_df)
    st.stop()from compliance import (
    DEFAULT_PPE_CLASSES,
    compute_iou,
    assign_ppe_to_people,
    compute_compliance_for_people,
    draw_annotations,
)

# ───────── sidebar ─────────
st.sidebar.title("🦺 PPE Compliance Monitor")
st.sidebar.write("Upload an image or capture from webcam.")

conf = st.sidebar.slider("Detection confidence", 0.05, 0.8, 0.25, 0.01)
iou_thr = st.sidebar.slider("IoU threshold (associate PPE→person)", 0.05, 0.7, 0.25, 0.01)
max_det = st.sidebar.slider("Max detections", 50, 300, 100, 10)
show_ppe_boxes = st.sidebar.checkbox("Show individual PPE boxes", True)
show_person_boxes = st.sidebar.checkbox("Show person boxes", True)
show_labels = st.sidebar.checkbox("Show labels", True)

st.sidebar.markdown("---")
st.sidebar.subheader("Weights")
weights_src = st.sidebar.radio(
    "Load weights from:",
    ["models/best.pt (recommended)", "Upload .pt file"],
)
uploaded_weights = None
if weights_src == "Upload .pt file":
    uploaded_weights = st.sidebar.file_uploader("Upload a YOLO .pt weights file", type=["pt"])


@st.cache_resource(show_spinner=True)
def load_model(_uploaded_bytes=None):
    if _uploaded_bytes:
        tmp_path = Path("models/uploaded_best.pt")
        tmp_path.parent.mkdir(parents=True, exist_ok=True)
        with open(tmp_path, "wb") as f:
            f.write(_uploaded_bytes.read())
        model_path = str(tmp_path)
    else:
        default_best = Path("models/best.pt")
        if default_best.exists():
            model_path = str(default_best)
        else:
            st.warning(
                "models/best.pt not found. Falling back to 'yolov8n.pt' (COCO) — this won't detect PPE classes."
            )
            model_path = "yolov8n.pt"
    model = YOLO(model_path)
    return model


model = load_model(uploaded_weights if weights_src == "Upload .pt file" else None)

# ───────── main tabs ─────────
tab1, tab2 = st.tabs(["📤 Upload Image", "📷 Camera"])
with tab1:
    image_files = st.file_uploader(
        "Upload one or more images", type=["jpg", "jpeg", "png"], accept_multiple_files=True
    )
with tab2:
    cam_img = st.camera_input("Capture from webcam")
    if cam_img is not None:
        image_files = [cam_img]

if not image_files:
    st.info("Upload an image or take a photo to begin.")
    st.stop()

all_people_records = []

for idx, up in enumerate(image_files, start=1):
    bytes_data = up.read()
    image = Image.open(io.BytesIO(bytes_data)).convert("RGB")
    img_np = np.array(image)

    try:
        results = model.predict(source=img_np, conf=conf, iou=0.45, max_det=int(max_det), verbose=False)
    except Exception as e:
        st.error(f"Model inference failed: {e}")
        st.stop()

    res = results[0]
    if res.boxes is None or len(res.boxes) == 0:
        st.warning(f"No objects detected in image {idx}.")
        continue

    boxes_xyxy = res.boxes.xyxy.cpu().numpy()
    classes = res.boxes.cls.cpu().numpy().astype(int)
    confs = res.boxes.conf.cpu().numpy()
    model_names = res.names if hasattr(res, "names") else {i: str(i) for i in np.unique(classes)}

    dets = []
    for (x1, y1, x2, y2), c, cf in zip(boxes_xyxy, classes, confs):
        dets.append(
            {
                "xyxy": [float(x1), float(y1), float(x2), float(y2)],
                "cls_id": int(c),
                "cls_name": model_names.get(int(c), str(int(c))),
                "conf": float(cf),
            }
        )

    people, ppe = assign_ppe_to_people(dets, model_names)
    people_status, annotated = compute_compliance_for_people(
        img_np,
        people,
        ppe,
        iou_thr=iou_thr,
        show_ppe_boxes=show_ppe_boxes,
        show_person_boxes=show_person_boxes,
        show_labels=show_labels,
    )

    st.subheader(f"Image {idx}")
    st.image(annotated, caption="Detections & Compliance", use_column_width=True)

    df = pd.DataFrame(people_status)
    if not df.empty:
        df["image_index"] = idx
        all_people_records.append(df)
        st.dataframe(
            df[
                [
                    "image_index",
                    "person_id",
                    "status",
                    "present_items",
                    "missing_items",
                    "violations",
                    "num_ppe_detected",
                ]
            ]
        )

# ───────── summary / report ─────────
if len(all_people_records):
    report = pd.concat(all_people_records, ignore_index=True)
    summary = report["status"].value_counts().reindex(["GREEN", "YELLOW", "RED"], fill_value=0)
    c1, c2, c3 = st.columns(3)
    c1.metric("✅ Fully Compliant", int(summary.get("GREEN", 0)))
    c2.metric("⚠️ Partially Compliant", int(summary.get("YELLOW", 0)))
    c3.metric("❌ Non-Compliant", int(summary.get("RED", 0)))

    pie_df = summary.reset_index()
    pie_df.columns = ["status", "count"]
    fig = px.pie(pie_df, names="status", values="count", hole=0.4, title="Compliance Breakdown")
    st.plotly_chart(fig, use_container_width=True)

    csv_bytes = report.to_csv(index=False).encode("utf-8")
    st.download_button(
        "📥 Download Compliance Report (CSV)",
        data=csv_bytes,
        file_name="ppe_compliance_report.csv",
        mime="text/csv",
    )
else:
    st.info("No people detected across the submitted images.")
