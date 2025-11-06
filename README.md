# FA2_Mathematics-II
# 🦺 PPE Compliance Monitor using AI and Computer Vision  
**Author:** Hemer Pandya  
**Subject:** Mathematics AI (FA-2 Project)  
**Technology Stack:** Python · Streamlit · YOLOv8 · OpenCV · NumPy · Pandas  

---

## 📖 Overview  
This project demonstrates the **real-world application of Artificial Intelligence and Mathematics** through a **PPE (Personal Protective Equipment) Compliance Detection System**.  

Using a **YOLOv8 object detection model**, the system identifies **workers** and their **safety gear** (helmets, vests, masks, etc.) from uploaded images or live webcam feeds. The app classifies each individual into three compliance levels:  
- ✅ **Green** – Fully compliant  
- ⚠️ **Yellow** – Partially compliant *(optional in extended model)*  
- ❌ **Red** – Non-compliant  

The project is built using **Streamlit**, allowing an interactive dashboard that visualizes detections, compliance summaries, and downloadable reports — bridging **AI concepts with mathematical analysis**.

---

## 🎯 Objectives  
- Apply **mathematical modelling** and **AI principles** to a real-world industrial safety problem.  
- Use **functions**, **probabilities**, and **matrices** in interpreting detection outcomes.  
- Develop an **interactive web app** using Python and Streamlit.  
- Integrate **YOLOv8** to perform image-based PPE detection efficiently.  

---

## 🧠 How It Works  
1. **Image Input:** Upload an image or capture from webcam.  
2. **Model Inference:** The YOLO model detects objects (people and PPE).  
3. **Computation:** IoU (Intersection-over-Union) values and detection confidences are used to mathematically assign PPE to individuals.  
4. **Classification:** Each person is marked as compliant or non-compliant based on overlapping PPE detections.  
5. **Visualization:** Results are drawn directly on the image with bounding boxes and compliance colors.  
6. **Reporting:** The app generates a summary dashboard and allows downloading a detailed CSV report.

---

## 🧮 Mathematical Connections  
- **Functions & Modelling:** Detection confidence thresholds are treated as functional parameters.  
- **Matrix Operations:** YOLO internally relies on matrix multiplication for neural network propagation.  
- **Probability & Statistics:** Compliance is evaluated using detection probabilities and confidence intervals.  
- **Graphical Representation:** Pie charts represent compliance distribution percentages.  

---

## ⚙️ Installation & Usage  

### 1️⃣ Clone the repository  
```bash
git clone https://github.com/<your-username>/ppe_app_streamlit.git
cd ppe_app_streamlit
