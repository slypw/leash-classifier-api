# Leash Classifier API

A serverless computer vision microservice that detects dogs in images using **YOLOv8-Nano** and classifies whether they are **On-Leash** or **Off-Leash** using a fine-tuned **ConvNeXt-Tiny** model.

Designed using edge cases like trail-camera photos supplemented with Stanford Actions and Stanford Dogs datasets, this API automatically crops, squares, and expands bounding boxes for classification accuracy.

---

## Architecture

- **Framework:** FastAPI
- **Serverless Infrastructure:** Modal (Containerized runtime with cloud volume storage for model weights)
- **Object Detection:** Ultralytics YOLOv8-Nano (COCO class `16` for dogs)
- **Image Classification:** TIMM ConvNeXt-Tiny (`convnext_tiny.fb_in1k`)
- **Image Processing:** Pillow (PIL), NumPy

---

## Computer Vision Pipeline

When an image is sent to the API, it undergoes the following processing steps:

1. **Detection:** YOLOv8 scans the image and extracts bounding box coordinates for all detected dogs.
2. **Squaring & Expansion:** To prevent distortion, each bounding box is transformed into a square based on its maximum dimension, then scaled up by **40%** to capture context (collars, leashes, and handlers).
3. **Boundary Correction:** Shift logic ensures crops that fall outside image boundaries are safely shifted and clamped within valid pixel ranges.
4. **Classification:** Each processed dog crop is passed through the fine-tuned ConvNeXt-Tiny model to output a probabilistic prediction (`on_leash` vs. `off_leash`).
5. **Visualization:** Color-coded bounding boxes (**Green** for on-leash, **Red** for off-leash) and confidence scores are drawn directly onto the image and returned as an HTML response.

---

## Research & Methodology

This API implements the computer vision pipeline, bounding box squaring, and 40% expansion methodology detailed in my academic research with a re-finetuned model on Stanford datasets as well as trail camera images for generalizability:

📄 **[Read the Full Research Paper (PDF)](https://website-e86815.gitlab.io/wildland-trail-camera-paper.pdf)**

---

## Project Structure

```text
leash-classifier-api/
│
├── deploy_api.py      # Main FastAPI app and Modal serverless configuration
├── requirements.txt   # Python dependencies
├── .gitignore         # Ignores local weights and cache files
└── README.md          # Project documentation
```
