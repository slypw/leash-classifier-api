import modal
import io
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image, ImageDraw
import base64

# cloud container image with system libraries for OpenCV and dependencies
image = (
    modal.Image.debian_slim(python_version="3.10")
    .apt_install("libgl1", "libglib2.0-0")
    .pip_install(
        "torch",
        "torchvision",
        "timm",
        "ultralytics",
        "pillow",
        "fastapi[standard]",
        "numpy",
        "scikit-learn",
        "opencv-python-headless",
    )
)

app = modal.App(name="leash-classifier-api")

volume = modal.Volume.from_name("model-weights-vol", create_if_missing=True)
MODEL_DIR = "/weights"


@app.cls(image=image, volumes={MODEL_DIR: volume}, cpu=2.0, scaledown_window=60)
class LeashDetectorService:
    @modal.enter()
    def load_model(self):
        """Loads YOLO and ConvNeXt models when server container starts"""
        import torch
        import timm
        from ultralytics import YOLO

        print("Loading YOLOv8-nano...")
        self.yolo_model = YOLO("yolov8n.pt")

        print("Loading ConvNeXt-Tiny model...")
        weights_path = f"{MODEL_DIR}/model_best.pth.tar"
        self.model = timm.create_model(
            model_name="convnext_tiny.fb_in1k",
            pretrained=False,
            checkpoint_path=weights_path,
            num_classes=2,
        )
        self.model.to("cpu")
        self.model.eval()

        config = timm.data.resolve_data_config({}, model=self.model)
        self.transform = timm.data.create_transform(**config, is_training=False)
        self.labels = ["off_leash", "on_leash"]

        print("Models successfully loaded.")

    def get_dog_crops(self, image: Image.Image):
        """Image preprocessing (from paper) for model"""
        # dogs = coco idx 16
        results = self.yolo_model(image, classes=[16], verbose=False)

        if not results or len(results[0].boxes) == 0:
            return None

        # get all dog bboxes
        boxes = results[0].boxes.xyxy.cpu().numpy()
        dog_data = []

        for box in boxes:
            x1, y1, x2, y2 = box

            # calculate coors
            w = x2 - x1
            h = y2 - y1
            cx = x1 + (w / 2.0)
            cy = y1 + (h / 2.0)

            s = max(w, h)

            s_scaled = s * 1.40
            half_s = s_scaled / 2.0

            new_x1, new_y1 = cx - half_s, cy - half_s
            new_x2, new_y2 = cx + half_s, cy + half_s

            # shifts
            if new_x1 < 0:
                new_x2 += abs(new_x1)
                new_x1 = 0
            if new_y1 < 0:
                new_y2 += abs(new_y1)
                new_y1 = 0
            if new_x2 > image.width:
                new_x1 -= new_x2 - image.width
                new_x2 = image.width
            if new_y2 > image.height:
                new_y1 -= new_y2 - image.height
                new_y2 = image.height

            # in case + 40% is larger than photo
            new_x1, new_y1 = max(0, new_x1), max(0, new_y1)
            new_x2, new_y2 = min(image.width, new_x2), min(image.height, new_y2)

            cropped = image.crop((new_x1, new_y1, new_x2, new_y2))
            dog_data.append((cropped, (x1, y1, x2, y2)))

        return dog_data

    @modal.asgi_app()
    def fastapi_app(self):
        f_app = FastAPI()

        f_app.add_middleware(
            CORSMiddleware,
            allow_origins=["*"],
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )

        @f_app.post("/predict")
        async def predict(file: UploadFile = File(...)) -> HTMLResponse:
            """Given an uploaded returns the predicted labels and bboxes for all detected dogs"""
            import torch

            if not file.content_type.startswith("image/"):
                return HTMLResponse(
                    "<div style='color: red;'>Error: Provided file is not an image.</div>"
                )

            try:
                img_bytes = await file.read()
                img = Image.open(io.BytesIO(img_bytes)).convert("RGB")

                crops_and_boxes = self.get_dog_crops(img)

                if not crops_and_boxes:
                    return HTMLResponse(
                        "<div style='color: #d97706; font-weight: bold;'>No dogs found in this image.</div>"
                    )

                html_resp = "<div>"
                draw = ImageDraw.Draw(img)

                for cropped_img, box_coors in crops_and_boxes:
                    tensor = self.transform(cropped_img).unsqueeze(0)

                    with torch.no_grad():
                        probs = torch.softmax(self.model(tensor), dim=1)[0]

                    idx = probs.argmax().item()
                    pred_class = self.labels[idx]
                    pred_conf = probs[idx].item() * 100

                    color = "green" if pred_class == "on_leash" else "red"
                    display_label = pred_class.replace("_", "-").upper()

                    html_resp += f"""
                        <p>Prediction: <span style="color: {color}; font-weight: bold;">
                        {display_label}</span> ({pred_conf:.1f}%)</p>
                    """

                    # draw bboxes for each dog
                    x1, y1, x2, y2 = box_coors
                    draw.rectangle([x1, y1, x2, y2], outline=color, width=4)

                html_resp += "</div>"

                buff = io.BytesIO()
                img.save(buff, format="JPEG")
                img_str = base64.b64encode(buff.getvalue()).decode("utf-8")

                html_resp += f"""
                    <br>
                    <img src="data:image/jpeg;base64,{img_str}" style="max-width: 100%; height: auto; border-radius: 8px;" alt="Detected Dogs">
                    <script>document.getElementById('preview').style.display = 'none';</script>
                """

                return HTMLResponse(content=html_resp)

            except Exception as e:
                return HTMLResponse(
                    f"<div style='color: red;'>Error processing image: {str(e)}</div>"
                )

        return f_app
