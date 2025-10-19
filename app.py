from fastapi import FastAPI, File, UploadFile
from ultralytics import YOLO
import cv2
import numpy as np
from src.utils import draw_bbox
from src.measure import calculate_size
from fastapi.responses import StreamingResponse
import io

app = FastAPI()
model = YOLO("models/best.pt")  # replace with your weights

@app.post("/detect/")
async def detect(file: UploadFile = File(...)):
    image_bytes = await file.read()
    np_arr = np.frombuffer(image_bytes, np.uint8)
    frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

    results = model.predict(frame)

    for result in results:
        boxes = result.boxes.xyxy
        for box in boxes:
            w, h = calculate_size(box)
            label = f"{w}cm x {h}cm"
            frame = draw_bbox(frame, box, label)

    _, img_encoded = cv2.imencode('.jpg', frame)
    return StreamingResponse(io.BytesIO(img_encoded.tobytes()), media_type="image/jpeg")
