import cv2
import os
import pandas as pd
from ultralytics import YOLO
from utils.ocr_utils import extract_text_from_plate
from datetime import datetime

# Load models
vehicle_model = YOLO("models/yolov5s.pt")  # pretrained, detects cars
plate_model = YOLO("models/yolov5s.pt")    # optional: fine-tune for plates

# Input video
cap = cv2.VideoCapture("input/sample_video.mp4")
frame_count = 0
logs = []

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame_count += 1
    if frame_count % 10 != 0:  # Process every 10th frame to speed up
        continue

    # Detect vehicles
    vehicle_results = vehicle_model(frame)[0]
    for det in vehicle_results.boxes:
        x1, y1, x2, y2 = map(int, det.xyxy[0])
        vehicle_img = frame[y1:y2, x1:x2]

        # Detect plate in vehicle region
        plate_results = plate_model(vehicle_img)[0]
        for plate_det in plate_results.boxes:
            px1, py1, px2, py2 = map(int, plate_det.xyxy[0])
            plate_img = vehicle_img[py1:py2, px1:px2]

            # OCR
            plate_text = extract_text_from_plate(plate_img)
            logs.append({
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "plate_number": plate_text
            })

# Save logs
df = pd.DataFrame(logs)
os.makedirs("output", exist_ok=True)
df.to_csv("output/logs.csv", index=False)

print("✅ Detection complete. Logs saved in output/logs.csv")
cap.release()
