# from ultralytics import YOLO

# class YOLODetector:
#     def __init__(self, model_path="yolov8n.pt", conf=0.35):
#         print("[INFO] Loading YOLO model:", model_path)
#         self.model = YOLO(model_path)
#         self.conf = conf
#         self.names = self.model.names  # class names

#     def detect(self, frame):
#         """
#         Returns unified detection format:
#         [x1, y1, x2, y2, conf, cls_id, cls_name]
#         """
#         results = self.model.predict(frame, conf=self.conf, verbose=False)
#         detections = []

#         for box in results[0].boxes:
#             x1, y1, x2, y2 = box.xyxy[0]
#             conf = float(box.conf[0])
#             cls_id = int(box.cls[0])
#             cls_name = self.names[cls_id]

#             detections.append([
#                 float(x1), float(y1),
#                 float(x2), float(y2),
#                 conf,
#                 cls_id,
#                 cls_name
#             ])

#         return detections
# from ultralytics import YOLO

# class YOLODetector:
#     def __init__(self, model_path="yolov8n.pt", conf=0.35):
#         print("[INFO] Loading YOLO model:", model_path)
#         self.model = YOLO(model_path)
#         self.conf = conf
#         self.names = self.model.names
#         print("[INFO] Model loaded successfully.")

#     def detect(self, frame):
#         """
#         Returns detections in format:
#         [x1, y1, x2, y2, conf, cls_id, cls_name]
#         """
#         results = self.model.predict(frame, conf=self.conf, verbose=False)

#         if len(results) == 0 or len(results[0].boxes) == 0:
#             return []

#         boxes = results[0].boxes
#         detections = []

#         for box in boxes:
#             x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
#             conf = float(box.conf[0].cpu())
#             cls_id = int(box.cls[0].cpu())
#             cls_name = self.names[cls_id]

#             detections.append([x1, y1, x2, y2, conf, cls_id, cls_name])

#         return detections
# src/detectors/yolo_detector.py

from ultralytics import YOLO
import torch
import cv2

class YOLODetector:
    def __init__(self, model_path="models/yolov8n.pt", conf=0.35, device=None):
        print("[INFO] Loading YOLO model:", model_path)
        
        # Detect device
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device
        
        # Load model without device argument
        self.model = YOLO(model_path)
        self.conf = conf
        self.names = self.model.names
        print(f"[INFO] Model loaded successfully on {self.device}.")

    def detect(self, frame, resize_frame=True):
        if frame is None:
            return []

        original_h, original_w = frame.shape[:2]

        if resize_frame and self.device == "cpu":
            frame_small = cv2.resize(frame, (640, 360))
        else:
            frame_small = frame

        # Pass device in predict()
        results = list(self.model.predict(
            frame_small,
            conf=self.conf,
            verbose=False,
            stream=True,
            device=self.device  # fix here
        ))

        if len(results) == 0 or len(results[0].boxes) == 0:
            return []

        detections = []
        for box in results[0].boxes:
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
            conf = float(box.conf[0].cpu())
            cls_id = int(box.cls[0].cpu())
            cls_name = self.names[cls_id]

            if resize_frame and self.device == "cpu":
                x1 *= original_w / 640
                x2 *= original_w / 640
                y1 *= original_h / 360
                y2 *= original_h / 360

            detections.append([x1, y1, x2, y2, conf, cls_id, cls_name])

        return detections
