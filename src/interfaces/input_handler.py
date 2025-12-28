import cv2
import os

class InputHandler:
    def load_image(self, path):
        if not path or not os.path.exists(path):
            raise FileNotFoundError(f"Image not found: {path}")
        img = cv2.imread(path)
        if img is None:
            raise ValueError(f"Failed to load image: {path}")
        return img

    def open_video(self, path_or_index):
        cap = cv2.VideoCapture(path_or_index)
        if not cap.isOpened():
            raise ValueError(f"Failed to open video or camera: {path_or_index}")
        return cap
