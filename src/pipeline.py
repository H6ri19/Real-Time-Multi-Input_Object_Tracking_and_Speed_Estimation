# src/pipeline.py
import os
import cv2
import hashlib
from datetime import datetime
from pathlib import Path

from src.config import Config
from src.interfaces.input_handler import InputHandler
from src.detectors.yolo_detector import YOLODetector
from src.trackers.track_manager import TrackManager
from src.utils.secure_logging import SecureLogger
from src.utils.drawing_utils import draw_tracks_on_frame, compute_speed_px_per_frame

cfg = Config()

# create dirs
Path(cfg.OUTPUT_DIR).mkdir(parents=True, exist_ok=True)
Path("uploads").mkdir(parents=True, exist_ok=True)

# initialize shared singletons (one model load)
_detector = None
_tracker = None
_logger = None
_input_handler = InputHandler()

def init_components():
    global _detector, _tracker, _logger
    if _detector is None:
        _detector = YOLODetector(cfg.MODEL_PATH, cfg.MIN_CONF)
    if _tracker is None:
        _tracker = TrackManager(max_age=cfg.MAX_AGE, iou_threshold=cfg.IOU_THRESHOLD)
    if _logger is None:
        _logger = SecureLogger(cfg.ENCRYPTION_KEY_PATH, cfg.LOG_FILE)

def sha256_file(path):
    h = hashlib.sha256()
    with open(path,'rb') as f:
        for chunk in iter(lambda: f.read(8192), b''):
            h.update(chunk)
    return h.hexdigest()

def process_image_file(input_path: str, output_path: str=None):
    """
    Process a single image file (detect -> single-frame update -> draw and save).
    """
    init_components()
    img = _input_handler.load_image(input_path)
    dets = _detector.detect(img)
    tracks = _tracker.update(dets)

    # annotate
    annotated = draw_tracks_on_frame(img, tracks, fps=1.0)

    # save
    out = output_path or os.path.join(cfg.OUTPUT_DIR, f"image_{Path(input_path).stem}_proc.jpg")
    cv2.imwrite(out, annotated)

    # log (encrypted)
    _logger.log({
        "event": "image_processed",
        "input": input_path,
        "output": out,
        "detections": len(dets),
        "tracks": [ {"id": t.track_id, "cls": getattr(t,"cls",None)} for t in tracks],
        "sha256": sha256_file(input_path),
        "timestamp": str(datetime.utcnow())
    })
    return out

def process_video_file(input_path: str, output_path: str=None, max_frames: int=None):
    """
    Process a video file: read frames, do detect->track, annotate, write video.
    """
    init_components()
    cap = _input_handler.open_video(input_path)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 25

    out = output_path or os.path.join(cfg.OUTPUT_DIR, f"video_{Path(input_path).stem}_proc.mp4")
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(out, fourcc, fps, (w,h))

    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if max_frames and frame_idx >= max_frames:
            break

        dets = _detector.detect(frame)
        tracks = _tracker.update(dets)

        # compute last_speed for each track (px/frame)
        for t in tracks:
            t.last_speed = compute_speed_px_per_frame(t.trajectory)

        annotated = draw_tracks_on_frame(frame, tracks, fps=fps)
        writer.write(annotated)

        # log per-frame summary (encrypted)
        _logger.log({
            "event": "video_frame",
            "input": input_path,
            "frame": frame_idx,
            "detections": len(dets),
            "active_tracks": len(tracks),
            "timestamp": str(datetime.utcnow())
        })

        frame_idx += 1

    writer.release()
    cap.release()

    _logger.log({
        "event": "video_done",
        "input": input_path,
        "output": out,
        "frames": frame_idx,
        "sha256": sha256_file(input_path),
        "timestamp": str(datetime.utcnow())
    })
    return out

def process_stream(source, output_path: str=None, run_for_frames: int=None):
    """
    Process a live stream (webcam index or RTSP URL). Blocks until stream ends or run_for_frames reached.
    """
    init_components()
    cap = _input_handler.open_video(source)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 25

    out = output_path or os.path.join(cfg.OUTPUT_DIR, f"stream_{datetime.utcnow().strftime('%Y%m%d%H%M%S')}.mp4")
    writer = cv2.VideoWriter(out, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w,h))

    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if run_for_frames and frame_idx >= run_for_frames:
            break

        dets = _detector.detect(frame)
        tracks = _tracker.update(dets)
        for t in tracks:
            t.last_speed = compute_speed_px_per_frame(t.trajectory)

        annotated = draw_tracks_on_frame(frame, tracks, fps=fps)
        writer.write(annotated)

        _logger.log({
            "event": "stream_frame",
            "frame": frame_idx,
            "active_tracks": len(tracks),
            "timestamp": str(datetime.utcnow())
        })
        frame_idx += 1

    writer.release()
    cap.release()
    _logger.log({
        "event": "stream_done",
        "output": out,
        "frames": frame_idx,
        "timestamp": str(datetime.utcnow())
    })
    return out
