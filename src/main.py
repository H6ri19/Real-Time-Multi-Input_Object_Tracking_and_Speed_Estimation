import argparse, os, cv2, time
from datetime import datetime
from src.config import Config
from src.interfaces.input_handler import InputHandler
from src.detectors.yolo_detector import YOLODetector
from src.trackers.track_manager import TrackManager
from src.utils.secure_logging import SecureLogger
from src.utils.object_utils import get_category, draw_labeled_box

cfg = Config()

def get_color(track_id):
    import numpy as np
    np.random.seed(track_id)
    return tuple(int(x) for x in np.random.randint(0, 255, 3))

def draw_annotations(frame, tracks):
    for t in tracks:
        box = t.get_state_as_bbox()
        category = get_category(t.cls_name)
        color = get_color(t.track_id)
        label = f"ID {t.track_id} | {t.cls_name} | {category} | {t.last_speed:.1f} km/h"
        draw_labeled_box(frame, box, label, color)
        # Draw trajectory
        if len(t.trajectory) > 1:
            for i in range(1, len(t.trajectory)):
                x1, y1 = t.trajectory[i-1]
                x2, y2 = t.trajectory[i]
                cv2.line(frame, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
    return frame

def process_image(img, detector, tracker, logger):
    detections = detector.detect(img)
    tracks = tracker.update(detections, fps=25)

    logger.log({
        "event": "image",
        "detections": len(detections),
        "tracks": len(tracks),
        "timestamp": str(datetime.now())
    })

    annotated = draw_annotations(img, tracks)

    # Unique filename
    filename = f"processed_image_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg"
    out = os.path.join(cfg.OUTPUT_DIR, filename)
    os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
    cv2.imwrite(out, annotated)
    print("[SAVED] →", out)

def process_video(cap, detector, tracker, logger, time_limit=None, show_window=True):
    os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)

    # Unique filename
    filename = f"processed_video_{datetime.now().strftime('%Y%m%d_%H%M%S')}.mp4"
    out_path = os.path.join(cfg.OUTPUT_DIR, filename)

    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 25

    writer = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
    frame_count = 0
    start_time = time.time()

    while True:
        ret, frame = cap.read()
        if not ret or frame is None:
            break

        # Stop if time limit reached (webcam/RTSP)
        if time_limit and (time.time() - start_time) > time_limit:
            print(f"[INFO] Time limit {time_limit}s reached.")
            break

        detections = detector.detect(frame)
        tracks = tracker.update(detections, fps=fps)

        logger.log({
            "frame": frame_count,
            "detections": len(detections),
            "tracks": len(tracks),
            "timestamp": str(datetime.now())
        })

        annotated = draw_annotations(frame, tracks)
        writer.write(annotated)

        if show_window:
            cv2.imshow("Tracking", annotated)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        frame_count += 1

    writer.release()
    cap.release()
    if show_window:
        cv2.destroyAllWindows()
    print("[SAVED] →", out_path)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["image", "video", "webcam", "rtsp"], required=True)
    parser.add_argument("--path", default=None)
    parser.add_argument("--webcam_time", type=int, default=15)
    parser.add_argument("--show_window", action="store_true", help="Display window while processing video/webcam")
    args = parser.parse_args()

    ih = InputHandler()
    detector = YOLODetector(cfg.MODEL_PATH, cfg.MIN_CONF)
    tracker = TrackManager(cfg.MAX_AGE, cfg.IOU_THRESHOLD)
    logger = SecureLogger(cfg.ENCRYPTION_KEY_PATH, cfg.LOG_FILE)

    if args.mode == "image":
        if args.path is None:
            raise ValueError("Please provide --path for image mode")
        img = ih.load_image(args.path)
        process_image(img, detector, tracker, logger)

    elif args.mode == "video":
        if args.path is None:
            raise ValueError("Please provide --path for video mode")
        cap = ih.open_video(args.path)
        process_video(cap, detector, tracker, logger, show_window=args.show_window)

    elif args.mode == "webcam":
        cap = ih.open_video(0)
        process_video(cap, detector, tracker, logger, time_limit=args.webcam_time, show_window=args.show_window)

    elif args.mode == "rtsp":
        if args.path is None:
            raise ValueError("Please provide --path for RTSP stream")
        cap = ih.open_video(args.path)
        process_video(cap, detector, tracker, logger, show_window=args.show_window)

if __name__ == "__main__":
    main()
