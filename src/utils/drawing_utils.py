# src/utils/drawing_utils.py
import cv2
import math
import numpy as np
from typing import Tuple, List

def angle_to_bgr(angle_deg: float) -> Tuple[int, int, int]:
    """
    Map angle (0-360) to a visually distinct BGR color via HSV hue mapping.
    """
    hue = int((angle_deg % 360) / 360.0 * 180)  # OpenCV hue range [0,179]
    hsv = np.uint8([[[hue, 200, 200]]])
    bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)[0][0]
    return int(bgr[0]), int(bgr[1]), int(bgr[2])

def compute_speed_px_per_frame(trajectory: List[Tuple[float,float]]) -> float:
    """
    Compute instantaneous speed in px/frame using last two points.
    Returns 0.0 if insufficient points.
    """
    if len(trajectory) < 2:
        return 0.0
    (x1,y1), (x2,y2) = trajectory[-2], trajectory[-1]
    return math.hypot(x2-x1, y2-y1)

def px_per_frame_to_mps(px_per_frame: float, fps: float, pixels_per_meter: float=None) -> float:
    """
    Convert px/frame to meters/second. Requires pixels_per_meter calibration.
    If no calibration provided, returns px/sec (not converted).
    """
    px_per_sec = px_per_frame * fps
    if pixels_per_meter and pixels_per_meter > 0:
        return px_per_sec / pixels_per_meter
    return px_per_sec  # fallback: px/sec

def draw_trajectory_and_direction(frame, track, max_history: int=30):
    """
    Draw a colored trajectory polyline (last N points), a directional arrow from the previous
    point to current point, and a speed label. 'track' is expected to have:
      - track.trajectory : list of (x,y) centroids
      - track.track_id, track.cls, track.last_speed (optional)
    """
    traj = track.trajectory[-max_history:]
    if len(traj) < 1:
        return frame

    # compute angle and color (if >=2 points)
    color = (0,255,0)
    angle_deg = 0.0
    if len(traj) >= 2:
        (px,py) = traj[-2]; (cx,cy) = traj[-1]
        dx, dy = cx - px, cy - py
        angle_deg = (math.degrees(math.atan2(dy, dx)) + 360) % 360
        color = angle_to_bgr(angle_deg)

    # draw polyline
    pts = np.array(traj, dtype=np.int32)
    if len(pts) >= 2:
        cv2.polylines(frame, [pts], isClosed=False, color=color, thickness=2)

    # arrow
    if len(traj) >= 2:
        start = (int(traj[-2][0]), int(traj[-2][1]))
        end = (int(traj[-1][0]), int(traj[-1][1]))
        cv2.arrowedLine(frame, start, end, color, 2, tipLength=0.3)

    # speed text (px/frame or px/sec depending on caller)
    speed_px_f = getattr(track, "last_speed", compute_speed_px_per_frame(traj))
    txt = f"ID:{track.track_id} {speed_px_f:.1f}px/f"
    # optionally append class label
    cls_label = getattr(track, "cls", None)
    if cls_label is not None:
        txt += f" | cls:{cls_label}"
    pos = (int(traj[-1][0]) + 5, int(traj[-1][1]) - 10)
    cv2.putText(frame, txt, pos, cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 2)
    return frame

def draw_bbox_and_label(frame, bbox, label: str, color: Tuple[int,int,int]=(0,255,0)):
    x1,y1,x2,y2 = bbox
    x1,y1,x2,y2 = int(x1), int(y1), int(x2), int(y2)
    cv2.rectangle(frame, (x1,y1), (x2,y2), color, 2)
    cv2.putText(frame, label, (x1, y1-8), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

def draw_tracks_on_frame(frame, tracks, fps: float=30.0):
    """
    Convenience: draw all tracks (bbox, trajectory, direction, speed).
    """
    for tr in tracks:
        bbox = tr.get_state_as_bbox()  # track class must implement this
        # compute color from last motion angle if available
        frame = draw_trajectory_and_direction(frame, tr)
        color = angle_to_bgr((getattr(tr, "last_angle", 0.0)) if hasattr(tr, "last_angle") else 0.0)
        label = f"ID:{tr.track_id}"
        if hasattr(tr, "cls"):
            label += f" cls:{tr.cls}"
        draw_bbox_and_label(frame, bbox, label, color=color)
    return frame
