# import numpy as np
# from scipy.optimize import linear_sum_assignment
# from .kalman_filter import create_kalman_filter

# PIXEL_TO_METER = 0.05  # adjust based on real-world calibration
# FPS = 25  # default, can dynamically set from video


# # ------------------- IOU Association ------------------- #
# def iou(b1, b2):
#     xA = max(b1[0], b2[0])
#     yA = max(b1[1], b2[1])
#     xB = min(b1[2], b2[2])
#     yB = min(b1[3], b2[3])
#     inter = max(0, xB - xA) * max(0, yB - yA)
#     if inter <= 0: return 0.0
#     area1 = (b1[2]-b1[0])*(b1[3]-b1[1])
#     area2 = (b2[2]-b2[0])*(b2[3]-b2[1])
#     return inter / (area1 + area2 - inter)


# def match_tracks_to_detections(tracks, detections, iou_threshold=0.3):
#     if len(tracks) == 0:
#         return [], [], list(range(len(detections)))
    
#     iou_matrix = np.zeros((len(tracks), len(detections)))
#     for t, trk in enumerate(tracks):
#         t_box = trk.get_state_as_bbox()
#         for d, det in enumerate(detections):
#             iou_matrix[t, d] = iou(t_box, det[:4])

#     row_ind, col_ind = linear_sum_assignment(-iou_matrix)
#     matches, unmatched_tracks, unmatched_dets = [], [], []

#     for t in range(len(tracks)):
#         if t not in row_ind:
#             unmatched_tracks.append(t)
#     for d in range(len(detections)):
#         if d not in col_ind:
#             unmatched_dets.append(d)
#     for r, c in zip(row_ind, col_ind):
#         if iou_matrix[r, c] >= iou_threshold:
#             matches.append((r, c))
#         else:
#             unmatched_tracks.append(r)
#             unmatched_dets.append(c)
#     return matches, unmatched_tracks, unmatched_dets


# # ------------------- Track Class ------------------- #
# class Track:
#     def __init__(self, track_id, det):
#         x1, y1, x2, y2, conf, cls_id, cls_name = det
#         cx, cy = (x1+x2)/2, (y1+y2)/2
#         w, h = x2-x1, y2-y1

#         self.kf = create_kalman_filter(cx, cy, w, h)
#         self.track_id = track_id
#         self.cls_id = cls_id
#         self.cls_name = cls_name
#         self.conf = conf
#         self.age = 0
#         self.lost = 0
#         self.trajectory = [(cx, cy)]
#         self.last_speed = 0.0

#     def get_state_as_bbox(self):
#         x, y, w, h, _, _ = self.kf.x.flatten()
#         return [x - w/2, y - h/2, x + w/2, y + h/2]

#     def predict(self):
#         self.kf.predict()
#         self.age += 1
#         return self.get_state_as_bbox()

#     def update(self, det):
#         x1, y1, x2, y2, conf, cls_id, cls_name = det
#         cx, cy = (x1+x2)/2, (y1+y2)/2
#         w, h = x2-x1, y2-y1
#         z = np.array([cx, cy, w, h])
#         self.kf.update(z)
#         self.trajectory.append((cx, cy))
#         self.lost = 0
#         self.conf = conf
#         self.cls_id = cls_id
#         self.cls_name = cls_name

#         # Calculate speed (m/s)
#         if len(self.trajectory) >= 2:
#             dx = (self.trajectory[-1][0] - self.trajectory[-2][0]) * PIXEL_TO_METER
#             dy = (self.trajectory[-1][1] - self.trajectory[-2][1]) * PIXEL_TO_METER
#             self.last_speed = np.sqrt(dx**2 + dy**2) * FPS  # m/s
#         else:
#             self.last_speed = 0.0


# # ------------------- Track Manager ------------------- #
# class TrackManager:
#     def __init__(self, max_age=30, iou_threshold=0.3):
#         self.max_age = max_age
#         self.iou_threshold = iou_threshold
#         self.next_id = 1
#         self.tracks = []

#     def update(self, detections):
#         # Predict all tracks
#         for t in self.tracks:
#             t.predict()

#         # Match tracks to detections
#         matches, unmatched_tracks, unmatched_dets = match_tracks_to_detections(
#             self.tracks, detections, self.iou_threshold
#         )

#         # Update matched tracks
#         for trk_idx, det_idx in matches:
#             self.tracks[trk_idx].update(detections[det_idx])

#         # Mark unmatched tracks as lost
#         for idx in unmatched_tracks:
#             self.tracks[idx].lost += 1

#         # Create new tracks for unmatched detections
#         for det_idx in unmatched_dets:
#             self.tracks.append(Track(self.next_id, detections[det_idx]))
#             self.next_id += 1

#         # Remove lost tracks
#         self.tracks = [t for t in self.tracks if t.lost <= self.max_age]

#         return self.tracks

import numpy as np
import math
from .kalman_filter import create_kalman_filter
from .association import match_tracks_to_detections
from src.config import Config
cfg = Config()

class Track:
    def __init__(self, track_id, det):
        x1, y1, x2, y2, conf, cls_id, cls_name = det
        cx, cy, w, h = (x1+x2)/2, (y1+y2)/2, x2-x1, y2-y1
        self.kf = create_kalman_filter(cx, cy, w, h)
        self.track_id = track_id
        self.cls_id = cls_id
        self.cls_name = cls_name
        self.conf = conf
        self.age = 0
        self.lost = 0
        self.trajectory = [(cx, cy)]
        self.last_speed = 0.0

    def get_state_as_bbox(self):
        x, y, w, h, _, _ = self.kf.x.flatten()
        return [x-w/2, y-h/2, x+w/2, y+h/2]

    def predict(self):
        self.kf.predict()
        self.age += 1
        return self.get_state_as_bbox()

    def update(self, det):
        x1, y1, x2, y2, conf, cls_id, cls_name = det
        cx, cy, w, h = (x1+x2)/2, (y1+y2)/2, x2-x1, y2-y1
        z = np.array([cx, cy, w, h])
        self.kf.update(z)
        self.trajectory.append((cx, cy))
        self.lost = 0
        self.conf = conf
        self.cls_id = cls_id
        self.cls_name = cls_name

    def compute_speed(self, fps):
        if len(self.trajectory)<2:
            self.last_speed=0
            return
        x1,y1=self.trajectory[-2]
        x2,y2=self.trajectory[-1]
        dx,dy = x2-x1, y2-y1
        distance_pixels = math.sqrt(dx**2+dy**2)
        distance_meters = distance_pixels * cfg.PIXEL_TO_METER
        dt = 1/fps
        self.last_speed = (distance_meters/dt)*3.6  # km/h

class TrackManager:
    def __init__(self, max_age=30, iou_threshold=0.3):
        self.max_age=max_age
        self.iou_threshold=iou_threshold
        self.next_id=1
        self.tracks=[]

    def update(self, detections, fps=25):
        for t in self.tracks:
            t.predict()
        matches, unmatched_tracks, unmatched_dets = match_tracks_to_detections(self.tracks, detections, self.iou_threshold)
        for trk_idx, det_idx in matches:
            self.tracks[trk_idx].update(detections[det_idx])
        for idx in unmatched_tracks:
            self.tracks[idx].lost +=1
        for det_idx in unmatched_dets:
            self.tracks.append(Track(self.next_id, detections[det_idx]))
            self.next_id+=1
        self.tracks=[t for t in self.tracks if t.lost<=self.max_age]
        for t in self.tracks:
            t.compute_speed(fps)
        return self.tracks
