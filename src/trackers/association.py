import numpy as np
from scipy.optimize import linear_sum_assignment

def iou(b1, b2):
    xA = max(b1[0], b2[0])
    yA = max(b1[1], b2[1])
    xB = min(b1[2], b2[2])
    yB = min(b1[3], b2[3])

    inter = max(0, xB - xA) * max(0, yB - yA)
    if inter <= 0:
        return 0.0
    area1 = (b1[2]-b1[0])*(b1[3]-b1[1])
    area2 = (b2[2]-b2[0])*(b2[3]-b2[1])
    return inter / (area1 + area2 - inter)

def match_tracks_to_detections(tracks, detections, iou_threshold=0.3):
    if len(tracks)==0:
        return [], [], list(range(len(detections)))
    iou_matrix = np.zeros((len(tracks), len(detections)))
    for t, trk in enumerate(tracks):
        t_box = trk.get_state_as_bbox()
        for d, det in enumerate(detections):
            iou_matrix[t,d] = iou(t_box, det[:4])

    row_ind, col_ind = linear_sum_assignment(-iou_matrix)
    matches, unmatched_tracks, unmatched_dets = [], [], []

    for t in range(len(tracks)):
        if t not in row_ind: unmatched_tracks.append(t)
    for d in range(len(detections)):
        if d not in col_ind: unmatched_dets.append(d)

    for r,c in zip(row_ind, col_ind):
        if iou_matrix[r,c]>=iou_threshold:
            matches.append((r,c))
        else:
            unmatched_tracks.append(r)
            unmatched_dets.append(c)
    return matches, unmatched_tracks, unmatched_dets
