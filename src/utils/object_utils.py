import cv2

def draw_labeled_box(frame, bbox, label, color=(0,255,0), thickness=2):
    x1, y1, x2, y2 = map(int, bbox)
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)
    t_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)[0]
    cv2.rectangle(frame, (x1, y1 - t_size[1] - 4), (x1 + t_size[0], y1), color, -1)
    cv2.putText(frame, label, (x1, y1-2), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1)

def get_category(cls_name):
    """Return category based on class (customizable)"""
    vehicle_classes = ["car", "truck", "bus", "motorbike"]
    person_classes = ["person"]
    if cls_name.lower() in vehicle_classes:
        return "Vehicle"
    elif cls_name.lower() in person_classes:
        return "Person"
    else:
        return "Other"
