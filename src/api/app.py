# src/api/app.py
import os
import shutil
import uuid
from pathlib import Path
from fastapi import FastAPI, UploadFile, File, BackgroundTasks, Form
from fastapi.responses import JSONResponse
from typing import Dict
import asyncio

from src.pipeline import process_image_file, process_video_file, process_stream, sha256_file

UPLOAD_DIR = Path("uploads")
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

app = FastAPI(title="KF-Directional-Tracking API")

# In-memory job registry (simple). For production use persistent store.
jobs: Dict[str, Dict] = {}

ALLOWED_IMAGE = {"jpg","jpeg","png","bmp"}
ALLOWED_VIDEO = {"mp4","avi","mov","mkv"}

def save_upload_to_disk(upload: UploadFile, dest: Path) -> Path:
    with dest.open("wb") as buffer:
        shutil.copyfileobj(upload.file, buffer)
    return dest

def validate_extension(filename: str, allowed_set):
    ext = filename.split(".")[-1].lower()
    return ext in allowed_set

def make_job_record(job_id, kind, input_path):
    jobs[job_id] = {"status": "queued", "kind": kind, "input": str(input_path)}
    return job_id

@app.get("/status")
async def status():
    return {"jobs": jobs}

@app.post("/upload/image")
async def upload_image(background_tasks: BackgroundTasks, file: UploadFile = File(...)):
    if not validate_extension(file.filename, ALLOWED_IMAGE):
        return JSONResponse({"error":"invalid image type"}, status_code=400)

    job_id = str(uuid.uuid4())
    dest = UPLOAD_DIR / f"{job_id}_{file.filename}"
    save_upload_to_disk(file, dest)
    file_hash = sha256_file(dest)
    make_job_record(job_id, "image", dest)

    def _bg_process():
        try:
            out = process_image_file(str(dest))
            jobs[job_id].update({"status": "done", "output": out, "sha256": file_hash})
        except Exception as e:
            jobs[job_id].update({"status":"failed", "error": str(e)})

    background_tasks.add_task(_bg_process)
    return {"job_id": job_id, "sha256": file_hash}

@app.post("/upload/video")
async def upload_video(background_tasks: BackgroundTasks, file: UploadFile = File(...)):
    if not validate_extension(file.filename, ALLOWED_VIDEO):
        return JSONResponse({"error":"invalid video type"}, status_code=400)

    job_id = str(uuid.uuid4())
    dest = UPLOAD_DIR / f"{job_id}_{file.filename}"
    save_upload_to_disk(file, dest)
    file_hash = sha256_file(dest)
    make_job_record(job_id, "video", dest)

    def _bg_process():
        try:
            out = process_video_file(str(dest))
            jobs[job_id].update({"status": "done", "output": out, "sha256": file_hash})
        except Exception as e:
            jobs[job_id].update({"status":"failed", "error": str(e)})

    background_tasks.add_task(_bg_process)
    return {"job_id": job_id, "sha256": file_hash}

@app.post("/stream")
async def start_stream(background_tasks: BackgroundTasks,
                       source: str = Form(...),
                       run_for_frames: int = Form(None)):
    """
    Start processing an RTSP/HTTP stream or webcam index. 'source' can be:
      - integer index like '0' for local webcam
      - RTSP URL string
    """
    job_id = str(uuid.uuid4())
    make_job_record(job_id, "stream", source)

    def _bg_stream():
        try:
            # if source is integer index
            src = int(source) if source.isdigit() else source
            out = process_stream(src, run_for_frames=run_for_frames)
            jobs[job_id].update({"status":"done", "output": out})
        except Exception as e:
            jobs[job_id].update({"status":"failed", "error": str(e)})

    background_tasks.add_task(_bg_stream)
    return {"job_id": job_id, "status": "queued"}

@app.post("/upload/raw-path")
async def upload_raw_path(path: str = Form(...)):
    """
    Helper endpoint for systems that already uploaded file to a shared mount.
    We will validate path and sha256 and schedule processing.
    """
    p = Path(path)
    if not p.exists():
        return JSONResponse({"error":"path not found"}, status_code=404)
    # choose handler by extension
    ext = p.suffix.lower().lstrip('.')
    if ext in ALLOWED_IMAGE:
        job_id = str(uuid.uuid4()); make_job_record(job_id, "image", p)
        asyncio.create_task(async_wrap_process_image(job_id, str(p)))
        return {"job_id": job_id}
    elif ext in ALLOWED_VIDEO:
        job_id = str(uuid.uuid4()); make_job_record(job_id, "video", p)
        asyncio.create_task(async_wrap_process_video(job_id, str(p)))
        return {"job_id": job_id}
    else:
        return JSONResponse({"error":"unsupported file type"}, status_code=400)

# async wrappers to update job dict for raw path usage
async def async_wrap_process_image(job_id, path):
    try:
        out = process_image_file(path)
        jobs[job_id].update({"status":"done","output":out})
    except Exception as e:
        jobs[job_id].update({"status":"failed","error":str(e)})

async def async_wrap_process_video(job_id, path):
    try:
        out = process_video_file(path)
        jobs[job_id].update({"status":"done","output":out})
    except Exception as e:
        jobs[job_id].update({"status":"failed","error":str(e)})



# from fastapi import FastAPI, UploadFile, File, HTTPException, Body
# from pydantic import BaseModel
# from fastapi.responses import JSONResponse
# import os, cv2, numpy as np, base64, requests
# from datetime import datetime
# import shutil

# from src.config import Config
# from src.detectors.yolo_detector import YOLODetector
# from src.trackers.track_manager import TrackManager
# from src.utils.secure_logging import SecureLogger
# from src.utils.object_utils import get_category, draw_labeled_box

# cfg = Config()
# app = FastAPI(title="YOLO Tracking API")

# detector = YOLODetector(cfg.MODEL_PATH, cfg.MIN_CONF)
# tracker = TrackManager(cfg.MAX_AGE, cfg.IOU_THRESHOLD)
# logger = SecureLogger(cfg.ENCRYPTION_KEY_PATH, cfg.LOG_FILE)


# class ImageInput(BaseModel):
#     image_url: str = None
#     image_base64: str = None


# def get_color(track_id):
#     np.random.seed(track_id)
#     return tuple(int(x) for x in np.random.randint(0, 255, 3))


# def draw_annotations(frame, tracks):
#     for t in tracks:
#         box = t.get_state_as_bbox()
#         category = get_category(t.cls_name)
#         color = get_color(t.track_id)
#         label = f"ID {t.track_id} | {t.cls_name} | {category}"
#         draw_labeled_box(frame, box, label, color)
#         if len(t.trajectory) > 1:
#             for i in range(1, len(t.trajectory)):
#                 x1, y1 = t.trajectory[i - 1]
#                 x2, y2 = t.trajectory[i]
#                 cv2.line(frame, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
#     return frame


# @app.post("/detect/image", response_class=JSONResponse)
# async def detect_image(
#     input: ImageInput = Body(None),
#     file: UploadFile = File(None)
# ):
#     """
#     Accepts either:
#     1. File upload via multipart/form-data
#     2. JSON body with image_url or image_base64
#     """

#     try:
#         # 1️⃣ Handle file upload
#         if file:
#             temp_path = f"temp_{datetime.now().timestamp()}.jpg"
#             with open(temp_path, "wb") as buffer:
#                 shutil.copyfileobj(file.file, buffer)
#             img = cv2.imread(temp_path)
#             os.remove(temp_path)

#         # 2️⃣ Handle JSON input
#         elif input:
#             if input.image_url:
#                 resp = requests.get(input.image_url, stream=True)
#                 if resp.status_code != 200:
#                     raise HTTPException(status_code=400, detail="Failed to fetch image from URL")
#                 img_arr = np.asarray(bytearray(resp.content), dtype=np.uint8)
#                 img = cv2.imdecode(img_arr, cv2.IMREAD_COLOR)
#             elif input.image_base64:
#                 img_data = base64.b64decode(input.image_base64)
#                 img_arr = np.frombuffer(img_data, np.uint8)
#                 img = cv2.imdecode(img_arr, cv2.IMREAD_COLOR)
#             else:
#                 raise HTTPException(status_code=400, detail="No image provided")
#         else:
#             raise HTTPException(status_code=400, detail="No image provided")

#         if img is None:
#             raise HTTPException(status_code=400, detail="Invalid image data")

#         # Detect and track
#         detections = detector.detect(img)
#         tracks = tracker.update(detections)

#         # Log
#         logger.log({
#             "event": "image",
#             "detections": len(detections),
#             "tracks": len(tracks),
#             "timestamp": str(datetime.now())
#         })

#         # Annotate
#         annotated = draw_annotations(img, tracks)

#         # Save output
#         os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
#         out_path = os.path.join(cfg.OUTPUT_DIR, f"processed_{datetime.now().timestamp()}.jpg")
#         cv2.imwrite(out_path, annotated)

#         return {
#             "message": "Image processed successfully",
#             "detections": len(detections),
#             "tracks": len(tracks),
#             "output_path": out_path
#         }

#     except Exception as e:
#         raise HTTPException(status_code=500, detail=str(e))
