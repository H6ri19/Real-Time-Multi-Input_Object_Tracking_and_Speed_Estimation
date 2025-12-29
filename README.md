# Real-Time Multi-Input Object Tracking and Speed Estimation

A real-time multi-object tracking and speed estimation system combining **YOLOv8 object detection** with **Kalman Filter-based tracking** to handle multiple video inputs, maintain identity across frames, and estimate object speeds accurately. Designed for robust performance in environments with occlusions, noisy detections, and varying motion. :contentReference[oaicite:0]{index=0}

---

## 📌 Features

- 🔍 **Real-Time Detection:** Uses YOLOv8 (You Only Look Once) for fast and accurate object detection.
- 🧠 **Tracking:** Applies a Kalman Filter-based tracker to associate detections across frames.
- 🎯 **Speed Estimation:** Computes object speed from motion data over time.
- 📹 **Multi-Input Support:** Accepts and tracks objects from multiple video sources.
- 🚦 **Occlusion Handling:** Maintains identity even in partially obscured views.

---

## 📁 Project Structure

Real-Time-Multi-Input_Object_Tracking_and_Speed_Estimation/
├── data/ # Dataset files for training/testing
├── docs/ # Documentation
├── experiments/ # Experimental results/logs
├── models/ # Pretrained models or checkpoints
├── src/ # Core source code
├── requirements.txt # Python dependencies
└── README.md # This documentation

▶️ Running the System
👇 Basic Usage
python src/main.py --input_video path/to/video.mp4

🖥 Multi-Input Example
python src/main.py \
  --input_video1 camera1.mp4 \
  --input_video2 camera2.mp4

🛠 Options
--input_video         Path to video file
--model_weights       Custom YOLOv8 weights
--output_dir          Save results and logs
--display             Show live output window


Adjust flags depending on how you structured your CLI parsing.

Research Paper Publication 
<img width="612" height="433" alt="image" src="https://github.com/user-attachments/assets/98adfdeb-e77a-489f-963a-1446ff7c7764" />
