# APEX — Adaptive Pose & Action Intelligence System

> Real-time multi-modal AI system combining object detection, pose estimation, skeleton analysis, action recognition, and persistent memory — in a single Python file.

---

## Features

- **Object Detection** — YOLOv8n detects and labels objects in real-time
- **Pose Estimation** — YOLOv8n-pose tracks 17 COCO keypoints per person
- **Skeleton Analysis** — computes 8 joint angles + posture score (0–100)
- **Action Recognition** — rule-based engine + LSTM on 30-frame sequences
- **3-Layer Memory System**
  - Short-term RAM buffer (last 60 frames)
  - Pattern learning — discovers and remembers action sequences
  - Long-term SQLite storage — sessions, performance history, cross-session comparison

---

## Demo

```
┌─────────────────────────────────────────────────────────┐
│  ── APEX SYSTEM ──                                      │
│  FPS:     28                                            │
│  ACTION:  SQUATTING                                     │
│  CONF:    85%                                           │
│  POSTURE: 76/100          ████████████░░░  POSTURE      │
│                                                         │
│  ── JOINT ANGLES ──                                     │
│  left_elbow:    142°                                    │
│  right_elbow:   138°                                    │
│  left_knee:     112°                                    │
│  right_knee:    108°                                    │
│                                                         │
│  ── MEMORY ──                                           │
│  Sessions: 4                                            │
│  Patterns: 7                                            │
│  vs avg: +3.2 (above avg)                               │
└─────────────────────────────────────────────────────────┘
```

---

## Installation

```bash
pip install ultralytics opencv-python torch numpy
```

---

## Usage

```bash
# Webcam
python apex.py --source 0

# Video file
python apex.py --source path/to/video.mp4
```

Or set source directly inside the file:

```python
if __name__ == "__main__":
    source = 0              # webcam
    # source = "video.mp4"  # video file
    run(source=source, lstm_path=None)
```

**Controls:**

| Key | Action |
|-----|--------|
| `Q` | Quit + auto-save session |
| `S` | Save session manually |

---

## Architecture

```
apex.py
│
├── YOLOv8n-pose          # object detection + 17 keypoints
│
├── Skeleton Analyzer
│   ├── extract_angles()  # 8 joint angles from keypoint triplets
│   └── posture_score()   # symmetry + joint range scoring
│
├── Action Recognizer
│   ├── Rule-based        # instant, no training needed
│   └── LSTM              # temporal sequence model (optional)
│
└── APEX Memory
    ├── Short-term         # RAM deque — last 60 frames
    ├── Pattern Memory     # learns repeated action sequences
    └── Long-term (SQLite) # sessions + performance comparison
```

---

## Detected Actions

| Action | Method |
|--------|--------|
| Standing | LSTM / default |
| Squatting | Rule: knee angle < 120° |
| Raising Arms | Rule: shoulder angle < 60° |
| Sitting | Rule: hip angle 80°–130° |
| Walking | LSTM |
| Unknown | fallback |

---

## Memory System

Every session is saved to `apex_memory.db` (SQLite):

```
sessions    → timestamp, duration, actions, avg angles, posture score
patterns    → learned action sequences + occurrence count
performance → per-metric history for cross-session comparison
```

---

## Tech Stack

| Tool | Role |
|------|------|
| YOLOv8 (Ultralytics) | Detection + Pose |
| OpenCV | Video I/O + rendering |
| PyTorch | LSTM action model |
| SQLite | Persistent memory |
| NumPy | Angle computation |

---

## Project Structure

```
APEX/
├── apex.py           # entire system — single file
├── apex_memory.db    # auto-generated on first run
└── yolov8n-pose.pt   # auto-downloaded on first run
```

---

## Optional: Train Your Own LSTM

To use a custom-trained LSTM model:

```bash
python apex.py --source 0 --lstm path/to/model.pt
```

The model expects input shape `(batch, 30, 8)` — 30-frame sequences of 8 joint angles.

---

## CV Description

> **APEX — Adaptive Pose & Action Intelligence System** | Python · YOLOv8 · PyTorch · OpenCV · SQLite  
> Built a real-time multi-modal CV system integrating YOLOv8 object detection, 17-keypoint skeleton analysis, rule-based + LSTM action recognition on temporal sequences, and a 3-layer memory architecture enabling cross-session performance comparison and unsupervised action pattern discovery.

---

## Author

**Youssef** — CS & AI Student, Cairo University  
AI Engineer Intern @ Novus
