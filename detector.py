"""
APEX v3 — Adaptive Pose & Action Intelligence System
=====================================================
Level 1 : YOLOv8-Pose        → 17 body keypoints + joint angles + posture + action
Level 2 : pose-based          → hand state + arm extension + direction
Level 3 : YuNet + LBF        → 68 face landmarks + EAR/MAR + expressions + head pose

Zero MediaPipe. Pure YOLOv8 + OpenCV DNN.
Face runs in a dedicated thread → ~20-28 FPS instead of 3 FPS.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
INSTALL:
    pip install ultralytics opencv-python torch numpy

MODELS  (download once, put next to this file):
  Face detector  → face_detection_yunet_2023mar.onnx  (~400 KB)
    https://github.com/opencv/opencv_zoo/raw/main/models/face_detection_yunet/face_detection_yunet_2023mar.onnx

  Face landmarks → lbfmodel.yaml  (~54 MB)
    https://github.com/kurnianggoro/GSOC2017/raw/master/data/lbfmodel.yaml

  Pose model     → yolov8n-pose.pt  (auto-downloaded by ultralytics)

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
RUN:
    python apex_v2.py
"""

import cv2
import numpy as np
import sqlite3
import json
import time
import os
import threading
import torch
import torch.nn as nn
from datetime import datetime
from collections import deque, Counter
from ultralytics import YOLO

# ══════════════════════════════════════════════════════════════
#  CONFIG
# ══════════════════════════════════════════════════════════════

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
YUNET_PATH = os.path.join(SCRIPT_DIR, "face_detection_yunet_2023mar.onnx")
LBF_PATH   = os.path.join(SCRIPT_DIR, "lbfmodel.yaml")

# ══════════════════════════════════════════════════════════════
#  CONSTANTS
# ══════════════════════════════════════════════════════════════

KEYPOINT_NAMES = [
    'nose', 'left_eye', 'right_eye', 'left_ear', 'right_ear',
    'left_shoulder', 'right_shoulder', 'left_elbow', 'right_elbow',
    'left_wrist', 'right_wrist', 'left_hip', 'right_hip',
    'left_knee', 'right_knee', 'left_ankle', 'right_ankle'
]
KP = {name: i for i, name in enumerate(KEYPOINT_NAMES)}

SKELETON_CONNECTIONS = [
    (0,1),(0,2),(1,3),(2,4),
    (5,6),(5,7),(7,9),(6,8),(8,10),
    (5,11),(6,12),(11,12),
    (11,13),(13,15),(12,14),(14,16)
]

JOINT_TRIPLETS = {
    'left_elbow':     ('left_shoulder',  'left_elbow',    'left_wrist'),
    'right_elbow':    ('right_shoulder', 'right_elbow',   'right_wrist'),
    'left_shoulder':  ('left_elbow',     'left_shoulder', 'left_hip'),
    'right_shoulder': ('right_elbow',    'right_shoulder','right_hip'),
    'left_knee':      ('left_hip',       'left_knee',     'left_ankle'),
    'right_knee':     ('right_hip',      'right_knee',    'right_ankle'),
    'left_hip':       ('left_shoulder',  'left_hip',      'left_knee'),
    'right_hip':      ('right_shoulder', 'right_hip',     'right_knee'),
}

ACTIONS = ['standing', 'walking', 'sitting', 'raising_arms', 'squatting', 'unknown']

# LBF 68-point groups
LBF_JAW         = list(range(0,  17))
LBF_LBROW       = list(range(17, 22))
LBF_RBROW       = list(range(22, 27))
LBF_NOSE        = list(range(27, 36))
LBF_LEYE        = list(range(36, 42))
LBF_REYE        = list(range(42, 48))
LBF_OUTER_MOUTH = list(range(48, 60))
LBF_INNER_MOUTH = list(range(60, 68))

COLORS = {
    'skeleton': (0, 255, 120),
    'joint':    (0, 180, 255),
    'face':     (180, 80, 255),
    'face_dim': (110, 50, 160),
    'hand_l':   (0, 220, 255),
    'hand_r':   (255, 180, 0),
}


# ══════════════════════════════════════════════════════════════
#  LEVEL 1 — BODY / POSE
# ══════════════════════════════════════════════════════════════

def calc_angle(a, b, c):
    a, b, c = np.array(a), np.array(b), np.array(c)
    v1 = a - b
    v2 = c - b
    cos_a = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-6)
    return float(np.degrees(np.arccos(np.clip(cos_a, -1, 1))))


def extract_angles(kps):
    angles = {}
    for name, (a, b, c) in JOINT_TRIPLETS.items():
        ia, ib, ic = KP[a], KP[b], KP[c]
        if kps[ia][2] > 0.4 and kps[ib][2] > 0.4 and kps[ic][2] > 0.4:
            angles[name] = calc_angle(kps[ia][:2], kps[ib][:2], kps[ic][:2])
    return angles


def posture_score(angles):
    if not angles:
        return 0.0
    score = 100.0
    for l, r in [('left_elbow','right_elbow'),
                 ('left_knee', 'right_knee'),
                 ('left_hip',  'right_hip')]:
        if l in angles and r in angles:
            score -= min(abs(angles[l] - angles[r]) * 0.3, 15)
    for side in ['left_knee', 'right_knee']:
        if side in angles and angles[side] < 80:
            score -= (80 - angles[side]) * 0.2
    return max(0.0, min(100.0, score))


ACTION_RULES = {
    'squatting':    lambda a: a.get('left_knee',  180) < 120 and a.get('right_knee', 180) < 120,
    'raising_arms': lambda a: a.get('left_shoulder', 180) < 60 or  a.get('right_shoulder', 180) < 60,
    'sitting':      lambda a: 80 < a.get('left_hip', 180) < 130,
}


class ActionLSTM(nn.Module):
    def __init__(self, input_size=8, hidden_size=128, num_layers=2, num_classes=6):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers,
                            batch_first=True, dropout=0.3)
        self.norm = nn.LayerNorm(hidden_size)
        self.fc   = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        out, _ = self.lstm(x)
        return self.fc(self.norm(out[:, -1, :]))


class ActionRecognizer:
    SEQ_LEN    = 30
    ANGLE_KEYS = ['left_elbow','right_elbow','left_shoulder','right_shoulder',
                  'left_knee', 'right_knee', 'left_hip',    'right_hip']

    def __init__(self, model_path=None):
        self.sequence = deque(maxlen=self.SEQ_LEN)
        self.lstm     = None
        if model_path:
            try:
                self.lstm = ActionLSTM(num_classes=len(ACTIONS))
                self.lstm.load_state_dict(torch.load(model_path, map_location='cpu'))
                self.lstm.eval()
                print(f"[APEX] LSTM loaded: {model_path}")
            except Exception as e:
                print(f"[APEX] LSTM failed: {e}")

    def predict(self, angles):
        self.sequence.append([angles.get(k, 180.0) for k in self.ANGLE_KEYS])
        for action, rule in ACTION_RULES.items():
            if rule(angles):
                return action, 0.85
        if self.lstm and len(self.sequence) == self.SEQ_LEN:
            x = torch.tensor([list(self.sequence)], dtype=torch.float32)
            with torch.no_grad():
                probs = torch.softmax(self.lstm(x), dim=1)[0]
                idx   = int(probs.argmax())
                return ACTIONS[idx], float(probs[idx])
        return 'standing', 0.60


def draw_skeleton(frame, kps):
    for (a, b) in SKELETON_CONNECTIONS:
        if kps[a][2] > 0.4 and kps[b][2] > 0.4:
            cv2.line(frame,
                     (int(kps[a][0]), int(kps[a][1])),
                     (int(kps[b][0]), int(kps[b][1])),
                     COLORS['skeleton'], 2)
    for x, y, c in kps:
        if c > 0.4:
            cv2.circle(frame, (int(x), int(y)), 4, COLORS['joint'], -1)


# ══════════════════════════════════════════════════════════════
#  LEVEL 2 — HAND ANALYSIS (from pose wrist keypoints)
# ══════════════════════════════════════════════════════════════

class HandAnalyzer:
    """
    Derives hand information purely from YOLOv8-Pose wrist/elbow/shoulder keypoints.
    No extra model needed.

    Computes:
      - Wrist position (pixel)
      - Arm extension ratio  (0=bent, 1=straight)
      - Hand pointing direction (degrees)
      - Raised above shoulder flag
      - Contextual hand state label
    """

    WRIST_L = KP['left_wrist']
    WRIST_R = KP['right_wrist']
    ELBOW_L = KP['left_elbow']
    ELBOW_R = KP['right_elbow']
    SHLDR_L = KP['left_shoulder']
    SHLDR_R = KP['right_shoulder']

    def _get(self, kps, idx, thresh=0.4):
        if kps[idx][2] > thresh:
            return (float(kps[idx][0]), float(kps[idx][1]))
        return None

    def _extension(self, shoulder, elbow, wrist):
        if any(x is None for x in [shoulder, elbow, wrist]):
            return 0.0
        angle = calc_angle(shoulder, elbow, wrist)
        return float(np.clip((angle - 60) / 120.0, 0.0, 1.0))

    def _direction(self, wrist, elbow):
        if wrist is None or elbow is None:
            return None
        v = np.array(wrist) - np.array(elbow)
        return float(np.degrees(np.arctan2(-v[1], v[0])))

    def _raised(self, wrist, shoulder):
        if wrist is None or shoulder is None:
            return False
        return wrist[1] < shoulder[1]

    def _state(self, ext, raised):
        if ext > 0.85 and raised:
            return 'raised + extended ✋'
        if ext > 0.75:
            return 'extended →'
        if ext < 0.30:
            return 'bent / fist ✊'
        return 'mid position'

    def analyze_from_pose(self, kps):
        hands = []
        for side, w_i, e_i, s_i, ck in [
            ('Left',  self.WRIST_L, self.ELBOW_L, self.SHLDR_L, 'hand_l'),
            ('Right', self.WRIST_R, self.ELBOW_R, self.SHLDR_R, 'hand_r'),
        ]:
            wrist    = self._get(kps, w_i)
            elbow    = self._get(kps, e_i)
            shoulder = self._get(kps, s_i)
            if wrist is None:
                continue
            ext   = self._extension(shoulder, elbow, wrist)
            dirn  = self._direction(wrist, elbow)
            rasd  = self._raised(wrist, shoulder)
            hands.append({
                'side':      side,
                'wrist_px':  (int(wrist[0]), int(wrist[1])),
                'extension': ext,
                'direction': dirn,
                'raised':    rasd,
                'state':     self._state(ext, rasd),
                'color':     COLORS[ck],
            })
        return hands

    def draw(self, frame, hands):
        for hand in hands:
            wx, wy = hand['wrist_px']
            color  = hand['color']
            cv2.circle(frame, (wx, wy), 11, color, 2)
            cv2.circle(frame, (wx, wy),  4, color, -1)
            if hand['direction'] is not None:
                rad = np.radians(hand['direction'])
                ext = int(hand['extension'] * 55 + 15)
                tx  = int(wx + ext * np.cos(rad))
                ty  = int(wy - ext * np.sin(rad))
                cv2.arrowedLine(frame, (wx, wy), (tx, ty), color, 2, tipLength=0.3)
            cv2.putText(frame, hand['state'],
                        (wx - 45, wy - 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.44, color, 1)
            # Extension bar
            blen   = 44
            filled = int(blen * hand['extension'])
            bx, by = wx - 22, wy - 40
            cv2.rectangle(frame, (bx, by), (bx + blen, by + 9), (40,40,40), -1)
            cv2.rectangle(frame, (bx, by), (bx + filled, by + 9), color, -1)


# ══════════════════════════════════════════════════════════════
#  LEVEL 3 — FACE ANALYSIS (YuNet + LBF 68-point)
# ══════════════════════════════════════════════════════════════

class FaceAnalyzer:
    """
    Detection  : YuNet  (OpenCV official ONNX model)
    Landmarks  : LBF    (OpenCV face module — 68 points)
    Fallback   : Haar cascade if YuNet model not found

    Metrics computed from 68 landmarks:
      EAR         Eye Aspect Ratio  (blink / drowsiness)
      MAR         Mouth Aspect Ratio (mouth open / speech)
      brow_raise  Normalized eyebrow height (surprise / attention)
      corners_up  Mouth corner elevation (smile / frown)
      expression  AU-based label
      head pose   yaw / pitch / roll from facial geometry
      symmetry    left vs right face balance
      blink count running total
    """

    def __init__(self, yunet_path=YUNET_PATH, lbf_path=LBF_PATH,
                 input_size=(640, 480)):
        self.W, self.H    = input_size
        self.detector     = None
        self.facemark     = None
        self.haar         = None
        self.mode         = 'none'
        self.blink_count  = 0
        self.blink_total  = 0
        self.ear_buf      = deque(maxlen=4)
        self._init_detector(yunet_path)
        self._init_landmarks(lbf_path)

    # ── init ──────────────────────────────────────────────────

    def _init_detector(self, path):
        if os.path.exists(path):
            try:
                self.detector = cv2.FaceDetectorYN.create(
                    path, "", (self.W, self.H),
                    score_threshold=0.55,
                    nms_threshold=0.30,
                    top_k=1
                )
                self.mode = 'yunet'
                print(f"[L3] YuNet loaded ✓")
                return
            except Exception as e:
                print(f"[L3] YuNet error: {e}")
        # Haar fallback
        try:
            hpath = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
            self.haar = cv2.CascadeClassifier(hpath)
            self.mode = 'haar'
            print("[L3] Haar cascade fallback active")
        except Exception as e:
            print(f"[L3] Face detection unavailable: {e}")

    def _init_landmarks(self, path):
        if os.path.exists(path):
            try:
                self.facemark = cv2.face.createFacemarkLBF()
                self.facemark.loadModel(path)
                print(f"[L3] LBF 68-pt landmarks loaded ✓")
            except Exception as e:
                print(f"[L3] LBF error: {e}")
        else:
            print(f"[L3] lbfmodel.yaml not found — landmark analysis disabled")
            print(f"     Get it: https://github.com/kurnianggoro/GSOC2017/raw/master/data/lbfmodel.yaml")

    # ── helpers ───────────────────────────────────────────────

    @staticmethod
    def _p(lm, i):
        return np.array(lm[i], dtype=float)

    @staticmethod
    def _d(a, b):
        return float(np.linalg.norm(a - b))

    def _ear(self, lm, eye_idx):
        """Eye Aspect Ratio — 6 point ring."""
        pts = [self._p(lm, i) for i in eye_idx]
        v1  = self._d(pts[1], pts[5])
        v2  = self._d(pts[2], pts[4])
        h   = self._d(pts[0], pts[3])
        return (v1 + v2) / (2.0 * h + 1e-6)

    def _mar(self, lm):
        """Mouth Aspect Ratio — outer lip."""
        top   = self._p(lm, 51)
        bot   = self._p(lm, 57)
        left  = self._p(lm, 48)
        right = self._p(lm, 54)
        return self._d(top, bot) / (self._d(left, right) + 1e-6)

    def _brow_raise(self, lm):
        """Normalized brow-to-eye distance."""
        l_brow  = self._p(lm, 19)
        r_brow  = self._p(lm, 24)
        l_eye_c = np.mean([self._p(lm, i) for i in LBF_LEYE], axis=0)
        r_eye_c = np.mean([self._p(lm, i) for i in LBF_REYE], axis=0)
        eye_w   = self._d(self._p(lm, 36), self._p(lm, 39))
        l_raise = (l_eye_c[1] - l_brow[1]) / (eye_w + 1e-6)
        r_raise = (r_eye_c[1] - r_brow[1]) / (eye_w + 1e-6)
        return float((l_raise + r_raise) / 2.0)

    def _corners_up(self, lm):
        """Smile metric — positive = corners above mid lip line."""
        lc   = self._p(lm, 48)
        rc   = self._p(lm, 54)
        top  = self._p(lm, 51)
        bot  = self._p(lm, 57)
        mid_y = (top[1] + bot[1]) / 2.0
        avg_c = (lc[1] + rc[1]) / 2.0
        mouth_h = abs(bot[1] - top[1]) + 1e-6
        return float((mid_y - avg_c) / mouth_h)

    def _head_pose(self, lm, fw, fh):
        """Yaw/pitch/roll from facial geometry."""
        nose      = self._p(lm, 30)
        chin      = self._p(lm, 8)
        l_eye_out = self._p(lm, 36)
        r_eye_out = self._p(lm, 45)
        l_mouth   = self._p(lm, 48)
        r_mouth   = self._p(lm, 54)

        face_cx  = (l_eye_out[0] + r_eye_out[0]) / 2.0
        eye_mid_y= (l_eye_out[1] + r_eye_out[1]) / 2.0
        eye_w    = self._d(l_eye_out, r_eye_out)
        face_h   = abs(chin[1] - eye_mid_y) + 1e-6

        # Yaw
        dx      = (nose[0] - face_cx) / (eye_w + 1e-6)
        yaw     = 'right →' if dx > 0.12 else ('left ←' if dx < -0.12 else 'center')

        # Pitch
        dy      = (nose[1] - eye_mid_y) / face_h
        pitch   = 'up ↑' if dy < 0.20 else ('down ↓' if dy > 0.55 else 'level')

        # Roll
        ev      = r_eye_out - l_eye_out
        roll    = float(np.degrees(np.arctan2(ev[1], ev[0])))

        # Symmetry
        l_m2e   = self._d(l_mouth, l_eye_out)
        r_m2e   = self._d(r_mouth, r_eye_out)
        sym     = round(min(l_m2e, r_m2e) / (max(l_m2e, r_m2e) + 1e-6), 2)

        return {'yaw': yaw, 'pitch': pitch, 'roll': round(roll, 1), 'symmetry': sym}

    def _expression(self, ear, mar, brow, corners):
        if ear < 0.18:
            return 'eyes closed 😴'
        if mar > 0.55 and brow > 0.48:
            return 'surprised 😮'
        if mar > 0.50:
            return 'mouth open 😮'
        if brow > 0.55:
            return 'brows raised 🤨'
        if corners > 0.12:
            return 'smiling 😊'
        if corners < -0.12:
            return 'frowning 😠'
        return 'neutral 😐'

    # ── detect ────────────────────────────────────────────────

    def _detect_boxes(self, gray, frame):
        if self.mode == 'yunet':
            h, w = frame.shape[:2]
            self.detector.setInputSize((w, h))
            _, faces = self.detector.detect(frame)
            if faces is None:
                return []
            return [(int(f[0]), int(f[1]), int(f[2]), int(f[3])) for f in faces]
        elif self.mode == 'haar':
            faces = self.haar.detectMultiScale(
                gray, scaleFactor=1.1, minNeighbors=5, minSize=(60, 60))
            if len(faces) == 0:
                return []
            return [(int(x), int(y), int(w), int(h)) for x, y, w, h in faces]
        return []

    # ── main ──────────────────────────────────────────────────

    def analyze(self, frame):
        gray  = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        boxes = self._detect_boxes(gray, frame)
        if not boxes:
            return None

        box    = max(boxes, key=lambda b: b[2] * b[3])
        x, y, w, h = box

        result = {
            'bbox':       box,
            'landmarks':  None,
            'ear':        None,
            'mar':        None,
            'brow_raise': None,
            'expression': 'detected',
            'head':       None,
            'blink_total':self.blink_total,
            'eyes_open':  True,
            'mouth_open': False,
        }

        if self.facemark is not None:
            rect_arr = np.array([[x, y, w, h]], dtype=np.int32)
            ok, shapes = self.facemark.fit(gray, rect_arr)
            if ok and len(shapes) > 0:
                lm = [(int(p[0][0]), int(p[0][1])) for p in shapes[0][0]]
                if len(lm) == 68:
                    result['landmarks'] = lm

                    ear   = (self._ear(lm, LBF_LEYE) + self._ear(lm, LBF_REYE)) / 2.0
                    mar   = self._mar(lm)
                    brow  = self._brow_raise(lm)
                    corn  = self._corners_up(lm)
                    head  = self._head_pose(lm, frame.shape[1], frame.shape[0])
                    expr  = self._expression(ear, mar, brow, corn)

                    # Blink
                    self.ear_buf.append(ear)
                    avg_ear = float(np.mean(self.ear_buf))
                    eyes_open = avg_ear > 0.20
                    if not eyes_open:
                        self.blink_count += 1
                    elif self.blink_count >= 2:
                        self.blink_total += 1
                        self.blink_count  = 0

                    result.update({
                        'ear':         round(ear,  3),
                        'mar':         round(mar,  3),
                        'brow_raise':  round(brow, 3),
                        'expression':  expr,
                        'head':        head,
                        'blink_total': self.blink_total,
                        'eyes_open':   eyes_open,
                        'mouth_open':  mar > 0.45,
                    })
        return result

    # ── draw ──────────────────────────────────────────────────

    def draw(self, frame, face):
        if not face:
            return
        x, y, w, h = face['bbox']
        lm = face.get('landmarks')

        # Face box
        cv2.rectangle(frame, (x, y), (x+w, y+h), COLORS['face'], 2)

        if lm and len(lm) == 68:
            # Jawline
            for i in range(len(LBF_JAW) - 1):
                cv2.line(frame, lm[LBF_JAW[i]], lm[LBF_JAW[i+1]], COLORS['face_dim'], 1)

            # Eyebrows
            for brow in [LBF_LBROW, LBF_RBROW]:
                for i in range(len(brow) - 1):
                    cv2.line(frame, lm[brow[i]], lm[brow[i+1]], COLORS['face'], 1)

            # Eyes
            for eye_idx in [LBF_LEYE, LBF_REYE]:
                pts = np.array([lm[i] for i in eye_idx], np.int32)
                cv2.polylines(frame, [pts], True, COLORS['face'], 1)
                # Pupil estimate
                cx = int(np.mean([lm[i][0] for i in eye_idx]))
                cy = int(np.mean([lm[i][1] for i in eye_idx]))
                cv2.circle(frame, (cx, cy), 3, (0, 255, 255), -1)

            # Nose bridge
            for i in range(len(LBF_NOSE) - 1):
                cv2.line(frame, lm[LBF_NOSE[i]], lm[LBF_NOSE[i+1]], COLORS['face_dim'], 1)

            # Outer mouth
            pts = np.array([lm[i] for i in LBF_OUTER_MOUTH], np.int32)
            cv2.polylines(frame, [pts], True, COLORS['face'], 1)

            # Inner mouth
            pts = np.array([lm[i] for i in LBF_INNER_MOUTH], np.int32)
            cv2.polylines(frame, [pts], True, COLORS['face_dim'], 1)

            # Nose tip
            cv2.circle(frame, lm[30], 4, COLORS['face'], -1)

            # Expression label
            expr = face.get('expression', '')
            cv2.putText(frame, expr, (x, y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.56, COLORS['face'], 2)

            # Head direction arrow (from nose tip, along roll angle)
            head = face.get('head')
            if head:
                rad = np.radians(head['roll'])
                nx, ny = lm[30]
                L  = 42
                tx = int(nx + L * np.cos(rad + np.pi / 2))
                ty = int(ny + L * np.sin(rad + np.pi / 2))
                cv2.arrowedLine(frame, (nx, ny), (tx, ty), (0,255,255), 2, tipLength=0.28)
        else:
            cv2.putText(frame, face.get('expression', 'face detected'),
                        (x, y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.56, COLORS['face'], 2)


# ══════════════════════════════════════════════════════════════
#  MEMORY SYSTEM
# ══════════════════════════════════════════════════════════════

class APEXMemory:
    def __init__(self, db_path="apex_memory.db"):
        self.conn           = sqlite3.connect(db_path, check_same_thread=False)
        self._init_db()
        self.short_term     = deque(maxlen=60)
        self.pattern_buffer = deque(maxlen=200)
        self.learned        = {}
        self._load_patterns()

    def _init_db(self):
        self.conn.executescript("""
            CREATE TABLE IF NOT EXISTS sessions (
                id            INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp     TEXT,
                duration_s    REAL,
                actions       TEXT,
                avg_angles    TEXT,
                posture_score REAL
            );
            CREATE TABLE IF NOT EXISTS patterns (
                id          INTEGER PRIMARY KEY AUTOINCREMENT,
                name        TEXT UNIQUE,
                sequence    TEXT,
                occurrences INTEGER DEFAULT 1,
                last_seen   TEXT
            );
            CREATE TABLE IF NOT EXISTS performance (
                id     INTEGER PRIMARY KEY AUTOINCREMENT,
                date   TEXT,
                metric TEXT,
                value  REAL
            );
        """)
        self.conn.commit()

    def push(self, data):
        data['ts'] = datetime.now().isoformat()
        self.short_term.append(data)
        self.pattern_buffer.append(data.get('action', 'unknown'))

    def recent_actions(self, n=15):
        return [f.get('action', '') for f in list(self.short_term)[-n:]]

    def learn_pattern(self, sequence):
        key = '→'.join(sequence)
        if key in self.learned:
            self.learned[key]['count'] += 1
        else:
            self.learned[key] = {'name': f"pat_{len(self.learned)+1}", 'count': 1}
        c = self.conn.cursor()
        c.execute("""
            INSERT INTO patterns (name, sequence, occurrences, last_seen)
            VALUES (?, ?, 1, ?)
            ON CONFLICT(name) DO UPDATE SET
                occurrences = occurrences + 1,
                last_seen   = excluded.last_seen
        """, (self.learned[key]['name'], key, datetime.now().isoformat()))
        self.conn.commit()

    def detect_pattern(self, window=15):
        recent = '→'.join(list(self.pattern_buffer)[-window:])
        for seq in self.learned:
            if seq in recent:
                return self.learned[seq]['name']
        return None

    def _load_patterns(self):
        for name, seq, cnt in self.conn.execute(
                "SELECT name, sequence, occurrences FROM patterns"):
            self.learned[seq] = {'name': name, 'count': cnt}

    def save_session(self, duration, actions, avg_angles, avg_posture):
        c = self.conn.cursor()
        c.execute("""
            INSERT INTO sessions (timestamp, duration_s, actions, avg_angles, posture_score)
            VALUES (?, ?, ?, ?, ?)
        """, (datetime.now().isoformat(), round(duration,2),
              json.dumps(actions),
              json.dumps({k: round(v,1) for k,v in avg_angles.items()}),
              round(avg_posture, 2)))
        self.conn.commit()
        return c.lastrowid

    def save_metric(self, metric, value):
        self.conn.execute(
            "INSERT INTO performance (date,metric,value) VALUES (?,?,?)",
            (datetime.now().date().isoformat(), metric, value))
        self.conn.commit()

    def compare(self, metric, current):
        row = self.conn.execute(
            "SELECT AVG(value),MAX(value) FROM performance WHERE metric=?",
            (metric,)).fetchone()
        if not row or row[0] is None:
            return {'vs_avg': 'First session!'}
        avg, best = row
        diff = current - avg
        sign = '+' if diff >= 0 else ''
        return {'vs_avg': f"vs avg: {sign}{round(diff,1)}", 'best': round(best,1)}

    def stats(self):
        ns = self.conn.execute("SELECT COUNT(*) FROM sessions").fetchone()[0]
        np_ = self.conn.execute("SELECT COUNT(*) FROM patterns").fetchone()[0]
        return ns, np_

    def last_sessions(self, n=5):
        return self.conn.execute("""
            SELECT timestamp, duration_s, posture_score, actions
            FROM sessions ORDER BY id DESC LIMIT ?
        """, (n,)).fetchall()


# ══════════════════════════════════════════════════════════════
#  HUD
# ══════════════════════════════════════════════════════════════

def draw_hud(frame, d):
    h, w    = frame.shape[:2]
    action  = d.get('action',  'unknown')
    conf    = d.get('conf',    0.0)
    p_score = d.get('posture', 0.0)
    angles  = d.get('angles',  {})
    fps     = d.get('fps',     0)
    mem     = d.get('mem',     (0, 0, ''))
    hands   = d.get('hands',   [])
    face    = d.get('face',    None)

    overlay = frame.copy()
    cv2.rectangle(overlay, (0,0), (272, h), (8,8,8), -1)
    cv2.addWeighted(overlay, 0.58, frame, 0.42, 0, frame)

    def T(txt, y, col=(210,210,210), sc=0.48, th=1):
        cv2.putText(frame, txt, (10, y), cv2.FONT_HERSHEY_SIMPLEX, sc, col, th)

    # L1
    T("─ L1  BODY ─",          25, (0,255,120), 0.52, 2)
    T(f"FPS    {fps}",          44, (140,140,140))
    T(f"ACT    {action.upper()}", 63, (0,255,200), 0.52, 2)
    T(f"CONF   {conf:.0%}",     82, (140,140,140))
    pc = (0,200,0) if p_score>70 else (0,140,255) if p_score>40 else (0,0,210)
    T(f"POST   {p_score:.0f}/100", 101, pc, 0.52, 2)
    y = 118
    for joint, ang in list(angles.items())[:5]:
        T(f" {joint[:13]}: {ang:.0f}°", y, (120,120,120), 0.37); y += 14

    # L2
    T("─ L2  HANDS ─",         y+7, (0,200,255), 0.50, 2); y += 24
    if hands:
        for hand in hands[:2]:
            c = hand['color']
            T(f" {hand['side']}: {hand['state']}", y, c, 0.42); y += 14
            T(f"   ext:{hand['extension']:.0%}  up:{hand['raised']}", y, (110,110,110), 0.37); y += 13
    else:
        T(" no hands detected", y, (75,75,75), 0.38); y += 13

    # L3
    T("─ L3  FACE ─",          y+7, (180,80,255), 0.50, 2); y += 24
    if face:
        T(f" {face.get('expression','?')}", y, (180,80,255), 0.44); y += 15
        ear = face.get('ear')
        mar = face.get('mar')
        if ear is not None:
            T(f" EAR:{ear:.2f}  MAR:{mar:.2f}", y, (120,120,120), 0.38); y += 13
        brow = face.get('brow_raise')
        if brow is not None:
            T(f" brow raise: {brow:.2f}", y, (120,120,120), 0.38); y += 13
        head = face.get('head')
        if head:
            T(f" {head['yaw']}  {head['pitch']}", y, (150,150,150), 0.38); y += 13
            T(f" roll:{head['roll']}°  sym:{head['symmetry']}", y, (120,120,120), 0.36); y += 13
        eyes_s  = "open" if face.get('eyes_open',True) else "CLOSED ⚠"
        mouth_s = "open" if face.get('mouth_open',False) else "closed"
        T(f" eyes:{eyes_s}  mouth:{mouth_s}", y, (120,120,120), 0.36); y += 13
        T(f" blinks: {face.get('blink_total',0)}", y, (120,120,120), 0.38)
    else:
        T(" no face detected", y, (75,75,75), 0.38)

    # Memory
    T("─ MEMORY ─",            h-62, (0,170,255), 0.42)
    T(f"sess:{mem[0]}  pat:{mem[1]}", h-47, (110,110,110), 0.37)
    T(str(mem[2]),             h-32, (0,210,90), 0.38)

    # Posture bar
    bx, by, bw, bh = w-152, 13, 132, 10
    cv2.rectangle(frame, (bx,by), (bx+bw, by+bh), (30,30,30), -1)
    fill = int(bw * p_score / 100)
    cv2.rectangle(frame, (bx,by), (bx+fill, by+bh), pc, -1)
    cv2.putText(frame, "POSTURE", (bx, by-3),
                cv2.FONT_HERSHEY_SIMPLEX, 0.32, (150,150,150), 1)

    # Point counter
    n_f     = 68 if (face and face.get('landmarks')) else (1 if face else 0)
    total   = 17 + len(hands) + n_f
    cv2.putText(frame, f"{total} pts tracked",
                (w-152, by+24), cv2.FONT_HERSHEY_SIMPLEX, 0.34, (90,90,90), 1)

    return frame


# ══════════════════════════════════════════════════════════════
#  SESSION
# ══════════════════════════════════════════════════════════════

def save_session(memory, start, action_log, angle_history, posture_scores):
    dur = time.time() - start
    acts = dict(Counter(action_log))
    avga = {k: float(np.mean(v)) for k, v in angle_history.items()}
    avgp = float(np.mean(posture_scores)) if posture_scores else 0.0
    sid  = memory.save_session(dur, acts, avga, avgp)
    memory.save_metric('posture_score', avgp)
    print(f"\n[MEMORY] Session #{sid}  {dur:.1f}s  posture:{avgp:.1f}")
    print(f"         actions: {acts}")
    for row in memory.last_sessions(3):
        print(f"  {row[0][:16]} | {row[1]:.0f}s | posture:{row[2]:.0f}")


# ══════════════════════════════════════════════════════════════
#  MAIN LOOP
# ══════════════════════════════════════════════════════════════

def run(source=0, lstm_path=None):
    print("\n[APEX v3] Initializing models...")

    print("  [1/4] YOLOv8-Pose...")
    pose_model = YOLO('yolov8n-pose.pt')

    print("  [2/4] Hand analyzer...")
    hand_analyzer = HandAnalyzer()

    print("  [3/4] Face analyzer (YuNet + LBF)...")
    cap_tmp = cv2.VideoCapture(source)
    ret, frm = cap_tmp.read()
    cap_tmp.release()
    in_size = (frm.shape[1], frm.shape[0]) if ret else (1280, 720)
    face_analyzer = FaceAnalyzer(input_size=in_size)

    print("  [4/4] LSTM + Memory...")
    recognizer = ActionRecognizer(model_path=lstm_path)
    memory     = APEXMemory()

    # ── Main capture ──
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        print(f"[APEX] Cannot open: {source}"); return
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    # ── Face thread setup ──
    # Face analysis (LBF) is the slowest step (~180ms).
    # Running it in a separate thread lets L1+L2 run at full speed
    # while L3 updates asynchronously → 20-28 FPS vs 3 FPS.
    face_lock    = threading.Lock()
    face_result  = [None]          # shared slot between threads
    stop_event   = threading.Event()
    cap2         = cv2.VideoCapture(source)  # dedicated capture for face thread
    cap2.set(cv2.CAP_PROP_FRAME_WIDTH,  1280)
    cap2.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    def face_thread_loop():
        while not stop_event.is_set():
            ret2, frame2 = cap2.read()
            if not ret2:
                time.sleep(0.01)
                continue
            if isinstance(source, int):
                frame2 = cv2.flip(frame2, 1)
            f = face_analyzer.analyze(frame2)
            with face_lock:
                face_result[0] = f

    face_t = threading.Thread(target=face_thread_loop, daemon=True)
    face_t.start()

    start      = time.time()
    prev_t     = time.time()
    fps_buf    = deque(maxlen=30)
    fc         = 0
    action_log = []
    angle_hist = {}
    p_scores   = []
    last = dict(angles={}, action='unknown', conf=0.0, posture=0.0,
                hands=[], face=None)

    print("\n[APEX v3] Running  —  Q=quit  S=save\n")

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if isinstance(source, int):
            frame = cv2.flip(frame, 1)
        fc += 1

        # ── L1: Pose (main thread) ──
        results = pose_model(frame, verbose=False, conf=0.4)[0]
        persons = []
        if results.keypoints is not None:
            for kps_t in results.keypoints.data:
                kps = kps_t.cpu().numpy()
                persons.append(kps)
                draw_skeleton(frame, kps)

        if persons:
            kps          = persons[0]
            angles       = extract_angles(kps)
            pscore       = posture_score(angles)
            action, conf = recognizer.predict(angles)
            p_scores.append(pscore)
            action_log.append(action)
            for k, v in angles.items():
                angle_hist.setdefault(k, []).append(v)
            memory.push({'action': action, 'confidence': conf,
                         'posture': pscore, 'angles': angles})
            if fc % 30 == 0:
                memory.learn_pattern(memory.recent_actions())

            # ── L2: Hands (main thread — fast, from pose kps) ──
            hands = hand_analyzer.analyze_from_pose(kps)
            hand_analyzer.draw(frame, hands)
            last.update(angles=angles, action=action, conf=conf,
                        posture=pscore, hands=hands)
        else:
            last['hands'] = []

        # ── L3: Face (read latest result from face thread) ──
        with face_lock:
            face = face_result[0]
        face_analyzer.draw(frame, face)
        last['face'] = face

        # ── FPS ──
        now = time.time()
        fps_buf.append(1.0 / (now - prev_t + 1e-6))
        fps    = int(np.mean(fps_buf))
        prev_t = now

        # ── HUD ──
        ns, np_ = memory.stats()
        cmp     = memory.compare('posture_score', last['posture'])
        frame   = draw_hud(frame, {
            'action':  last['action'],  'conf':    last['conf'],
            'posture': last['posture'], 'angles':  last['angles'],
            'fps':     fps,             'mem':     (ns, np_, cmp.get('vs_avg', '')),
            'hands':   last['hands'],   'face':    last['face'],
        })

        cv2.imshow("APEX v3  |  L1:Body  L2:Hands  L3:Face  |  Q=quit  S=save", frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('s'):
            save_session(memory, start, action_log, angle_hist, p_scores)

    # ── Cleanup ──
    stop_event.set()
    cap2.release()
    save_session(memory, start, action_log, angle_hist, p_scores)
    cap.release()
    cv2.destroyAllWindows()
    print("\n[APEX v3] Done.")


# ══════════════════════════════════════════════════════════════
#  ENTRY POINT
# ══════════════════════════════════════════════════════════════

if __name__ == "__main__":
    source = "test_video.f401.mp4"        
    run(source=source, lstm_path=None)