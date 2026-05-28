"""
enroll_camera.py — camera loop + pose validation gates for enrollment
"""

import cv2
import numpy as np
import time
import threading
from enroll_config import (
    POSES, FACE_HOLD_TIME, POSE_SHIFT_PX, SMILE_RATIO_MIN
)

face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)


def detect_face(frame_bgr):
    """Return (x, y, w, h) of largest face or None."""
    gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.1, 5, minSize=(80, 80))
    if not len(faces):
        return None
    return max(faces, key=lambda f: f[2] * f[3])


def check_pose_gate(face, frame_shape, gate_type, baseline_cx, baseline_cy, frame=None):
    """
    Returns (gate_ok: bool, hint: str, new_baseline: tuple)
    Validates that the user is in the correct pose before capturing.
    """
    if face is None:
        return False, "No face detected — centre your face", (baseline_cx, baseline_cy)

    fh, fw = frame_shape[:2]
    x, y, w, h = face
    cx = x + w // 2
    cy = y + h // 2

    # Update baseline if not set
    if baseline_cx is None:
        baseline_cx = cx
        baseline_cy = cy

    # ── Centre gate ─────────────────────────────────────
    if gate_type == "centre":
        ok = abs(cx - fw // 2) < 60 and w > 100 and h > 100
        hint = "Centre your face and move closer" if not ok else "Hold still…"
        return ok, hint, (baseline_cx, baseline_cy)

    # ── Left turn gate ──────────────────────────────────
    if gate_type == "left":
        shift = baseline_cx - cx
        ok = shift > POSE_SHIFT_PX
        hint = f"Turn more LEFT  ↙  ({max(0, POSE_SHIFT_PX - shift):.0f}px more)" if not ok else "Good — hold…"
        return ok, hint, (baseline_cx, baseline_cy)

    # ── Right turn gate ─────────────────────────────────
    if gate_type == "right":
        shift = cx - baseline_cx
        ok = shift > POSE_SHIFT_PX
        hint = f"Turn more RIGHT  ↘  ({max(0, POSE_SHIFT_PX - shift):.0f}px more)" if not ok else "Good — hold…"
        return ok, hint, (baseline_cx, baseline_cy)

    # ── Up tilt gate ────────────────────────────────────
    if gate_type == "up":
        shift = baseline_cy - cy
        ok = shift > POSE_SHIFT_PX
        hint = f"Tilt head UP  ↑  ({max(0, POSE_SHIFT_PX - shift):.0f}px more)" if not ok else "Good — hold…"
        return ok, hint, (baseline_cx, baseline_cy)

    # ── Down tilt gate ──────────────────────────────────
    if gate_type == "down":
        shift = cy - baseline_cy
        ok = shift > POSE_SHIFT_PX
        hint = f"Tilt head DOWN  ↓  ({max(0, POSE_SHIFT_PX - shift):.0f}px more)" if not ok else "Good — hold…"
        return ok, hint, (baseline_cx, baseline_cy)

    # ── Smile gate ──────────────────────────────────────
    if gate_type == "smile" and frame is not None:
        x, y, w, h = face
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        upper = gray[y:y + h // 2, x:x + w]
        lower = gray[y + h // 2:y + h, x:x + w]
        if upper.size == 0 or lower.size == 0:
            return False, "Smile and show your teeth ☺", (baseline_cx, baseline_cy)
        ratio = float(np.mean(lower)) / (float(np.mean(upper)) + 1e-6)
        ok = ratio > (1.0 + SMILE_RATIO_MIN)
        hint = "Smile wider and show teeth ☺" if not ok else "Great smile — hold…"
        return ok, hint, (baseline_cx, baseline_cy)

    return True, "Hold still…", (baseline_cx, baseline_cy)


def draw_overlay(display, face, gate_ok, hint, progress, pose_label):
    """Draw face box, progress arc, instruction bar."""
    h, w = display.shape[:2]

    # Instruction bar
    cv2.rectangle(display, (0, 0), (w, 44), (13, 17, 23), -1)
    cv2.putText(display, pose_label, (12, 16),
                cv2.FONT_HERSHEY_SIMPLEX, 0.52, (88, 166, 255), 1)
    hint_col = (63, 185, 80) if gate_ok else (247, 129, 102)
    cv2.putText(display, hint, (12, 36),
                cv2.FONT_HERSHEY_SIMPLEX, 0.48, hint_col, 1)

    if face is not None:
        x, y, fw, fh = face
        box_col = (63, 185, 80) if gate_ok else (88, 166, 255)
        cv2.rectangle(display, (x, y), (x + fw, y + fh), box_col, 2)

        if gate_ok and progress > 0:
            cx, cy = x + fw // 2, y + fh // 2
            radius = max(fw, fh) // 2 + 16
            for angle in range(0, int(360 * progress), 3):
                rad = np.deg2rad(angle - 90)
                px = int(cx + radius * np.cos(rad))
                py = int(cy + radius * np.sin(rad))
                cv2.circle(display, (px, py), 3, (63, 185, 80), -1)

    return display


class CameraCapture:
    """Manages camera loop for enrollment with pose validation."""

    def __init__(self, on_captured, on_frame, on_error):
        self.on_captured = on_captured
        self.on_frame = on_frame
        self.on_error = on_error
        self.cap = None
        self.running = False
        self.pose_index = 0
        self.current_frame = None
        self.baseline_cx = None
        self.baseline_cy = None
        self.face_stable_t = None
        self.completed = False  # Flag to stop loop when done

    def start(self):
        self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened():
            self.on_error("Could not open camera.")
            return
        self.running = True
        threading.Thread(target=self._loop, daemon=True).start()

    def stop(self):
        self.running = False
        time.sleep(0.12)
        if self.cap and self.cap.isOpened():
            self.cap.release()

    def manual_capture(self):
        if self.current_frame is not None and self.pose_index < len(POSES):
            self._do_capture(self.current_frame)

    def _loop(self):
        while self.running:
            ret, frame = self.cap.read()
            if not ret:
                time.sleep(0.03)
                continue
            frame = cv2.flip(frame, 1)
            self.current_frame = frame.copy()

            # Guard against index out of range
            if self.pose_index >= len(POSES):
                if not self.completed:
                    self.completed = True
                    # Signal that all poses are captured
                    self.on_frame(frame, None, False, "All poses captured!")
                time.sleep(0.04)
                continue

            face = detect_face(frame)
            pose_id, pose_label, gate_type = POSES[self.pose_index]
            display_label = f"Pose {self.pose_index + 1}/{len(POSES)}: {pose_label}"

            # Check pose gate
            gate_ok, hint, (self.baseline_cx, self.baseline_cy) = check_pose_gate(
                face, frame.shape, gate_type,
                self.baseline_cx, self.baseline_cy, frame
            )

            now = time.time()
            if gate_ok and face is not None:
                if self.face_stable_t is None:
                    self.face_stable_t = now
                progress = min((now - self.face_stable_t) / FACE_HOLD_TIME, 1.0)
                if progress >= 1.0:
                    self._do_capture(frame)
                    self.face_stable_t = None
                    progress = 0.0
            else:
                self.face_stable_t = None
                progress = 0.0

            display = draw_overlay(
                cv2.resize(frame, (480, 360)),
                self._scale_face(face, frame.shape, (480, 360)),
                gate_ok, hint, progress, display_label
            )
            self.on_frame(display, face, gate_ok, hint)
            time.sleep(0.04)

        if self.cap:
            self.cap.release()

    def _do_capture(self, frame):
        if self.pose_index >= len(POSES):
            return
        pose_id, _, _ = POSES[self.pose_index]
        self.pose_index += 1
        self.on_captured(frame.copy(), pose_id)

    @staticmethod
    def _scale_face(face, src_shape, dst_size):
        if face is None:
            return None
        sh, sw = src_shape[:2]
        dw, dh = dst_size
        x, y, w, h = face
        return (int(x * dw / sw), int(y * dh / sh),
                int(w * dw / sw), int(h * dh / sh))