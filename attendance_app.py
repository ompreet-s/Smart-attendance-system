"""
Smart Attendance System — Main Entry Point
Run: python attendance_app.py
"""

import os
import csv
import tkinter as tk
from tkinter import messagebox
import time
import threading
import cv2
from datetime import datetime

from attend_config import STUDENTS_DIR, ATTENDANCE_DIR
from attend_detectors import load_students, log_attendance
from attend_pipeline import VerificationPipeline, S_FACE, S_DONE
from attend_ui import AttendanceUI


def _already_marked_today(roll):
    """Return True if this roll number already has an entry in today's CSV."""
    today = datetime.now().strftime("%Y-%m-%d")
    csv_path = os.path.join(ATTENDANCE_DIR, f"attendance_{today}.csv")
    if not os.path.isfile(csv_path):
        return False
    with open(csv_path, newline="") as f:
        for row in csv.reader(f):
            if row and row[0] == roll:
                return True
    return False


class AttendanceApp:
    def __init__(self):
        os.makedirs(STUDENTS_DIR, exist_ok=True)
        os.makedirs(ATTENDANCE_DIR, exist_ok=True)

        self.students = load_students()
        self.cap = None
        self.camera_running = False
        self.current_frame = None
        self.pipeline = None
        self.ui = None
        self._last_frame_time = 0

        # ── Session state ──────────────────────────────────────────────
        # Set of roll numbers marked in THIS session (in-memory guard).
        # The CSV guard catches cross-session duplicates.
        self._marked_today = set()
        # Whether the operator has pressed "Start Attendance"
        self._session_active = False

        self._start_ui()
        self._start_camera()
        # Pipeline is created but NOT started — camera shows a live preview
        # but no matching runs until the operator starts the session.
        self._start_pipeline()

    # ── UI ─────────────────────────────────────────────────────────────

    def _start_ui(self):
        self.ui = AttendanceUI(
            self.students,
            self._reset_pipeline,
            on_session_start=self._on_session_start,
            on_session_stop=self._on_session_stop,
        )
        self.ui.protocol("WM_DELETE_WINDOW", self._on_close)

    # ── Camera ─────────────────────────────────────────────────────────

    def _start_camera(self):
        self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened():
            messagebox.showerror("Camera Error", "Could not open camera.")
            self.ui.after(100, self._on_close)
            return
        self.camera_running = True
        threading.Thread(target=self._camera_loop, daemon=True).start()

    def _camera_loop(self):
        while self.camera_running:
            ret, frame = self.cap.read()
            if not ret:
                time.sleep(0.03)
                continue
            frame = cv2.flip(frame, 1)
            self.current_frame = frame.copy()
            self.ui.after(0, self._process_frame, frame)
            time.sleep(0.04)
        if self.cap:
            self.cap.release()

    def _process_frame(self, frame):
        if self.pipeline is None:
            return
        # When session is inactive show a live feed but skip pipeline processing
        if not self._session_active:
            self.ui.update_frame_idle(frame)
            return
        result = self.pipeline.process_frame(frame)
        self.ui.update_frame(frame, result)
        self.ui.update_step(self.pipeline.step, "", "muted")

    # ── Pipeline ───────────────────────────────────────────────────────

    def _start_pipeline(self):
        self.pipeline = VerificationPipeline(
            self.students,
            on_complete=self._on_verification_complete,
            on_update=self._on_pipeline_update,
        )

    def _reset_pipeline(self):
        if self.pipeline:
            self.pipeline.reset()
        self.ui.refresh_steps(S_FACE)

    def _on_pipeline_update(self, step, msg, tone):
        self.ui.update_step(step, msg, tone)

    # ── Session start / stop ───────────────────────────────────────────

    def _on_session_start(self):
        """Operator pressed START ATTENDANCE."""
        self._session_active = True
        self._reset_pipeline()
        self.ui.set_session_ui(active=True)
        self.ui.status_lbl.config(
            text="Session started — stand in front of the camera", fg="#16A34A"
        )

    def _on_session_stop(self):
        """Operator pressed STOP SESSION."""
        self._session_active = False
        if self.pipeline:
            self.pipeline.reset()
        self.ui.set_session_ui(active=False)
        self.ui.status_lbl.config(
            text="Session stopped — press Start Attendance to begin", fg="#64748B"
        )

    # ── Verification complete ──────────────────────────────────────────

    def _on_verification_complete(self, student, frame, voice_skipped):
        if not student:
            return

        roll = student["roll"]

        # ── Duplicate guard ────────────────────────────────────────────
        # Check in-memory set first (fast), then CSV (catches app restarts)
        if roll in self._marked_today or _already_marked_today(roll):
            self._marked_today.add(roll)           # keep set in sync
            self.ui.show_already_marked(student)   # friendly UI message
            self.ui.after(3000, self._reset_pipeline)
            return

        # ── First-time mark ────────────────────────────────────────────
        now = datetime.now()
        snap_dir = os.path.join(ATTENDANCE_DIR, now.strftime("%Y-%m-%d"))
        os.makedirs(snap_dir, exist_ok=True)
        photo_path = os.path.join(
            snap_dir, f"{roll}_{now.strftime('%H%M%S')}.jpg"
        )
        if self.current_frame is not None:
            cv2.imwrite(photo_path, self.current_frame)

        log_attendance(student, photo_path, voice_skipped)
        self._marked_today.add(roll)

        self.ui.show_marked(student, photo_path, now, voice_skipped)
        self.ui.add_attendee(student, self.current_frame)

        # Auto-reset after 4 seconds so next student can go
        self.ui.after(4000, self._reset_pipeline)

    # ── Close ──────────────────────────────────────────────────────────

    def _on_close(self):
        self.camera_running = False
        time.sleep(0.15)
        if self.cap and self.cap.isOpened():
            self.cap.release()
        self.ui.destroy()

    def run(self):
        self.ui.mainloop()


if __name__ == "__main__":
    app = AttendanceApp()
    app.run()