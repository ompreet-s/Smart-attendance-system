"""
attend_pipeline.py — sequential 4-step verification state machine
"""

import cv2
import time
import threading
from attend_config import (
    FACE_MATCH_FRAMES, HAND_CONFIRM_FRAMES, BLINK_CONSEC_FRAMES,
    NOD_SHIFT_PX, NOD_WINDOW_S, RECORD_SECONDS, VOICE_MAX_RETRIES
)
from attend_detectors import (
    detect_face, match_face, detect_hand, detect_blink,
    record_voice, match_voice, log_attendance, AUDIO_OK
)

S_FACE = "face"
S_HAND_PROMPT = "hand_prompt"
S_HAND = "hand"
S_LIVENESS = "liveness"
S_VOICE = "voice"
S_DONE = "done"


class VerificationPipeline:
    def __init__(self, students, on_complete, on_update):
        self.students = students
        self.on_complete = on_complete
        self.on_update = on_update
        self.reset()

    def reset(self):
        self.step = S_FACE
        self.matched_student = None
        self._face_buf = []
        self._prompt_start = None
        self._hand_buf = 0
        self._liveness_start = None
        self._blink_ok = False
        self._blink_frames = 0
        self._head_base_y = None
        self._nodded = False
        self._voice_retries = 0
        self._voice_busy = False
        self._voice_skipped = False
        self._captured_frame = None
        self.done = False

    def process_frame(self, frame_bgr):
        if self.done:
            return self._make_result(frame_bgr, None, None, [], False, None)

        face = detect_face(frame_bgr)
        fh, fw = frame_bgr.shape[:2]
        overlay_text = None
        hand_kpts = []
        hand_fingers = 0
        gate_ok = False

        # Step 1: FACE
        if self.step == S_FACE:
            overlay_text = "Look straight at the camera"
            if face is not None:
                x, y, w, h = face
                student, sim = match_face(frame_bgr[y:y + h, x:x + w], self.students)
                if student:
                    self._face_buf.append(student)
                else:
                    self._face_buf = []

                if len(self._face_buf) >= FACE_MATCH_FRAMES:
                    self.matched_student = self._face_buf[-1]
                    self.step = S_HAND_PROMPT
                    self._prompt_start = time.time()
                    self.on_update(S_HAND_PROMPT, f"✓ Face matched: {self.matched_student['name']}", "green")
            else:
                self._face_buf = []

        # Step 2: HAND_PROMPT (banner only)
        elif self.step == S_HAND_PROMPT:
            overlay_text = "✋  RAISE YOUR HAND  ✋"
            gate_ok = True
            if time.time() - (self._prompt_start or time.time()) >= 1.5:
                self.step = S_HAND
                self.on_update(S_HAND, "Raise your hand above your shoulder…", "accent")

        # Step 3: HAND
        elif self.step == S_HAND:
            overlay_text = "✋  KEEP HAND RAISED  ✋"
            hand_ok, hand_fingers, hand_kpts, _ = detect_hand(frame_bgr, face_box=face)

            if hand_ok and hand_fingers >= 3:
                self._hand_buf += 1
            else:
                self._hand_buf = max(0, self._hand_buf - 1)

            # gate_ok drives overlay colour — show green when hand is active
            gate_ok = hand_ok and hand_fingers >= 3

            if self._hand_buf >= HAND_CONFIRM_FRAMES:
                gate_ok = True
                self.step = S_LIVENESS
                self._liveness_start = time.time()
                self.on_update(S_LIVENESS, "✓ Hand confirmed — Blink once, then nod your head", "green")

        # Step 4: LIVENESS
        elif self.step == S_LIVENESS:
            overlay_text = "Blink once  →  then nod your head"
            elapsed = time.time() - (self._liveness_start or time.time())

            if face is not None and elapsed < NOD_WINDOW_S * 2:
                x, y, w, h = face
                gray_roi = cv2.cvtColor(frame_bgr[y:y + h, x:x + w], cv2.COLOR_BGR2GRAY)
                cy = y + h // 2

                # Blink detection
                eyes_closed = detect_blink(gray_roi)
                if eyes_closed:
                    self._blink_frames += 1
                else:
                    if self._blink_frames >= BLINK_CONSEC_FRAMES and not self._blink_ok:
                        self._blink_ok = True
                    self._blink_frames = 0

                # Nod detection
                if self._head_base_y is None:
                    self._head_base_y = cy
                if abs(cy - self._head_base_y) >= NOD_SHIFT_PX:
                    self._nodded = True

            if self._blink_ok and self._nodded:
                self.step = S_VOICE
                self.on_update(S_VOICE, "✓ Liveness confirmed — preparing voice check…", "green")
                threading.Timer(0.8, self._start_voice).start()

        # Step 5: VOICE
        elif self.step == S_VOICE:
            overlay_text = "🎤  Speak your phrase now" if not self._voice_busy else "🔴  Recording…"

        return self._make_result(frame_bgr, face, overlay_text, hand_kpts, gate_ok, hand_fingers)

    def _make_result(self, frame, face, overlay_text, hand_kpts, gate_ok, hand_fingers):
        return type('Result', (), {
            'frame': frame, 'step': self.step, 'face': face,
            'overlay_text': overlay_text, 'hand_kpts': hand_kpts,
            'hand_fingers': hand_fingers, 'gate_ok': gate_ok
        })()

    def _start_voice(self):
        if self.step != S_VOICE or self._voice_busy or self.done:
            return
        if not AUDIO_OK:
            self._finish(voice_skipped=True)
            return
        self._voice_busy = True
        threading.Thread(target=self._voice_thread, daemon=True).start()

    def _voice_thread(self):
        audio = record_voice(RECORD_SECONDS)
        matched, sim = match_voice(audio, self.matched_student)
        self._voice_busy = False
        if matched:
            self.on_update(S_VOICE, f"✓ Voice verified (similarity {sim:.2f})", "green")
            time.sleep(0.4)
            self._finish(voice_skipped=False)
        else:
            self._voice_retries += 1
            if self._voice_retries < VOICE_MAX_RETRIES:
                self.on_update(S_VOICE, f"Voice didn't match ({sim:.2f}) — retry {self._voice_retries}/{VOICE_MAX_RETRIES}", "warn")
                time.sleep(1.2)
                self._start_voice()
            else:
                self.on_update(S_VOICE, "Voice check failed — marking attendance (flagged)", "warn")
                time.sleep(0.6)
                self._finish(voice_skipped=True)

    def _finish(self, voice_skipped=False):
        if self.done:
            return
        self.done = True
        self._voice_skipped = voice_skipped
        self.step = S_DONE
        self.on_complete(self.matched_student, None, voice_skipped)