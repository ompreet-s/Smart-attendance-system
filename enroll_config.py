"""
enroll_config.py — shared constants for enrollment system
Professional white + navy blue theme
"""

import os

# ── Directories ──────────────────────────────────────────
STUDENTS_DIR = "students"

# ── Voice settings ───────────────────────────────────────
VOICE_PHRASE = "My name is {name} and I am present"
SAMPLE_RATE = 44100
RECORD_SECONDS = 3

# ── Face capture settings ────────────────────────────────
FACE_HOLD_TIME = 1.0          # seconds face must hold pose
POSE_SHIFT_PX = 22             # pixels for head turn detection
SMILE_RATIO_MIN = 0.18        # lower/upper face brightness ratio

# ── 9 poses with gate type ───────────────────────────────
# gate types: centre, left, right, up, down, smile
POSES = [
    ("front_1",     "Look straight at the camera",         "centre"),
    ("front_2",     "Stay straight — second shot",         "centre"),
    ("left",        "Turn your head slightly LEFT",        "left"),
    ("right",       "Turn your head slightly RIGHT",       "right"),
    ("up",          "Tilt your head slightly UP",          "up"),
    ("down",        "Tilt your head slightly DOWN",        "down"),
    ("smile",       "Give a natural SMILE  ☺",             "smile"),
    ("left_extra",  "Turn a little more to the LEFT",      "left"),
    ("right_extra", "Turn a little more to the RIGHT",     "right"),
]

# ── Upload labels for UI ─────────────────────────────────
UPLOAD_LABELS = [
    ("front_1",     "Front — neutral"),
    ("front_2",     "Front — second"),
    ("left",        "Left turn"),
    ("right",       "Right turn"),
    ("up",          "Head up"),
    ("down",        "Head down"),
    ("smile",       "Smiling"),
    ("left_extra",  "Far left"),
    ("right_extra", "Far right"),
]

# ── Professional White + Navy Blue Palette ────────────────
BG = "#F0F4F8"
BG2 = "#FFFFFF"
BG3 = "#E8EEF4"
NAVY = "#1B2A4A"
ACCENT = "#1D4ED8"
ACCENT2 = "#16A34A"
WARN = "#DC2626"
TEXT = "#1B2A4A"
MUTED = "#64748B"
BORDER = "#CBD5E1"
STEP_DONE = "#16A34A"
STEP_TODO = "#CBD5E1"
STEP_NOW = "#1D4ED8"
HDR_BG = "#1B2A4A"
HDR_TEXT = "#FFFFFF"
BTN_FG = "#FFFFFF"

# ── Fonts ─────────────────────────────────────────────────
FONT_HEAD = ("Segoe UI", 16, "bold")
FONT_BODY = ("Segoe UI", 12)
FONT_SMALL = ("Segoe UI", 10)
FONT_BTN = ("Segoe UI", 11, "bold")
FONT_STEP = ("Segoe UI", 9, "bold")

# ── Window ────────────────────────────────────────────────
WIN_WIDTH = 920
WIN_HEIGHT = 680