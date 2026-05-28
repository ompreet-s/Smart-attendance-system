"""
attend_detectors.py — pure detection functions for attendance
"""

import cv2
import numpy as np
import os
import json
import csv
import time
from datetime import datetime
from attend_config import (
    STUDENTS_DIR, ATTENDANCE_DIR, FACE_HIST_BINS, FACE_MATCH_THRESH,
    HAND_MIN_AREA, HAND_UPPER_FRAC, BLINK_CONSEC_FRAMES,
    SAMPLE_RATE, VOICE_MATCH_THRESH
)

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_eye.xml")

try:
    import sounddevice as sd
    from scipy.io import wavfile
    AUDIO_OK = True
except Exception:
    sd = None
    wavfile = None
    AUDIO_OK = False


def detect_face(frame_bgr):
    gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.1, 5, minSize=(80, 80))
    if not len(faces):
        return None
    return max(faces, key=lambda f: f[2] * f[3])


def face_histogram(img_bgr):
    if img_bgr is None:
        return None
    img = cv2.resize(img_bgr, (64, 64))
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.1, 4, minSize=(20, 20))
    if len(faces):
        x, y, w, h = max(faces, key=lambda f: f[2] * f[3])
        crop = img[y:y + h, x:x + w]
        if crop.size:
            img = cv2.resize(crop, (64, 64))
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    h_ch = cv2.calcHist([hsv], [0], None, [FACE_HIST_BINS], [0, 180]).flatten()
    s_ch = cv2.calcHist([hsv], [1], None, [FACE_HIST_BINS], [0, 256]).flatten()
    gx = cv2.Sobel(gray.astype(np.float32), cv2.CV_32F, 1, 0)
    gy = cv2.Sobel(gray.astype(np.float32), cv2.CV_32F, 0, 1)
    mag = np.sqrt(gx ** 2 + gy ** 2).flatten()
    t_ch = np.histogram(mag, bins=64, range=(0, 300))[0].astype(np.float32)
    feat = np.concatenate([h_ch, s_ch, t_ch])
    n = np.linalg.norm(feat)
    return feat / n if n > 0 else feat


def match_face(frame_bgr, students):
    hist = face_histogram(frame_bgr)
    if hist is None:
        return None, 0.0
    best_s, best_sim = None, 0.0
    for s in students:
        for sh in s["face_hists"]:
            sim = float(np.dot(hist, sh) / (np.linalg.norm(hist) * np.linalg.norm(sh) + 1e-9))
            if sim > best_sim:
                best_sim, best_s = sim, s
    return (best_s, best_sim) if best_sim >= FACE_MATCH_THRESH else (None, best_sim)


def extract_voice_features(audio_int16):
    signal = audio_int16.astype(np.float32).flatten()
    signal /= (np.max(np.abs(signal)) + 1e-6)
    frame_len = int(SAMPLE_RATE * 0.02)
    frames = [signal[i:i + frame_len]
              for i in range(0, len(signal) - frame_len, frame_len // 2)
              if len(signal[i:i + frame_len]) == frame_len]
    if not frames:
        return np.zeros(26, dtype=np.float32)
    window = np.hanning(frame_len)
    avg_spec = np.mean([np.abs(np.fft.rfft(f * window)) for f in frames], axis=0)
    n_bins = len(avg_spec)
    edges = np.logspace(np.log10(1), np.log10(n_bins), 27).astype(int)
    edges = np.clip(edges, 0, n_bins - 1)
    bands = np.array([avg_spec[edges[i]:edges[i + 1]].mean()
                      for i in range(26)], dtype=np.float32)
    n = np.linalg.norm(bands)
    return bands / n if n > 0 else bands


def record_voice(seconds=None):
    if not AUDIO_OK:
        return None
    secs = seconds or 3
    try:
        rec = sd.rec(int(secs * SAMPLE_RATE), samplerate=SAMPLE_RATE, channels=1, dtype="int16")
        sd.wait()
        return rec
    except Exception:
        return None


def match_voice(audio_int16, student):
    stored = student.get("voice_feat")
    if stored is None:
        return True, 1.0
    if audio_int16 is None:
        return False, 0.0
    feat = extract_voice_features(audio_int16)
    sim = float(np.dot(feat, stored) / (np.linalg.norm(feat) * np.linalg.norm(stored) + 1e-9))
    return sim >= VOICE_MATCH_THRESH, sim


def detect_blink(face_roi_gray):
    """
    Detect blink by checking if eyes are closed.
    More tolerant for low-quality cameras - returns True if no eyes detected
    OR if fewer than 2 eyes detected (more sensitive)
    """
    # Try different minSize values for better detection
    eyes = eye_cascade.detectMultiScale(face_roi_gray, 1.1, 4, minSize=(15, 15))
    # Also try with larger minSize for better accuracy
    if len(eyes) == 0:
        eyes = eye_cascade.detectMultiScale(face_roi_gray, 1.1, 4, minSize=(25, 25))
    
    # Blink is detected when NO eyes are found in the face region
    return len(eyes) == 0


def detect_hand(frame_bgr, face_box=None):
    """
    Detect hand with skin color segmentation and convexity defects.
    Returns (hand_detected, finger_count, keypoints, skin_mask)
    """
    fh, fw = frame_bgr.shape[:2]

    # Convert to different color spaces for better skin detection
    ycrcb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2YCrCb)
    hsv   = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)

    # ── Skin colour ranges ─────────────────────────────────────────────
    # YCrCb — widened Cb range (77→60) to capture darker/Indian skin tones
    m1 = cv2.inRange(ycrcb, np.array([0, 133, 60]), np.array([255, 180, 135]))

    # HSV — two hue bands covering light to dark/olive skin
    m2 = cv2.inRange(hsv, np.array([0,  15, 50]), np.array([25, 255, 255]))
    m3 = cv2.inRange(hsv, np.array([160, 15, 50]), np.array([180, 255, 255]))

    # Combine all masks
    skin = cv2.bitwise_or(m1, cv2.bitwise_or(m2, m3))

    # Morphological operations to clean up noise
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    skin = cv2.morphologyEx(skin, cv2.MORPH_CLOSE, kernel, iterations=2)
    skin = cv2.morphologyEx(skin, cv2.MORPH_OPEN,  kernel, iterations=1)
    skin = cv2.GaussianBlur(skin, (5, 5), 0)
    _, skin = cv2.threshold(skin, 127, 255, cv2.THRESH_BINARY)

    # Restrict to upper portion of frame (hand should be above waist)
    upper_limit = int(fh * HAND_UPPER_FRAC)
    skin[upper_limit:, :] = 0

    # ── Exclude the face region from skin mask ─────────────────────────
    # Without this, the face itself is the biggest skin blob and gets
    # picked as the "hand" — mask it out before finding contours.
    if face_box is not None:
        fx, fy, fw_face, fh_face = face_box
        pad = 20  # small padding so we don't clip jaw/ears
        x1 = max(0, fx - pad)
        y1 = max(0, fy - pad)
        x2 = min(fw - 1, fx + fw_face + pad)
        y2 = min(fh - 1, fy + fh_face + pad)
        skin[y1:y2, x1:x2] = 0

    # Find contours
    contours, _ = cv2.findContours(skin, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return False, 0, [], skin

    # Get the largest contour (assumed to be the hand)
    hand = max(contours, key=cv2.contourArea)
    area = cv2.contourArea(hand)
    if area < HAND_MIN_AREA:
        return False, 0, [], skin

    # Get centroid
    M = cv2.moments(hand)
    if M["m00"] == 0:
        return False, 0, [], skin
    cx, cy = int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"])

    # BUG FIX: original check `cy > fy + fhh` rejected the hand when it
    # was BELOW the face bottom — but a raised hand is ABOVE the face top
    # (smaller y value).  Correct guard: reject if hand centroid is below
    # the face's bottom edge (i.e. hand is near the body, not raised).
    if face_box is not None:
        _, fy, _, fhh = face_box
        if cy > fy + fhh + 40:   # allow a little slack below chin
            return False, 0, [], skin
    
    # Count fingers using convexity defects
    hull_idx = cv2.convexHull(hand, returnPoints=False)
    fingers = 0
    defect_points = []
    
    if hull_idx is not None and len(hull_idx) > 3:
        defects = cv2.convexityDefects(hand, hull_idx)
        if defects is not None:
            for i in range(defects.shape[0]):
                s, e, f, d = defects[i, 0]
                if d / 256.0 > 18:  # Depth threshold for finger valleys
                    fingers += 1
                    # Store defect points for drawing
                    far_point = tuple(hand[f][0])
                    defect_points.append(far_point)
    
    # Add 1 because convexity defects count = fingers - 1 for open palm
    fingers = min(fingers + 1, 5)
    
    # Get keypoints for drawing (centroid + hull points)
    hull_pts = cv2.convexHull(hand)
    keypoints = [(cx, cy)]  # Start with centroid
    
    # Add hull points (fingertip approximations)
    if len(hull_pts) > 0:
        hull_arr = hull_pts[:, 0, :]
        # Sort by y to get topmost points (likely fingertips)
        top_pts = hull_arr[hull_arr[:, 1].argsort()][:8]
        for pt in top_pts:
            keypoints.append((int(pt[0]), int(pt[1])))
    
    # Add defect points (valleys between fingers)
    keypoints.extend(defect_points)
    
    return True, fingers, keypoints, skin


def draw_hand_overlay(frame, keypoints, fingers_up, gesture_ok):
    """
    Draw hand overlay with red circles and connecting lines.
    Style matches the reference image.
    """
    # Colors: Red/pink for detection, Green for confirmed
    if gesture_ok:
        dot_color = (0, 200, 0)      # Green for confirmed
        line_color = (0, 180, 0)     # Darker green for lines
        outer_color = (100, 255, 100)  # Light green outer ring
    else:
        dot_color = (0, 0, 255)      # Red for detecting
        line_color = (0, 0, 200)     # Darker red for lines
        outer_color = (100, 100, 255) # Light red outer ring
    
    # Draw lines between consecutive keypoints
    if len(keypoints) >= 2:
        for i in range(len(keypoints) - 1):
            cv2.line(frame, keypoints[i], keypoints[i + 1], line_color, 2)
    
    # Draw circles at each keypoint
    for pt in keypoints:
        # Outer glow ring
        cv2.circle(frame, pt, 15, outer_color, 2)
        # Inner circle
        cv2.circle(frame, pt, 8, dot_color, -1)
        # Center highlight
        cv2.circle(frame, pt, 3, (255, 255, 255), -1)
    
    # Display finger count near the centroid
    if keypoints:
        cx, cy = keypoints[0]
        label = f"{fingers_up} finger{'s' if fingers_up != 1 else ''}"
        # Background for text
        (text_w, text_h), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
        cv2.rectangle(frame, (cx - text_w - 5, cy - text_h - 30), 
                      (cx + 5, cy - 20), (0, 0, 0), -1)
        cv2.putText(frame, label, (cx - text_w - 3, cy - 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    
    return frame


def load_students():
    students = []
    if not os.path.isdir(STUDENTS_DIR):
        return students
    for entry in os.listdir(STUDENTS_DIR):
        folder = os.path.join(STUDENTS_DIR, entry)
        meta_f = os.path.join(folder, "metadata.json")
        if not os.path.isdir(folder) or not os.path.isfile(meta_f):
            continue
        with open(meta_f) as f:
            meta = json.load(f)
        hists = []
        for fname in meta.get("photos", []):
            img = cv2.imread(os.path.join(folder, fname))
            h = face_histogram(img)
            if h is not None:
                hists.append(h)
        feat_path = os.path.join(folder, "voice_features.npy")
        if os.path.isfile(feat_path):
            voice_feat = np.load(feat_path).astype(np.float32)
        else:
            voice_feat = None
        students.append({
            "name": meta["name"], "roll": meta["roll_number"],
            "folder": folder, "face_hists": hists, "voice_feat": voice_feat,
        })
    return students


def log_attendance(student, photo_path="", voice_skipped=False):
    os.makedirs(ATTENDANCE_DIR, exist_ok=True)
    today = datetime.now().strftime("%Y-%m-%d")
    csv_path = os.path.join(ATTENDANCE_DIR, f"attendance_{today}.csv")
    is_new = not os.path.isfile(csv_path)
    now = datetime.now()
    with open(csv_path, "a", newline="") as f:
        w = csv.writer(f)
        if is_new:
            w.writerow(["Roll", "Name", "Date", "Time", "Photo", "VoiceSkipped"])
        w.writerow([student["roll"], student["name"],
                    now.strftime("%Y-%m-%d"), now.strftime("%H:%M:%S"),
                    photo_path, "yes" if voice_skipped else "no"])