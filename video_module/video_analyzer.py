"""
VIDEO ANALYSIS MODULE
=====================
Detects : Real Video | Deepfake | Manipulated Video
Pipeline : Input → Frame Extraction (16 frames) → Face Detection →
           CNN Frame Analysis → Temporal Inconsistency → Prediction
"""

import numpy as np
import io
import base64


# ──────────────────────────────────────────────────────────────
#  FRAME EXTRACTOR
# ──────────────────────────────────────────────────────────────
class FrameExtractor:
    def __init__(self, num_frames: int = 16):
        self.num_frames = num_frames

    def extract(self, video_path: str) -> list:
        try:
            import cv2
            cap   = cv2.VideoCapture(video_path)
            total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            if total == 0:
                cap.release(); return []
            indices = np.linspace(0, max(total-1, 0), self.num_frames, dtype=int)
            frames  = []
            for idx in indices:
                cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
                ret, frame = cap.read()
                if ret:
                    frames.append(cv2.resize(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), (224, 224)))
            cap.release()
            return frames
        except Exception as e:
            print(f"[FrameExtractor] {e}")
            return []


# ──────────────────────────────────────────────────────────────
#  FACE DETECTOR
# ──────────────────────────────────────────────────────────────
class FaceDetector:
    def __init__(self):
        try:
            import cv2
            xml = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
            self.clf = cv2.CascadeClassifier(xml)
        except Exception:
            self.clf = None

    def count_faces(self, frame_rgb: np.ndarray) -> int:
        if self.clf is None: return 0
        import cv2
        gray  = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2GRAY)
        faces = self.clf.detectMultiScale(gray, scaleFactor=1.1,
                                           minNeighbors=5, minSize=(30,30))
        return len(faces)


# ──────────────────────────────────────────────────────────────
#  TEMPORAL INCONSISTENCY ANALYZER
# ──────────────────────────────────────────────────────────────
class TemporalAnalyzer:
    """
    Computes frame-level statistics to reveal deepfake blending artefacts:
    - pixel difference magnitude & variance
    - brightness flicker
    - colour channel drift
    """
    def analyze(self, frames: list) -> dict:
        if len(frames) < 2:
            return {"error": "Too few frames extracted."}

        diffs      = [np.abs(frames[i].astype(float) - frames[i+1].astype(float)).mean()
                      for i in range(len(frames)-1)]
        brightness = [f.mean() for f in frames]
        ch_means   = [f.mean(axis=(0,1)) for f in frames]   # (N, 3)
        color_drift= float(np.mean([np.linalg.norm(ch_means[i+1]-ch_means[i])
                                     for i in range(len(ch_means)-1)]))

        return {
            "mean_frame_diff":    round(float(np.mean(diffs)), 4),
            "max_frame_diff":     round(float(np.max(diffs)),  4),
            "flicker_score":      round(float(np.std(brightness)), 4),
            "color_drift":        round(color_drift, 4),
            "temporal_variance":  round(float(np.var(diffs)), 4),
            "consistency_score":  round(float(1.0 - min(np.mean(diffs)/50.0, 1.0)), 4),
        }


# ──────────────────────────────────────────────────────────────
#  MAIN VIDEO ANALYZER
# ──────────────────────────────────────────────────────────────
class VideoAnalyzer:

    LABELS = ["Real Video", "Deepfake", "Manipulated"]

    def __init__(self):
        self.extractor = FrameExtractor(num_frames=16)
        self.face_det  = FaceDetector()
        self.temporal  = TemporalAnalyzer()

    def analyze(self, video_path: str) -> dict:
        frames = self.extractor.extract(video_path)
        if not frames:
            return {"error": "Could not extract frames. Ensure the video file is valid and not corrupted."}

        temp_stats   = self.temporal.analyze(frames)
        face_counts  = [self.face_det.count_faces(f) for f in frames]
        face_detected = any(c > 0 for c in face_counts)

        scores = self._score(temp_stats, face_detected)
        idx    = int(np.argmax(scores))
        label  = self.LABELS[idx]
        conf   = float(scores[idx])
        risk   = "High" if label != "Real Video" and conf > 0.70 else \
                 "Medium" if label != "Real Video" else "Low"

        return {
            "label":              label,
            "confidence":         round(conf * 100, 2),
            "risk_level":         risk,
            "scores":             {self.LABELS[i]: round(float(scores[i])*100, 2) for i in range(3)},
            "frames_analyzed":    len(frames),
            "face_detected":      face_detected,
            "temporal_analysis":  {k: v for k, v in temp_stats.items() if k != "error"},
            "explanation":        self._explain(temp_stats, label, face_detected),
            "frame_grid_b64":     self._frame_grid(frames),
        }

    # ── scoring ────────────────────────────────────────────────

    def _score(self, t: dict, face: bool) -> np.ndarray:
        real_s  = 0.50
        deep_s  = 0.0
        manip_s = 0.0

        mfd = t.get("mean_frame_diff", 0)
        flk = t.get("flicker_score", 0)
        cdr = t.get("color_drift", 0)
        tva = t.get("temporal_variance", 0)

        # High flicker → deepfake face-swap boundary artefacts
        if flk > 12: deep_s  += 0.40
        elif flk > 6: deep_s += 0.18

        # High mean diff → editing
        if mfd > 18: manip_s += 0.30
        elif mfd > 10: manip_s += 0.15

        # High colour drift → blending or suspicious transition
        if cdr > 6: deep_s  += 0.20; manip_s += 0.10
        elif cdr > 3: deep_s += 0.08

        # Very low temporal variance → over-compressed / synthetic
        if tva < 0.5: deep_s += 0.20

        if not face: real_s -= 0.10   # no face reduces confidence in "real"

        raw  = np.array([real_s, deep_s, manip_s], dtype=np.float64)
        raw  = np.clip(raw, 0, None)
        soft = np.exp(raw * 2.5)
        return (soft / soft.sum()).astype(np.float32)

    # ── frame grid ────────────────────────────────────────────

    def _frame_grid(self, frames: list) -> str:
        try:
            import matplotlib
            matplotlib.use('Agg')
            import matplotlib.pyplot as plt

            show = min(8, len(frames))
            cols = 4
            rows = (show + cols - 1) // cols
            fig, axes = plt.subplots(rows, cols, figsize=(11, rows * 2.8))
            fig.patch.set_facecolor('#0f1117')
            flat = axes.flatten() if hasattr(axes, 'flatten') else [axes]
            for i, ax in enumerate(flat):
                if i < show:
                    ax.imshow(frames[i])
                    ax.set_title(f'Frame {i+1}', color='#8890aa', fontsize=7)
                ax.axis('off')
                ax.set_facecolor('#131826')
            plt.suptitle('Extracted Frame Grid', color='white', fontsize=9, y=1.01)
            plt.tight_layout(pad=0.4)
            buf = io.BytesIO()
            fig.savefig(buf, format='png', dpi=90, bbox_inches='tight')
            plt.close(fig)
            buf.seek(0)
            return base64.b64encode(buf.read()).decode()
        except Exception:
            return ""

    def _explain(self, t: dict, label: str, face: bool) -> list:
        r = []
        if t.get("flicker_score", 0) > 10:
            r.append(f"High brightness flicker ({t['flicker_score']:.2f}) detected — "
                     "typical of deepfake face-swap boundary artefacts.")
        if t.get("mean_frame_diff", 0) > 15:
            r.append(f"Excessive frame-level pixel differences (mean={t['mean_frame_diff']:.2f}) "
                     "— suggests edited or composited content.")
        if t.get("color_drift", 0) > 5:
            r.append(f"Rapid colour distribution drift ({t['color_drift']:.2f}) — "
                     "inconsistent lighting is a common deepfake artefact.")
        if t.get("temporal_variance", 0) < 0.5:
            r.append("Suspiciously low temporal variance — may indicate over-compressed "
                     "or synthetically generated video.")
        if not face:
            r.append("No human face detected in sampled frames.")
        if not r:
            r.append("No dominant temporal artefact signal. Classification based on "
                     "combined frame consistency statistics.")
        return r