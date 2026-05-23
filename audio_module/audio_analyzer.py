"""
AUDIO / VOICE ANALYSIS MODULE
==============================
Detects : Real Voice | AI-Generated Voice | Cloned Voice
Pipeline : Audio → Librosa → MFCC + Spectral Features → CNN Classifier → Prediction + XAI
"""

## Standard libraries 
# No external dependencies at the top level (librosa, torch imported inside classes to allow graceful degradation)


import base64
import io
import numpy as np



# ──────────────────────────────────────────────────────────────
#  AUDIO FEATURE EXTRACTOR
# ──────────────────────────────────────────────────────────────
class AudioFeatureExtractor:
    """
    Extracts MFCC, pitch (F0), spectral contrast, ZCR, chroma, and RMS features.
    These are the core inputs for the audio classifier and XAI explanations.
    """
    def __init__(self, sr: int = 22050, n_mfcc: int = 40):
        self.sr     = sr
        self.n_mfcc = n_mfcc

    def extract(self, path: str) -> dict:
        try:
            import librosa
            y, sr = librosa.load(path, sr=self.sr, mono=True)
        except Exception as e:
            return {"error": str(e)}

        feats = {}

        # ── MFCC (40 coefficients) ────────────────────────────
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=self.n_mfcc)
        feats["mfcc_mean"]  = mfcc.mean(axis=1).tolist()
        feats["mfcc_std"]   = mfcc.std(axis=1).tolist()
        feats["mfcc_delta"] = librosa.feature.delta(mfcc).mean(axis=1).tolist()

        # ── Pitch / F0 ────────────────────────────────────────
        try:
            f0, voiced, _ = librosa.pyin(y,
                                          fmin=librosa.note_to_hz('C2'),
                                          fmax=librosa.note_to_hz('C7'))
            valid = f0[~np.isnan(f0)] if f0 is not None else np.array([0.0])
        except Exception:
            valid = np.array([0.0])
            voiced = np.array([0.5])

        feats["pitch_mean"]     = float(valid.mean())
        feats["pitch_variance"] = float(valid.var())
        feats["voiced_ratio"]   = float(voiced.mean()) if voiced is not None else 0.5

        # ── Spectral contrast ─────────────────────────────────
        sc = librosa.feature.spectral_contrast(y=y, sr=sr)
        feats["spectral_contrast_mean"] = float(sc.mean())
        feats["spectral_contrast_std"]  = float(sc.std())

        # ── Zero Crossing Rate ────────────────────────────────
        zcr = librosa.feature.zero_crossing_rate(y)
        feats["zcr_mean"] = float(zcr.mean())
        feats["zcr_std"]  = float(zcr.std())

        # ── RMS energy ───────────────────────────────────────
        rms = librosa.feature.rms(y=y)
        feats["rms_mean"] = float(rms.mean())
        feats["rms_std"]  = float(rms.std())

        # ── Chroma ───────────────────────────────────────────
        chroma = librosa.feature.chroma_stft(y=y, sr=sr)
        feats["chroma_mean"] = float(chroma.mean())

        # ── Duration ─────────────────────────────────────────
        feats["duration_sec"] = float(librosa.get_duration(y=y, sr=sr))

        # ── Raw waveform (used for visualisation only) ────────
        feats["_y"]  = y
        feats["_sr"] = sr
        return feats

    def flat_vector(self, feats: dict) -> np.ndarray:
        vec = []
        for k, v in feats.items():
            if k.startswith('_'): continue
            if isinstance(v, list): vec.extend(v)
            elif isinstance(v, float): vec.append(v)
        return np.array(vec, dtype=np.float32)

# ──────────────────────────────────────────────────────────────
#  CNN AUDIO CLASSIFIER  (PyTorch)
# ──────────────────────────────────────────────────────────────
try:
    import torch
    import torch.nn as nn

    class AudioCNN(nn.Module):
        """
        2-D CNN over MFCC spectrogram image.
        Input shape: (batch, 1, n_mfcc, time_frames)
        """
        def __init__(self, n_mfcc=40, num_classes=3):
            super().__init__()
            self.conv = nn.Sequential(
                nn.Conv2d(1, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(), nn.MaxPool2d(2),
                nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(), nn.MaxPool2d(2),
                nn.AdaptiveAvgPool2d((4, 4)),
            )
            self.fc = nn.Sequential(
                nn.Flatten(),
                nn.Linear(64*4*4, 128), nn.ReLU(), nn.Dropout(0.3),
                nn.Linear(128, num_classes),
            )
        def forward(self, x): return self.fc(self.conv(x))

    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    AudioCNN = None


# ──────────────────────────────────────────────────────────────
#  MAIN AUDIO ANALYZER
# ──────────────────────────────────────────────────────────────
class AudioAnalyzer:

    LABELS = ["Real Voice", "AI-Generated Voice", "Cloned Voice"]

    def __init__(self):
        self.extractor = AudioFeatureExtractor()
        self._cnn      = None
        self._load_model()

    def _load_model(self):
        try:
            import torch, os
            if AudioCNN and os.path.exists('models/audio_cnn.pt'):
                self._cnn = AudioCNN()
                self._cnn.load_state_dict(torch.load('models/audio_cnn.pt', map_location='cpu'))
                self._cnn.eval()
                print("[AudioAnalyzer] ✓ AudioCNN weights loaded")
            else:
                print("[AudioAnalyzer] ✓ Running in rule-based mode (no trained weights yet)")
        except Exception as e:
            print(f"[AudioAnalyzer] Could not load AudioCNN ({e})")

    def analyze(self, path: str) -> dict:
        feats = self.extractor.extract(path)
        if "error" in feats:
            return feats

        scores = self._rule_based_score(feats)
        idx    = int(np.argmax(scores))
        label  = self.LABELS[idx]
        conf   = float(scores[idx])
        risk   = "High" if label != "Real Voice" and conf > 0.70 else \
                 "Medium" if label != "Real Voice" else "Low"

        return {
            "label":       label,
            "confidence":  round(conf * 100, 2),
            "risk_level":  risk,
            "scores":      {self.LABELS[i]: round(float(scores[i])*100, 2) for i in range(3)},
            "audio_features": {
                "duration_sec":           round(feats["duration_sec"], 2),
                "pitch_mean_hz":          round(feats["pitch_mean"], 2),
                "pitch_variance":         round(feats["pitch_variance"], 4),
                "voiced_ratio":           round(feats["voiced_ratio"], 4),
                "zcr_mean":               round(feats["zcr_mean"], 6),
                "spectral_contrast_mean": round(feats["spectral_contrast_mean"], 4),
                "rms_mean":               round(feats["rms_mean"], 6),
                "rms_std":                round(feats["rms_std"], 6),
            },
            "explanation":      self._explain(feats, label),
            "spectrogram_b64":  self._mel_spectrogram(feats),
            "mfcc_plot_b64":    self._mfcc_plot(feats),
        }

    # ── scoring ────────────────────────────────────────────────

    def _rule_based_score(self, f: dict) -> np.ndarray:
        real_s  = 0.40
        ai_s    = 0.0
        clone_s = 0.0

        # Very low pitch variance → TTS / synthesised
        pv = f["pitch_variance"]
        if   pv < 300:  ai_s    += 0.50
        elif pv < 1000: ai_s    += 0.20

        # High voiced ratio → synthesised (no breath pauses)
        vr = f["voiced_ratio"]
        if vr > 0.93: ai_s += 0.25
        elif vr > 0.88: ai_s += 0.10

        # Very stable RMS → synthetic (no natural loudness variation)
        if f["rms_std"] < 0.008: ai_s += 0.20
        elif f["rms_std"] < 0.020: ai_s += 0.08

        # Low spectral contrast → synthesised harmonics
        sc = f["spectral_contrast_mean"]
        if sc < 15: ai_s += 0.20
        elif sc < 25: ai_s += 0.08

        # Cloning heuristic: moderate pitch var but off ZCR pattern
        if 300 < pv < 1200 and f["zcr_mean"] < 0.04:
            clone_s += 0.30

        raw  = np.array([real_s, ai_s, clone_s], dtype=np.float64)
        raw  = np.clip(raw, 0, None)
        soft = np.exp(raw * 3.0)
        return (soft / soft.sum()).astype(np.float32)

    # ── visualisations ─────────────────────────────────────────

    def _mel_spectrogram(self, feats: dict) -> str:
        try:
            import librosa, librosa.display, matplotlib
            matplotlib.use('Agg')
            import matplotlib.pyplot as plt

            y, sr = feats["_y"], feats["_sr"]
            mel   = librosa.power_to_db(librosa.feature.melspectrogram(y=y, sr=sr), ref=np.max)
            fig, ax = plt.subplots(figsize=(8, 3))
            fig.patch.set_facecolor('#0f1117')
            ax.set_facecolor('#131826')
            img = librosa.display.specshow(mel, x_axis='time', y_axis='mel', sr=sr, ax=ax, cmap='magma')
            fig.colorbar(img, ax=ax, format='%+2.0f dB')
            ax.set_title('Mel Spectrogram', color='white', fontsize=9)
            ax.tick_params(colors='#8890aa')
            for sp in ax.spines.values(): sp.set_edgecolor('#1e2438')
            plt.tight_layout()
            return self._fig_b64(fig)
        except Exception:
            return ""

    def _mfcc_plot(self, feats: dict) -> str:
        try:
            import librosa, librosa.display, matplotlib
            matplotlib.use('Agg')
            import matplotlib.pyplot as plt

            y, sr = feats["_y"], feats["_sr"]
            mfcc  = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40)
            fig, ax = plt.subplots(figsize=(8, 3))
            fig.patch.set_facecolor('#0f1117')
            ax.set_facecolor('#131826')
            img = librosa.display.specshow(mfcc, x_axis='time', ax=ax, cmap='coolwarm')
            fig.colorbar(img, ax=ax)
            ax.set_title('MFCC Coefficients', color='white', fontsize=9)
            ax.set_ylabel('MFCC #', color='#8890aa')
            ax.tick_params(colors='#8890aa')
            for sp in ax.spines.values(): sp.set_edgecolor('#1e2438')
            plt.tight_layout()
            return self._fig_b64(fig)
        except Exception:
            return ""

    def _fig_b64(self, fig) -> str:
        import matplotlib.pyplot as plt
        buf = io.BytesIO()
        fig.savefig(buf, format='png', dpi=90, bbox_inches='tight')
        plt.close(fig)
        buf.seek(0)
        return base64.b64encode(buf.read()).decode()

    def _explain(self, f: dict, label: str) -> list:
        r = []
        if f["pitch_variance"] < 300:
            r.append(f"Pitch variance is extremely low ({f['pitch_variance']:.1f}) — "
                     "natural speech contains much greater melodic variation.")
        if f["voiced_ratio"] > 0.92:
            r.append(f"Voiced frame ratio is {f['voiced_ratio']:.3f} — "
                     "TTS systems often produce near-continuous voicing with no breath pauses.")
        if f["rms_std"] < 0.010:
            r.append("RMS energy is suspiciously stable — "
                     "human speech naturally fluctuates in loudness.")
        if f["spectral_contrast_mean"] < 20:
            r.append(f"Spectral contrast is low ({f['spectral_contrast_mean']:.2f}) — "
                     "may indicate synthesised harmonic structure.")
        if not r:
            r.append("No single dominant signal. Prediction based on combined "
                     "MFCC + pitch + energy feature statistics.")
        return r
    

### audio detection do completed finish lets gggoooooo

# Taarak Mehta Ka Ooltah Chashmah

# Article
# Talk
# Read
# View source
# View history

# Tools
# Appearance hide
# Text

# Small

# Standard

# Large
# Width

# Standard

# Wide
# Color

# Automatic

# Light

# Dark
# Page semi-protected
# From Wikipedia, the free encyclopedia
# (Redirected from TMKOC)
# Taarak Mehta Ka Ooltah Chashmah
# Taarak Mehta Ka Ooltah Chashmah
# Also known as	TMKOC
# Genre	Sitcom
# Comedy
# Created by	Asit Kumarr Modi
# Based on	
# Duniya Ne Undha Chashmah
# by Tarak Mehta
# Directed by	
# Dharmessh Mehta
# Abhishek Sharma
# Dheeraj Palshetkar
# Harshad Joshi
# Malav Suresh Rajda
# Starring	See below
# Narrated by	Shailesh Lodha (2008–2022)
# Sachin Shroff (2022–present)
# Opening theme	Taarak Mehta Ka Ooltah Chashmah
# Composer	Sunil Patni
# Country of origin	India
# Original language	Hindi
# No. of seasons	1
# No. of episodes	4,712
# Production
# Producers	
# Asit Kumarr Modi
# Neela Asit Modi
# Camera setup	Multi-camera
# Running time	19–22 minutes
# Production company	Neela Film Productions
# Original release
# Network	Sony SAB
# Release	28 July 2008 –
# present
# Related
# Taarak Mehta Kka Chhota Chashmah
# Taarak Mehta Ka Ooltah Chashmah (transl. "Taarak Mehta's Inverted Spectacles"), often abbreviated as TMKOC, is an Indian sitcom and comedy based on the weekly column Duniya Ne Undha Chasma by Tarak Mehta for the magazine Chitralekha. Produced by Asit Kumarr Modi, it is one of the longest-running television series in India. The series premiered on 28 July 2008 on Sony SAB and is also digitally available on SonyLIV.[1]

# Plot
# The series is set in Mumbai and follows the lives of the residents of Gokuldham Co-operative Housing Society, a diverse community with people from different cultural and regional backgrounds.

# Most storylines focus on an individual, a family, or sometimes the entire society as they encounter and resolve various problems. The characters frequently support one another and celebrate festivals together, highlighting their close-knit bond.

# A recurring theme centers on Jethalal Champaklal Gada, who often finds himself in comical situations and troubles, with brief moments of relief before new challenges arise.

# Cast and characters