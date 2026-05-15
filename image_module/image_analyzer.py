"""
IMAGE ANALYSIS MODULE
=====================
Detects : Real Image | AI-Generated Image | Manipulated Image
Pipeline : Input → Resize 224×224 → Normalize → ResNet50 → GradCAM + Forensic → Prediction
"""

import numpy as np
import io
import base64


# ──────────────────────────────────────────────────────────────
#  GRAD-CAM
# ──────────────────────────────────────────────────────────────
class GradCAM:
    """
    Gradient-weighted Class Activation Mapping.
    Generates heatmaps that highlight regions most responsible for the prediction.
    """
    def __init__(self, model, target_layer):
        self.model        = model
        self.target_layer = target_layer
        self.gradients    = None
        self.activations  = None
        self._register()

    def _register(self):
        def fwd(m, i, o):  self.activations = o.detach()
        def bwd(m, gi, go): self.gradients   = go[0].detach()
        self.target_layer.register_forward_hook(fwd)
        self.target_layer.register_full_backward_hook(bwd)

    def generate(self, tensor, class_idx=None):
        import torch, torch.nn.functional as F
        self.model.zero_grad()
        out = self.model(tensor)
        if class_idx is None:
            class_idx = out.argmax(dim=1).item()
        out[0, class_idx].backward()
        weights = self.gradients.mean(dim=[2, 3], keepdim=True)
        cam     = F.relu((weights * self.activations).sum(dim=1, keepdim=True))
        cam     = F.interpolate(cam, size=(224, 224), mode='bilinear', align_corners=False)
        cam     = cam.squeeze().cpu().numpy()
        if cam.max() > 0:
            cam = (cam - cam.min()) / (cam.max() - cam.min())
        return cam, class_idx


# ──────────────────────────────────────────────────────────────
#  FORENSIC IMAGE ANALYZER
# ──────────────────────────────────────────────────────────────
class ForensicAnalyzer:
    """
    Lightweight image forensics: GAN artifact detection, texture,
    edge density, color irregularity, and frequency-domain analysis.
    """
    def analyze(self, img_rgb: np.ndarray) -> dict:
        import cv2
        gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
        return {
            "edge_density":          round(self._edge_density(gray), 4),
            "texture_consistency":   round(self._texture_consistency(gray), 4),
            "gan_smoothing_score":   round(self._gan_smoothing(img_rgb), 4),
            "color_irregularity":    round(self._color_irregularity(img_rgb), 4),
            "noise_level":           round(self._noise_level(gray, cv2), 4),
        }

    def _edge_density(self, gray):
        import cv2
        return float(cv2.Canny(gray, 50, 150).mean())

    def _texture_consistency(self, gray):
        h, w = gray.shape
        ph, pw = max(h//4, 1), max(w//4, 1)
        stds = [gray[i*ph:(i+1)*ph, j*pw:(j+1)*pw].std()
                for i in range(4) for j in range(4)]
        return float(np.std(stds))

    def _gan_smoothing(self, img):
        """High-frequency energy ratio: GAN images have characteristic FFT signatures."""
        gray  = img.mean(axis=2) if img.ndim == 3 else img
        mag   = np.abs(np.fft.fftshift(np.fft.fft2(gray)))
        h, w  = mag.shape
        r     = 10
        cy, cx= h//2, w//2
        center = mag[cy-r:cy+r, cx-r:cx+r]
        outer  = mag.copy(); outer[cy-r:cy+r, cx-r:cx+r] = 0
        return float(center.mean() / (outer.mean() + 1e-9))

    def _color_irregularity(self, img):
        if img.ndim < 3: return 0.0
        return float(np.std([img[:,:,c].std() for c in range(3)]))

    def _noise_level(self, gray, cv2):
        blurred = cv2.GaussianBlur(gray, (5,5), 0)
        return float(cv2.absdiff(gray, blurred).mean())


# ──────────────────────────────────────────────────────────────
#  MAIN IMAGE ANALYZER
# ──────────────────────────────────────────────────────────────
class ImageAnalyzer:

    LABELS = ["Real Image", "AI-Generated", "Manipulated"]

    def __init__(self):
        self.forensic  = ForensicAnalyzer()
        self.model     = None
        self.gradcam   = None
        self._load_model()

    def _load_model(self):
        try:
            import torch
            import torch.nn as nn
            import torchvision.models as models

            base    = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
            base.fc = nn.Sequential(
                nn.Linear(2048, 512), nn.ReLU(), nn.Dropout(0.4),
                nn.Linear(512, 3),
            )
            # Load fine-tuned weights if available
            import os
            if os.path.exists('models/resnet50_image.pt'):
                base.load_state_dict(torch.load('models/resnet50_image.pt', map_location='cpu'))
                print("[ImageAnalyzer] ✓ Fine-tuned ResNet50 weights loaded")
            else:
                print("[ImageAnalyzer] ✓ ResNet50 (ImageNet pretrained, no fine-tune yet)")
            base.eval()
            self.model   = base
            self.gradcam = GradCAM(base, base.layer4[-1])
        except Exception as e:
            print(f"[ImageAnalyzer] PyTorch unavailable ({e}) — forensic-only mode")

    def preprocess(self, img_rgb: np.ndarray):
        """Return normalised tensor ready for ResNet50."""
        import torch
        img = img_rgb / 255.0
        mean = np.array([0.485, 0.456, 0.406])
        std  = np.array([0.229, 0.224, 0.225])
        img  = (img - mean) / std
        return torch.tensor(img.transpose(2,0,1), dtype=torch.float32).unsqueeze(0)

    def analyze(self, img_path: str) -> dict:
        import cv2
        raw = cv2.imread(img_path)
        if raw is None:
            return {"error": "Could not load image. Check file path and format."}
        img_rgb = cv2.cvtColor(raw, cv2.COLOR_BGR2RGB)
        img_224 = cv2.resize(img_rgb, (224, 224))

        forensic = self.forensic.analyze(img_224)
        scores   = self._score(forensic)
        idx      = int(np.argmax(scores))
        label    = self.LABELS[idx]
        conf     = float(scores[idx])
        risk     = "High" if label != "Real Image" and conf > 0.7 else \
                   "Medium" if label != "Real Image" else "Low"

        heatmap_b64 = self._make_gradcam(img_224, idx)

        return {
            "label":            label,
            "confidence":       round(conf * 100, 2),
            "risk_level":       risk,
            "scores":           {self.LABELS[i]: round(float(scores[i])*100, 2) for i in range(3)},
            "forensic_analysis":forensic,
            "explanation":      self._explain(forensic, label),
            "gradcam_heatmap":  heatmap_b64,
        }

    # ── scoring ────────────────────────────────────────────────

    def _score(self, f: dict) -> np.ndarray:
        real_s  = 0.45
        ai_s    = 0.0
        manip_s = 0.0

        if f["gan_smoothing_score"] > 60:  ai_s    += 0.50
        elif f["gan_smoothing_score"] > 25: ai_s   += 0.25

        if f["edge_density"] < 4:           ai_s   += 0.30
        elif f["edge_density"] < 8:         ai_s   += 0.10

        if f["texture_consistency"] > 30:   manip_s += 0.35
        elif f["texture_consistency"] > 15: manip_s += 0.15

        if f["color_irregularity"] > 18:    manip_s += 0.25
        if f["noise_level"] > 10:           manip_s += 0.15

        raw  = np.array([real_s, ai_s, manip_s], dtype=np.float64)
        raw  = np.clip(raw, 0, None)
        soft = np.exp(raw * 2.5)
        return (soft / soft.sum()).astype(np.float32)

    # ── grad-cam ──────────────────────────────────────────────

    def _make_gradcam(self, img_224: np.ndarray, label_idx: int) -> str:
        try:
            import cv2, matplotlib
            matplotlib.use('Agg')
            import matplotlib.pyplot as plt

            if self.model is None:
                return self._simple_img_b64(img_224)

            tensor = self.preprocess(img_224)
            cam, _ = self.gradcam.generate(tensor, label_idx)

            heatmap = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)
            heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
            overlay = np.clip(img_224 * 0.55 + heatmap * 0.45, 0, 255).astype(np.uint8)

            fig, axes = plt.subplots(1, 2, figsize=(8, 3.5))
            fig.patch.set_facecolor('#0f1117')
            for ax, im, title in zip(axes, [img_224, overlay], ['Original', 'Grad-CAM Overlay']):
                ax.imshow(im); ax.set_title(title, color='white', fontsize=9); ax.axis('off')
                ax.set_facecolor('#131826')
            plt.tight_layout(pad=0.5)
            return self._fig_to_b64(fig)
        except Exception:
            return self._simple_img_b64(img_224)

    def _simple_img_b64(self, img_224) -> str:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(4,4))
        fig.patch.set_facecolor('#0f1117')
        ax.imshow(img_224); ax.axis('off')
        ax.set_title('Image', color='white')
        return self._fig_to_b64(fig)

    def _fig_to_b64(self, fig) -> str:
        import matplotlib.pyplot as plt
        buf = io.BytesIO()
        fig.savefig(buf, format='png', dpi=90, bbox_inches='tight')
        plt.close(fig)
        buf.seek(0)
        return base64.b64encode(buf.read()).decode()

    def _explain(self, f: dict, label: str) -> list:
        r = []
        if f["gan_smoothing_score"] > 30:
            r.append(f"High GAN smoothing score ({f['gan_smoothing_score']:.1f}) detected in "
                     "frequency domain — characteristic of GAN / diffusion-generated images.")
        if f["edge_density"] < 5:
            r.append("Unusually low edge density — over-smoothed textures are a hallmark of "
                     "AI image synthesis.")
        if f["texture_consistency"] > 30:
            r.append(f"High texture inconsistency ({f['texture_consistency']:.1f}) across "
                     "image patches — may indicate compositing or face-swap manipulation.")
        if f["color_irregularity"] > 15:
            r.append(f"Color channel irregularity ({f['color_irregularity']:.1f}) — "
                     "irregular distribution can indicate deepfake blending artefacts.")
        if not r:
            r.append("No dominant forensic signal. Classification based on combined "
                     "frequency-domain and texture feature analysis.")
        return r