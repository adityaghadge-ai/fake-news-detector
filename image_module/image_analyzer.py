"""
IMAGE ANALYSIS MODULE
=====================
Detects : Real Image | AI-Generated Image

Pipeline :
Input Image
→ MobileNetV2 CNN
→ Confidence Estimation
→ Forensic Feature Extraction
→ Grad-CAM Explainability
→ Explainable Analysis Report
"""

import io
import cv2
import base64
import numpy as np

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt


# =========================================================
# GRAD-CAM
# =========================================================

class GradCAM:

    def __init__(self, model, target_layer):

        self.model = model
        self.target_layer = target_layer

        self.gradients = None
        self.activations = None

        self.register_hooks()

    def register_hooks(self):

        def forward_hook(module, input, output):

            self.activations = output.detach()

        def backward_hook(module, grad_input, grad_output):

            self.gradients = grad_output[0].detach()

        self.target_layer.register_forward_hook(
            forward_hook
        )

        self.target_layer.register_full_backward_hook(
            backward_hook
        )

    def generate(self, tensor, class_idx=None):

        import torch
        import torch.nn.functional as F

        self.model.zero_grad()

        output = self.model(tensor)

        if class_idx is None:

            class_idx = output.argmax(dim=1).item()

        output[0, class_idx].backward()

        weights = self.gradients.mean(
            dim=[2,3],
            keepdim=True
        )

        cam = (weights * self.activations).sum(dim=1)

        cam = F.relu(cam)

        cam = F.interpolate(
            cam.unsqueeze(1),
            size=(224,224),
            mode='bilinear',
            align_corners=False
        )

        cam = cam.squeeze().cpu().numpy()

        cam = (cam - cam.min()) / (
            cam.max() + 1e-8
        )

        return cam, class_idx


# =========================================================
# FORENSIC ANALYSIS
# =========================================================

class ForensicAnalyzer:

    def analyze(self, img_rgb):

        gray = cv2.cvtColor(
            img_rgb,
            cv2.COLOR_RGB2GRAY
        )

        return {

            "edge_density":
            round(self.edge_density(gray), 4),

            "texture_consistency":
            round(self.texture_consistency(gray), 4),

            "gan_artifact_score":
            round(self.gan_score(img_rgb), 4),

            "noise_level":
            round(self.noise_level(gray), 4)
        }

    # Better edge thresholds
    def edge_density(self, gray):

        edges = cv2.Canny(
            gray,
            30,
            100
        )

        density = np.sum(edges > 0) / edges.size

        return float(density * 100)

    def texture_consistency(self, gray):

        return float(gray.std())

    def gan_score(self, img):

        gray = img.mean(axis=2)

        fft = np.abs(
            np.fft.fftshift(
                np.fft.fft2(gray)
            )
        )

        return float(
            fft.mean() / (fft.std() + 1e-8)
        )

    def noise_level(self, gray):

        blur = cv2.GaussianBlur(
            gray,
            (5,5),
            0
        )

        return float(
            cv2.absdiff(gray, blur).mean()
        )


# =========================================================
# IMAGE ANALYZER
# =========================================================

class ImageAnalyzer:

    LABELS = [
        "AI-Generated",
        "Real Image"
    ]

    def __init__(self):

        self.model = None

        self.gradcam = None

        self.forensic = ForensicAnalyzer()

        self.load_model()

    # =====================================================
    # LOAD MODEL
    # =====================================================

    def load_model(self):

        try:

            import torch
            import torch.nn as nn
            import torchvision.models as models

            model = models.mobilenet_v2(
                weights=models.MobileNet_V2_Weights.DEFAULT
            )

            model.classifier = nn.Sequential(

                nn.Dropout(0.3),

                nn.Linear(
                    model.last_channel,
                    2
                )
            )

            model.load_state_dict(

                torch.load(
                    "models/mobilenet_image.pt",
                    map_location="cpu"
                )
            )

            model.eval()

            self.model = model

            self.gradcam = GradCAM(

                model,

                model.features[-1]
            )

            print("MobileNet weights loaded")

        except Exception as e:

            print("Error loading model:", e)

    # =====================================================
    # PREPROCESS IMAGE
    # =====================================================

    def preprocess(self, img_rgb):

        import torch

        img = img_rgb / 255.0

        mean = np.array([
            0.485,
            0.456,
            0.406
        ])

        std = np.array([
            0.229,
            0.224,
            0.225
        ])

        img = (img - mean) / std

        tensor = torch.tensor(

            img.transpose(2,0,1),

            dtype=torch.float32

        ).unsqueeze(0)

        return tensor

    # =====================================================
    # MAIN ANALYSIS
    # =====================================================

    def analyze(self, image_path):

        import torch

        raw = cv2.imread(image_path)

        if raw is None:

            return {
                "error":"Image loading failed"
            }

        img_rgb = cv2.cvtColor(
            raw,
            cv2.COLOR_BGR2RGB
        )

        img_224 = cv2.resize(
            img_rgb,
            (224,224)
        )

        # =================================================
        # FORENSIC ANALYSIS
        # =================================================

        forensic = self.forensic.analyze(
            img_224
        )

        # =================================================
        # MODEL INFERENCE
        # =================================================

        tensor = self.preprocess(
            img_224
        )

        with torch.no_grad():

            output = self.model(tensor)

            probs = torch.softmax(
                output,
                dim=1
            )

            idx = probs.argmax(dim=1).item()

            confidence = probs[0, idx].item()

        label = self.LABELS[idx]

        # =================================================
        # AI RISK LOGIC
        # =================================================

        ai_score = float(probs[0][0]) * 100

        if ai_score > 85:

            risk = "High"

        elif ai_score > 60:

            risk = "Medium"

        elif ai_score > 40:

            risk = "Low"

        else:

            risk = "Minimal"

        # =================================================
        # GRAD-CAM
        # =================================================

        heatmap = self.generate_gradcam(

            img_224,

            tensor,

            idx
        )

        # =================================================
        # EXPLANATION
        # =================================================

        explanation = self.generate_explanation(

            forensic,

            label
        )

        # =================================================
        # FINAL RESULT
        # =================================================

        return {

            "label": label,

            "confidence":
            round(confidence * 100, 2),

            "risk_level":
            risk,

            "scores": {

                "AI-Generated":
                round(float(probs[0][0]) * 100, 2),

                "Real Image":
                round(float(probs[0][1]) * 100, 2)
            },

            "visual_features": {

                "edge_density":
                forensic["edge_density"],

                "texture_consistency":
                forensic["texture_consistency"],

                "gan_artifact_score":
                forensic["gan_artifact_score"],

                "noise_level":
                forensic["noise_level"]
            },

            "image_statistics": {

                "width":
                int(img_224.shape[1]),

                "height":
                int(img_224.shape[0]),

                "channels":
                int(img_224.shape[2])
            },

            "forensic_analysis":
            forensic,

            "gradcam_heatmap":
            heatmap,

            "explanation":
            explanation
        }

    # =====================================================
    # GENERATE GRADCAM
    # =====================================================

    def generate_gradcam(self, img, tensor, idx):

        cam, _ = self.gradcam.generate(
            tensor,
            idx
        )

        heatmap = cv2.applyColorMap(

            np.uint8(255 * cam),

            cv2.COLORMAP_JET
        )

        heatmap = cv2.cvtColor(
            heatmap,
            cv2.COLOR_BGR2RGB
        )

        overlay = np.clip(

            img * 0.6 + heatmap * 0.4,

            0,

            255

        ).astype(np.uint8)

        fig, ax = plt.subplots(
            figsize=(5,5)
        )

        ax.imshow(overlay)

        ax.axis("off")

        buf = io.BytesIO()

        plt.savefig(

            buf,

            format='png',

            bbox_inches='tight',

            pad_inches=0
        )

        plt.close(fig)

        buf.seek(0)

        image_base64 = base64.b64encode(
            buf.read()
        ).decode()

        buf.close()

        return image_base64

    # =====================================================
    # EXPLANATION ENGINE
    # =====================================================

    def generate_explanation(

        self,

        forensic,

        label
    ):

        explanation = []

        if forensic["gan_artifact_score"] > 40:

            explanation.append(
                "GAN-related smoothing artifacts detected."
            )

        if forensic["edge_density"] < 10:

            explanation.append(
                "Low edge density detected."
            )

        if forensic["texture_consistency"] > 50:

            explanation.append(
                "Texture inconsistency observed."
            )

        if forensic["noise_level"] < 2:

            explanation.append(
                "Noise patterns appear artificially uniform."
            )

        if not explanation:

            explanation.append(
                "No major forensic anomalies detected."
            )

        if label == "AI-Generated":

            explanation.append(
                "The image contains synthetic visual characteristics commonly associated with AI-generated imagery."
            )

        else:

            explanation.append(
                "The image exhibits natural photographic characteristics consistent with real-world imagery."
            )

        return explanation