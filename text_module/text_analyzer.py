"""
TEXT ANALYSIS MODULE
====================
Detects : Fake News | AI-Generated Text | Human-Written Text
Pipeline : Input → Preprocess → DistilBERT (768-d) + Stylometric (9-d) → HybridMLP → Prediction + XAI
"""

import re
import math
import numpy as np
from collections import Counter


# ──────────────────────────────────────────────────────────────
#  STYLOMETRIC FEATURE EXTRACTOR
# ──────────────────────────────────────────────────────────────
class StylometricExtractor:
    """
    Extracts 9 handcrafted linguistic / stylistic features.
    These complement deep embeddings and are key for XAI explanations.
    """

    FEATURE_NAMES = [
        "lexical_diversity",
        "sentence_length_variance",
        "repetition_score",
        "burstiness",
        "readability_score",
        "avg_word_length",
        "unique_bigram_ratio",
        "punctuation_density",
        "semantic_coherence",
    ]

    def extract(self, text: str) -> dict:
        sentences = self._split_sentences(text)
        words     = self._tokenize(text)
        if len(words) < 5:
            return {k: 0.0 for k in self.FEATURE_NAMES}

        return {
            "lexical_diversity":        self._lexical_diversity(words),
            "sentence_length_variance": self._sent_len_variance(sentences),
            "repetition_score":         self._repetition_score(words),
            "burstiness":               self._burstiness(sentences),
            "readability_score":        self._flesch_reading_ease(words, sentences),
            "avg_word_length":          float(np.mean([len(w) for w in words])),
            "unique_bigram_ratio":      self._bigram_diversity(words),
            "punctuation_density":      self._punct_density(text),
            "semantic_coherence":       self._semantic_coherence(sentences),
        }

    def feature_vector(self, text: str) -> np.ndarray:
        return np.array(list(self.extract(text).values()), dtype=np.float32)

    # ── private helpers ────────────────────────────────────────

    def _split_sentences(self, text):
        return [s.strip() for s in re.split(r'[.!?]+', text) if len(s.strip()) > 3]

    def _tokenize(self, text):
        return re.findall(r'\b[a-zA-Z]+\b', text.lower())

    def _lexical_diversity(self, words):
        return len(set(words)) / len(words) if words else 0.0

    def _sent_len_variance(self, sentences):
        lens = [len(s.split()) for s in sentences if s]
        return float(np.var(lens)) if len(lens) >= 2 else 0.0

    def _repetition_score(self, words):
        counts   = Counter(words)
        repeated = sum(c for c in counts.values() if c > 1)
        return repeated / len(words) if words else 0.0

    def _burstiness(self, sentences):
        """B = (σ - μ) / (σ + μ)  — negative means uniform (AI signal)."""
        lens = [len(s.split()) for s in sentences if s]
        if len(lens) < 3:
            return 0.0
        mu, sigma = np.mean(lens), np.std(lens)
        denom = sigma + mu
        return float((sigma - mu) / denom) if denom > 0 else 0.0

    def _flesch_reading_ease(self, words, sentences):
        if not sentences or not words:
            return 0.0
        avg_sl  = len(words) / max(len(sentences), 1)
        avg_syl = np.mean([self._syllables(w) for w in words])
        score   = 206.835 - 1.015 * avg_sl - 84.6 * avg_syl
        return float(np.clip(score, 0, 100))

    def _syllables(self, word):
        return max(1, len(re.findall(r'[aeiou]', word.lower())))

    def _bigram_diversity(self, words):
        if len(words) < 2:
            return 0.0
        bigrams = list(zip(words, words[1:]))
        return len(set(bigrams)) / len(bigrams)

    def _punct_density(self, text):
        punct = sum(1 for c in text if c in '.,;:!?—-')
        return punct / max(len(text), 1)

    def _semantic_coherence(self, sentences):
        """Jaccard similarity between consecutive sentences (proxy for coherence)."""
        if len(sentences) < 2:
            return 1.0
        scores = []
        for i in range(len(sentences) - 1):
            w1, w2 = set(sentences[i].lower().split()), set(sentences[i+1].lower().split())
            union  = w1 | w2
            if union:
                scores.append(len(w1 & w2) / len(union))
        return float(np.mean(scores)) if scores else 0.0


# ──────────────────────────────────────────────────────────────
#  HYBRID MLP  (PyTorch)
# ──────────────────────────────────────────────────────────────
try:
    import torch
    import torch.nn as nn

    class HybridMLP(nn.Module):
        """
        Two-branch network:
          Branch A : DistilBERT CLS embedding (768-d) → 128-d
          Branch B : Stylometric features   (  9-d) →  32-d
          Fusion   : [128+32] → 64 → 3 classes
        """
        def __init__(self, embed_dim=768, style_dim=9, num_classes=3):
            super().__init__()
            self.text_branch = nn.Sequential(
                nn.Linear(embed_dim, 256), nn.LayerNorm(256), nn.ReLU(), nn.Dropout(0.3),
                nn.Linear(256, 128),       nn.ReLU(),
            )
            self.style_branch = nn.Sequential(
                nn.Linear(style_dim, 32), nn.ReLU(),
                nn.Linear(32, 32),        nn.ReLU(),
            )
            self.classifier = nn.Sequential(
                nn.Linear(128 + 32, 64), nn.ReLU(), nn.Dropout(0.2),
                nn.Linear(64, num_classes),
            )

        def forward(self, embeddings, style_feats):
            t = self.text_branch(embeddings)
            s = self.style_branch(style_feats)
            return self.classifier(torch.cat([t, s], dim=-1))

    TORCH_AVAILABLE = True

except ImportError:
    TORCH_AVAILABLE = False
    HybridMLP = None


# ──────────────────────────────────────────────────────────────
#  MAIN TEXT ANALYZER
# ──────────────────────────────────────────────────────────────
class TextAnalyzer:

    LABELS = ["Human-Written", "AI-Generated", "Fake News"]

    # Sensational / clickbait keywords → Fake-News signal
    SENSATIONAL = {
        'shocking','bombshell','revealed','secret','conspiracy','exposed',
        'hoax','breaking','exclusive','scandal','coverup','truth','hidden',
        'illuminati','government hiding','they don\'t want you','wake up',
    }

    def __init__(self):
        self.extractor   = StylometricExtractor()
        self._tokenizer  = None
        self._bert       = None
        self._mlp        = None
        self._load_models()

    # ── model loading ──────────────────────────────────────────

    def _load_models(self):
        # DistilBERT
        try:
            from transformers import DistilBertTokenizer, DistilBertModel
            self._tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
            self._bert      = DistilBertModel.from_pretrained('distilbert-base-uncased')
            self._bert.eval()
            print("[TextAnalyzer] ✓ DistilBERT loaded")
        except Exception as e:
            print(f"[TextAnalyzer] DistilBERT unavailable ({e}) — rule-based mode")

        # Trained HybridMLP weights (optional)
        try:
            import torch, os
            if HybridMLP and os.path.exists('models/hybrid_mlp_text.pt'):
                self._mlp = HybridMLP()
                self._mlp.load_state_dict(torch.load('models/hybrid_mlp_text.pt', map_location='cpu'))
                self._mlp.eval()
                print("[TextAnalyzer] ✓ HybridMLP weights loaded")
        except Exception as e:
            print(f"[TextAnalyzer] No trained MLP weights ({e})")

    # ── public API ─────────────────────────────────────────────

    def get_embedding(self, text: str) -> np.ndarray:
        if self._bert is None:
            return np.zeros(768, dtype=np.float32)
        import torch
        inputs = self._tokenizer(text, return_tensors='pt', truncation=True,
                                  max_length=512, padding=True)
        with torch.no_grad():
            out = self._bert(**inputs)
        return out.last_hidden_state[:, 0, :].squeeze().numpy().astype(np.float32)

    def analyze(self, text: str) -> dict:
        text = text.strip()

        # Feature extraction
        style_dict = self.extractor.extract(text)
        style_vec  = np.array(list(style_dict.values()), dtype=np.float32)
        embedding  = self.get_embedding(text)

        # Inference
        if self._mlp is not None and TORCH_AVAILABLE:
            scores = self._mlp_inference(embedding, style_vec)
        else:
            scores = self._rule_based_score(style_dict, text)

        label_idx  = int(np.argmax(scores))
        label      = self.LABELS[label_idx]
        confidence = float(scores[label_idx])
        risk       = self._risk_level(confidence, label)

        return {
            "label":                label,
            "confidence":           round(confidence * 100, 2),
            "risk_level":           risk,
            "scores": {
                self.LABELS[i]: round(float(scores[i]) * 100, 2) for i in range(3)
            },
            "stylometric_features": {k: round(float(v), 4) for k, v in style_dict.items()},
            "explanation":          self._build_explanation(style_dict, text, label),
            "word_count":           len(text.split()),
            "char_count":           len(text),
        }

    # ── inference helpers ──────────────────────────────────────

    def _mlp_inference(self, embedding, style_vec) -> np.ndarray:
        import torch, torch.nn.functional as F
        with torch.no_grad():
            emb_t   = torch.tensor(embedding).unsqueeze(0)
            sty_t   = torch.tensor(style_vec).unsqueeze(0)
            logits  = self._mlp(emb_t, sty_t)
            probs   = F.softmax(logits, dim=-1).squeeze().numpy()
        return probs

    def _rule_based_score(self, f: dict, text: str) -> np.ndarray:
        """
        Heuristic scoring — used when no trained weights are available.
        Replace entirely with _mlp_inference() once you train the model.
        """
        human_score = 0.35
        ai_score    = 0.0
        fake_score  = 0.0

        # Burstiness: low → uniform → AI signal
        B = f["burstiness"]
        if   B < -0.1: ai_score += 0.40
        elif B < 0.05: ai_score += 0.20
        else:          human_score += 0.20

        # Sentence length variance: low → AI
        if f["sentence_length_variance"] < 4:
            ai_score += 0.20

        # Lexical diversity: high → human
        if f["lexical_diversity"] > 0.72:
            human_score += 0.25
        elif f["lexical_diversity"] < 0.45:
            ai_score += 0.15

        # High repetition → fake/content-farm
        if f["repetition_score"] > 0.60:
            fake_score += 0.30
        elif f["repetition_score"] > 0.50:
            fake_score += 0.15

        # Very high readability → AI (polished prose)
        if f["readability_score"] > 72:
            ai_score += 0.15

        # Sensational keywords → fake
        words_lower = text.lower()
        hits = sum(1 for kw in self.SENSATIONAL if kw in words_lower)
        fake_score += min(hits * 0.12, 0.40)

        # Low semantic coherence → hallucinated / stitched content
        if f["semantic_coherence"] < 0.08:
            fake_score += 0.15

        raw    = np.array([human_score, ai_score, fake_score], dtype=np.float64)
        raw    = np.clip(raw, 0, None)
        softm  = np.exp(raw * 3.0)
        return (softm / softm.sum()).astype(np.float32)

    def _risk_level(self, confidence: float, label: str) -> str:
        if label == "Human-Written":
            return "Low"
        if confidence > 0.80:
            return "High"
        if confidence > 0.55:
            return "Medium"
        return "Low"

    def _build_explanation(self, f: dict, text: str, label: str) -> list:
        reasons = []
        B = f["burstiness"]
        if B < 0.0:
            reasons.append(
                f"Burstiness score is {B:.3f} (negative) — sentence lengths are unusually uniform, "
                "a hallmark of AI-generated prose."
            )
        if f["sentence_length_variance"] < 4:
            reasons.append(
                "Extremely low sentence-length variance — human writing naturally varies sentence rhythm."
            )
        if f["lexical_diversity"] > 0.72:
            reasons.append(
                f"High lexical diversity ({f['lexical_diversity']:.2f}) — rich vocabulary range "
                "typical of human writers."
            )
        if f["repetition_score"] > 0.55:
            reasons.append(
                f"Elevated word repetition ({f['repetition_score']:.2f}) — may indicate "
                "content-farmed or AI-padded text."
            )
        if f["readability_score"] > 72:
            reasons.append(
                f"Unusually high Flesch Reading Ease ({f['readability_score']:.1f}) — "
                "AI models tend to produce polished, accessible prose."
            )
        words_lower = text.lower()
        hits = [kw for kw in self.SENSATIONAL if kw in words_lower]
        if hits:
            reasons.append(
                f"Sensational / clickbait keywords detected: {', '.join(hits[:4])} — "
                "strong fake-news signal."
            )
        if f["semantic_coherence"] < 0.08:
            reasons.append(
                "Very low inter-sentence coherence — content may be stitched together "
                "or hallucinated."
            )
        if not reasons:
            reasons.append(
                "No single dominant stylometric signal. Prediction is based on the "
                "combined weighted feature profile."
            )
        return reasons