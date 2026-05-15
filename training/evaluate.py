"""
VERITAS AI — Evaluation & Report Generator
==========================================
Generates ALL evaluation outputs for your trained models:

  1. Classification Report (precision, recall, F1 per class)
  2. Confusion Matrix        (heatmap image)
  3. Per-Class Accuracy Bar Chart
  4. Training Curve          (if history saved)
  5. Full HTML Evaluation Report
  6. Console summary

Usage:
  python training/evaluate.py text
  python training/evaluate.py image
  python training/evaluate.py audio
  python training/evaluate.py all

All outputs saved to:  models/eval/
"""

import os
import sys
import json
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
os.chdir(ROOT)

OUTPUT_DIR = 'models/eval'
os.makedirs(OUTPUT_DIR, exist_ok=True)

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns

# ── dark theme ────────────────────────────────────────────────────────────────
BG      = '#0f1117'
SURFACE = '#131826'
BORDER  = '#1e2438'
VIOLET  = '#7c6ff7'
CORAL   = '#ff6b6b'
GREEN   = '#3ae8a0'
YELLOW  = '#ffd166'
CYAN    = '#00d4ff'
TEXT    = '#dde1f0'
TEXT2   = '#8590ab'

def _theme(fig, axes):
    fig.patch.set_facecolor(BG)
    for ax in (axes if hasattr(axes, '__iter__') else [axes]):
        ax.set_facecolor(SURFACE)
        ax.tick_params(colors=TEXT2, labelsize=9)
        ax.xaxis.label.set_color(TEXT2)
        ax.yaxis.label.set_color(TEXT2)
        ax.title.set_color(TEXT)
        for sp in ax.spines.values():
            sp.set_edgecolor(BORDER)

def _save(fig, name):
    path = os.path.join(OUTPUT_DIR, name)
    fig.savefig(path, dpi=130, bbox_inches='tight', facecolor=BG)
    plt.close(fig)
    print(f"  ✓  Saved → {path}")
    return path


# ══════════════════════════════════════════════════════════════════════════════
#  PLOT 1 — CONFUSION MATRIX HEATMAP
# ══════════════════════════════════════════════════════════════════════════════
def plot_confusion_matrix(cm, class_names, title, filename):
    fig, ax = plt.subplots(figsize=(6, 5))
    _theme(fig, [ax])

    # Normalise for percentage display
    cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True).clip(min=1)

    # Custom colormap on dark background
    cmap = sns.diverging_palette(220, 20, as_cmap=True)
    sns.heatmap(
        cm_norm, annot=False, fmt='', cmap='Blues',
        ax=ax, linewidths=1.5, linecolor=BG,
        cbar_kws={'shrink': 0.8},
    )

    # Annotate cells with count + percent
    for i in range(len(class_names)):
        for j in range(len(class_names)):
            count   = cm[i, j]
            percent = cm_norm[i, j] * 100
            color   = 'white' if cm_norm[i, j] < 0.6 else '#0f1117'
            ax.text(j + 0.5, i + 0.5,
                    f'{count}\n({percent:.1f}%)',
                    ha='center', va='center',
                    color=color, fontsize=9, fontweight='bold',
                    fontfamily='monospace')

    ax.set_xticklabels(class_names, rotation=25, ha='right', color=TEXT, fontsize=9)
    ax.set_yticklabels(class_names, rotation=0,  color=TEXT, fontsize=9)
    ax.set_xlabel('Predicted Label', fontsize=10, color=TEXT2, labelpad=8)
    ax.set_ylabel('True Label',      fontsize=10, color=TEXT2, labelpad=8)
    ax.set_title(title, fontsize=11, fontweight='bold', color=TEXT, pad=12)

    # Colorbar styling
    cbar = ax.collections[0].colorbar
    cbar.ax.tick_params(colors=TEXT2, labelsize=8)
    cbar.set_label('Normalised Rate', color=TEXT2, fontsize=8)

    plt.tight_layout()
    return _save(fig, filename)


# ══════════════════════════════════════════════════════════════════════════════
#  PLOT 2 — PER-CLASS METRICS BAR CHART
# ══════════════════════════════════════════════════════════════════════════════
def plot_per_class_metrics(report_dict, class_names, title, filename):
    """
    Grouped bar chart: Precision, Recall, F1-Score per class.
    """
    metrics  = ['precision', 'recall', 'f1-score']
    colors   = [VIOLET, GREEN, CORAL]
    x        = np.arange(len(class_names))
    width    = 0.25

    fig, ax = plt.subplots(figsize=(max(7, len(class_names)*2.5), 4.5))
    _theme(fig, [ax])

    for i, (metric, color) in enumerate(zip(metrics, colors)):
        vals = []
        for cls in class_names:
            if cls in report_dict:
                vals.append(report_dict[cls].get(metric, 0) * 100)
            else:
                vals.append(0.0)
        bars = ax.bar(x + i*width, vals, width, label=metric.title(),
                      color=color, alpha=0.85, edgecolor='none')
        # Value labels on bars
        for bar, val in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                    f'{val:.1f}%', ha='center', va='bottom',
                    color=TEXT, fontsize=7.5, fontweight='bold',
                    fontfamily='monospace')

    ax.set_xticks(x + width)
    ax.set_xticklabels(class_names, color=TEXT, fontsize=9)
    ax.set_ylim(0, 110)
    ax.set_ylabel('Score (%)', fontsize=9)
    ax.set_title(title, fontsize=11, fontweight='bold', color=TEXT)
    ax.grid(True, axis='y', color=BORDER, alpha=0.6, linestyle='--', linewidth=0.6)
    ax.legend(facecolor=SURFACE, edgecolor=BORDER, labelcolor=TEXT, fontsize=9)
    plt.tight_layout()
    return _save(fig, filename)


# ══════════════════════════════════════════════════════════════════════════════
#  PLOT 3 — TRAINING CURVE  (re-plot if raw data available)
# ══════════════════════════════════════════════════════════════════════════════
def plot_training_curve(train_losses, val_accs, title, filename):
    epochs = range(1, len(train_losses)+1)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4))
    _theme(fig, [ax1, ax2])

    # Loss
    ax1.plot(epochs, train_losses, color=VIOLET, linewidth=2, label='Train Loss')
    ax1.fill_between(epochs, train_losses, alpha=0.12, color=VIOLET)
    ax1.set_title('Training Loss', fontsize=10, fontweight='bold')
    ax1.set_xlabel('Epoch'); ax1.set_ylabel('Loss')
    ax1.grid(True, color=BORDER, alpha=0.6, linestyle='--', linewidth=0.5)
    ax1.legend(facecolor=SURFACE, edgecolor=BORDER, labelcolor=TEXT, fontsize=8)

    # Accuracy
    ax2.plot(epochs, [v*100 for v in val_accs], color=GREEN, linewidth=2, label='Val Accuracy')
    ax2.fill_between(epochs, [v*100 for v in val_accs], alpha=0.12, color=GREEN)
    best_epoch = int(np.argmax(val_accs)) + 1
    best_acc   = max(val_accs) * 100
    ax2.axvline(best_epoch, color=CORAL, linestyle='--', linewidth=1.2,
                label=f'Best: {best_acc:.2f}% @ epoch {best_epoch}')
    ax2.set_title('Validation Accuracy', fontsize=10, fontweight='bold')
    ax2.set_xlabel('Epoch'); ax2.set_ylabel('Accuracy (%)')
    ax2.set_ylim(0, 105)
    ax2.grid(True, color=BORDER, alpha=0.6, linestyle='--', linewidth=0.5)
    ax2.legend(facecolor=SURFACE, edgecolor=BORDER, labelcolor=TEXT, fontsize=8)

    fig.suptitle(title, fontsize=12, fontweight='bold', color=TEXT)
    plt.tight_layout()
    return _save(fig, filename)


# ══════════════════════════════════════════════════════════════════════════════
#  PLOT 4 — OVERALL ACCURACY SUMMARY
# ══════════════════════════════════════════════════════════════════════════════
def plot_accuracy_summary(results: dict, filename='accuracy_summary.png'):
    """
    Horizontal bar showing overall accuracy for each trained module.
    results = {'Text': 96.13, 'Image': 85.0, 'Audio': 78.0}
    """
    labels = list(results.keys())
    values = list(results.values())
    colors = [VIOLET, GREEN, CORAL, CYAN][:len(labels)]

    fig, ax = plt.subplots(figsize=(7, 3.5))
    _theme(fig, [ax])

    bars = ax.barh(labels, values, color=colors, height=0.45, edgecolor='none')
    for bar, val in zip(bars, values):
        ax.text(min(val+0.5, 103), bar.get_y()+bar.get_height()/2,
                f'{val:.2f}%', va='center', color=TEXT,
                fontsize=10, fontweight='bold', fontfamily='monospace')

    ax.set_xlim(0, 108)
    ax.set_xlabel('Accuracy (%)', fontsize=9)
    ax.set_title('Model Accuracy Summary — VERITAS AI',
                 fontsize=11, fontweight='bold')
    ax.invert_yaxis()
    ax.grid(True, axis='x', color=BORDER, alpha=0.6, linestyle='--')
    plt.tight_layout()
    return _save(fig, filename)


# ══════════════════════════════════════════════════════════════════════════════
#  HTML EVALUATION REPORT
# ══════════════════════════════════════════════════════════════════════════════
def generate_html_report(modality, class_names, report_dict,
                          cm, overall_acc, plot_paths: dict):
    import base64, datetime

    def img_b64(path):
        if path and os.path.exists(path):
            with open(path, 'rb') as f:
                return base64.b64encode(f.read()).decode()
        return ''

    rows = ''
    for cls in class_names:
        if cls not in report_dict: continue
        m = report_dict[cls]
        rows += f"""
        <tr>
          <td>{cls}</td>
          <td>{m.get('precision',0)*100:.2f}%</td>
          <td>{m.get('recall',0)*100:.2f}%</td>
          <td>{m.get('f1-score',0)*100:.2f}%</td>
          <td>{int(m.get('support',0))}</td>
        </tr>"""

    plots_html = ''
    for name, path in plot_paths.items():
        b64 = img_b64(path)
        if b64:
            plots_html += f"""
            <div class="plot-block">
              <h3>{name}</h3>
              <img src="data:image/png;base64,{b64}" alt="{name}">
            </div>"""

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>VERITAS AI — {modality.title()} Evaluation Report</title>
<link href="https://fonts.googleapis.com/css2?family=Space+Mono&family=Outfit:wght@400;700;900&display=swap" rel="stylesheet">
<style>
*{{box-sizing:border-box;margin:0;padding:0}}
body{{font-family:'Outfit',sans-serif;background:#0f1117;color:#dde1f0;padding:40px 24px}}
.wrap{{max-width:960px;margin:0 auto}}
h1{{font-size:28px;font-weight:900;color:#a09aff;letter-spacing:-1px;margin-bottom:4px}}
.sub{{font-family:'Space Mono',monospace;font-size:11px;color:#4e566a;letter-spacing:2px;margin-bottom:32px}}
.section{{background:#111525;border:1px solid #1c2235;border-radius:10px;padding:24px;margin-bottom:24px}}
.section h2{{font-size:14px;font-weight:800;color:#7c6ff7;letter-spacing:1px;text-transform:uppercase;margin-bottom:16px;font-family:'Space Mono',monospace}}
.big-acc{{font-size:52px;font-weight:900;color:#3ae8a0;font-family:'Space Mono',monospace}}
.big-label{{font-size:12px;color:#8590ab;letter-spacing:2px;text-transform:uppercase;margin-top:4px}}
table{{width:100%;border-collapse:collapse;font-size:13px}}
th{{background:#1c2235;color:#7c6ff7;font-family:'Space Mono',monospace;font-size:10px;letter-spacing:1px;padding:10px 14px;text-align:left}}
td{{padding:10px 14px;border-bottom:1px solid #1c2235;color:#dde1f0}}
tr:nth-child(even) td{{background:#0e1220}}
tr:hover td{{background:#1a1d2e}}
.plot-block{{margin-bottom:20px}}
.plot-block h3{{font-size:12px;color:#8590ab;font-family:'Space Mono',monospace;letter-spacing:1px;margin-bottom:8px;text-transform:uppercase}}
.plot-block img{{width:100%;border-radius:8px;border:1px solid #1c2235}}
.badge{{display:inline-block;font-family:'Space Mono',monospace;font-size:9px;letter-spacing:1.5px;padding:3px 10px;border-radius:4px;text-transform:uppercase}}
.badge.high{{background:rgba(58,232,160,.15);color:#3ae8a0;border:1px solid rgba(58,232,160,.3)}}
.badge.med{{background:rgba(255,209,102,.15);color:#ffd166;border:1px solid rgba(255,209,102,.3)}}
.meta{{font-family:'Space Mono',monospace;font-size:10px;color:#4e566a;margin-top:32px;padding-top:16px;border-top:1px solid #1c2235}}
</style>
</head>
<body>
<div class="wrap">
  <h1>VERITAS AI</h1>
  <div class="sub">// {modality.upper()} MODULE — EVALUATION REPORT — {datetime.datetime.now().strftime('%Y-%m-%d %H:%M')}</div>

  <div class="section" style="display:flex;gap:40px;align-items:center">
    <div>
      <div class="big-acc">{overall_acc:.2f}%</div>
      <div class="big-label">Overall Accuracy</div>
    </div>
    <div>
      <div style="margin-bottom:8px"><span class="badge {'high' if overall_acc>=90 else 'med'}">{'Excellent' if overall_acc>=90 else 'Good'} Performance</span></div>
      <div style="font-size:13px;color:#8590ab">Modality: <strong style="color:#dde1f0">{modality.title()}</strong></div>
      <div style="font-size:13px;color:#8590ab;margin-top:4px">Classes: <strong style="color:#dde1f0">{', '.join(class_names)}</strong></div>
      <div style="font-size:13px;color:#8590ab;margin-top:4px">Total Val Samples: <strong style="color:#dde1f0">{sum(int(report_dict[c].get('support',0)) for c in class_names if c in report_dict):,}</strong></div>
    </div>
  </div>

  <div class="section">
    <h2>Per-Class Metrics</h2>
    <table>
      <thead><tr><th>Class</th><th>Precision</th><th>Recall</th><th>F1-Score</th><th>Support</th></tr></thead>
      <tbody>{rows}</tbody>
    </table>
  </div>

  <div class="section">
    <h2>Visualizations</h2>
    {plots_html}
  </div>

  <div class="meta">
    Generated by VERITAS AI Evaluation Suite · Research &amp; Educational Use Only<br>
    Model: {modality.title()} Classifier · Framework: PyTorch + scikit-learn
  </div>
</div>
</body>
</html>"""

    path = os.path.join(OUTPUT_DIR, f'{modality}_eval_report.html')
    with open(path, 'w', encoding='utf-8') as f:
        f.write(html)
    print(f"  ✓  HTML report → {path}")
    return path


# ══════════════════════════════════════════════════════════════════════════════
#  TEXT EVALUATOR
# ══════════════════════════════════════════════════════════════════════════════
def evaluate_text():
    print("\n" + "═"*58)
    print("  TEXT MODULE EVALUATION")
    print("═"*58)

    try:
        import torch
        import pandas as pd
        from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
        from transformers import DistilBertTokenizer, DistilBertModel
        from text_module.text_analyzer import StylometricExtractor, HybridMLP
    except ImportError as e:
        print(f"  ✗  Missing: {e}"); return

    MODEL_PATH = 'models/hybrid_mlp_text.pt'
    CSV_PATH   = 'datasets/text_dataset.csv'

    if not os.path.exists(MODEL_PATH):
        print(f"  ✗  Model not found at {MODEL_PATH}. Run training first.")
        return
    if not os.path.exists(CSV_PATH):
        print(f"  ✗  Dataset not found at {CSV_PATH}."); return

    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"  ·  Device: {DEVICE}")

    # Load dataset
    df = pd.read_csv(CSV_PATH)
    df.columns = [c.lower().strip() for c in df.columns]
    df = df[['text','label']].dropna()
    df['label'] = df['label'].astype(int)
    print(f"  ·  Dataset: {len(df)} samples")
    print(f"  ·  Label distribution:\n{df['label'].value_counts().to_string()}")

    # Load models
    tokenizer  = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
    bert_model = DistilBertModel.from_pretrained('distilbert-base-uncased')
    bert_model.eval().to(DEVICE)

    model = HybridMLP(embed_dim=768, style_dim=9, num_classes=3)
    model.load_state_dict(torch.load(MODEL_PATH, map_location='cpu'))
    model.eval().to(DEVICE)
    print(f"  ✓  Model loaded from {MODEL_PATH}")

    # Use last 20% as evaluation set (same split as training)
    extractor  = StylometricExtractor()
    n_eval     = max(200, len(df) // 5)
    eval_df    = df.tail(n_eval).reset_index(drop=True)
    print(f"  ·  Evaluating on {len(eval_df)} samples...")

    embeddings, style_vecs, labels = [], [], []
    for i, (_, row) in enumerate(eval_df.iterrows()):
        text  = str(row['text'])[:512]
        label = int(row['label'])
        inputs = tokenizer(text, return_tensors='pt', truncation=True,
                           max_length=512, padding=True)
        inputs = {k: v.to(DEVICE) for k, v in inputs.items()}
        with torch.no_grad():
            out = bert_model(**inputs)
        emb = out.last_hidden_state[:,0,:].squeeze().cpu().numpy()
        sty = extractor.feature_vector(text)
        embeddings.append(emb)
        style_vecs.append(sty)
        labels.append(label)
        if (i+1) % 500 == 0:
            print(f"  ·  {i+1}/{len(eval_df)} processed...")

    X_emb = torch.tensor(np.array(embeddings), dtype=torch.float32).to(DEVICE)
    X_sty = torch.tensor(np.array(style_vecs),  dtype=torch.float32).to(DEVICE)
    y     = np.array(labels)

    with torch.no_grad():
        preds = model(X_emb, X_sty).argmax(dim=1).cpu().numpy()

    # Metrics
    ALL_NAMES = ['Human-Written', 'AI-Generated', 'Fake News']
    present   = sorted(set(y.tolist()) | set(preds.tolist()))
    names     = [ALL_NAMES[i] for i in present if i < len(ALL_NAMES)]
    acc       = accuracy_score(y, preds) * 100
    cm        = confusion_matrix(y, preds, labels=present)
    report    = classification_report(y, preds, labels=present,
                                       target_names=names, output_dict=True,
                                       zero_division=0)

    print(f"\n  Overall Accuracy: {acc:.2f}%\n")
    print(classification_report(y, preds, labels=present,
                                 target_names=names, zero_division=0))

    # Save history from training curve PNG if exists (fallback: dummy)
    history_path = 'models/text_training_curve.png'

    # Generate plots
    p1 = plot_confusion_matrix(cm, names,
                                'Text Classifier — Confusion Matrix',
                                'text_confusion_matrix.png')
    p2 = plot_per_class_metrics(report, names,
                                 'Text Classifier — Per-Class Metrics',
                                 'text_per_class_metrics.png')

    # HTML report
    generate_html_report(
        modality    = 'text',
        class_names = names,
        report_dict = report,
        cm          = cm,
        overall_acc = acc,
        plot_paths  = {
            'Confusion Matrix':    p1,
            'Per-Class Metrics':   p2,
            'Training Curve':      history_path if os.path.exists(history_path) else None,
        }
    )

    return acc, names, report


# ══════════════════════════════════════════════════════════════════════════════
#  IMAGE EVALUATOR
# ══════════════════════════════════════════════════════════════════════════════
def evaluate_image():
    print("\n" + "═"*58)
    print("  IMAGE MODULE EVALUATION")
    print("═"*58)

    try:
        import torch
        import torch.nn as nn
        import torchvision.models as models
        import torchvision.transforms as T
        from torch.utils.data import DataLoader
        from torchvision.datasets import ImageFolder
        from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
    except ImportError as e:
        print(f"  ✗  Missing: {e}"); return

    MODEL_PATH = 'models/resnet50_image.pt'
    VAL_DIR    = 'datasets/images/val'
    DEVICE     = 'cuda' if torch.cuda.is_available() else 'cpu'

    if not os.path.exists(MODEL_PATH):
        print(f"  ✗  Model not found at {MODEL_PATH}. Run training first."); return
    if not os.path.exists(VAL_DIR):
        print(f"  ✗  Val dataset not found at {VAL_DIR}."); return

    transform = T.Compose([
        T.Resize((224,224)), T.ToTensor(),
        T.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225]),
    ])
    val_ds     = ImageFolder(VAL_DIR, transform=transform)
    val_loader = DataLoader(val_ds, batch_size=32, shuffle=False, num_workers=2)
    names      = val_ds.classes
    print(f"  ·  Val samples: {len(val_ds)}  |  Classes: {names}")

    # Load model
    model     = models.resnet50(weights=None)
    n_classes = len(names)
    model.fc  = nn.Sequential(
        nn.Linear(2048,512), nn.BatchNorm1d(512), nn.ReLU(), nn.Dropout(0.4),
        nn.Linear(512,128),  nn.ReLU(), nn.Dropout(0.2),
        nn.Linear(128, n_classes),
    )
    model.load_state_dict(torch.load(MODEL_PATH, map_location='cpu'))
    model.eval().to(DEVICE)
    print(f"  ✓  ResNet50 loaded from {MODEL_PATH}")

    all_preds, all_true = [], []
    with torch.no_grad():
        for imgs, labels in val_loader:
            preds = model(imgs.to(DEVICE)).argmax(1).cpu().numpy()
            all_preds.extend(preds)
            all_true.extend(labels.numpy())

    y, preds = np.array(all_true), np.array(all_preds)
    acc    = accuracy_score(y, preds) * 100
    cm     = confusion_matrix(y, preds)
    report = classification_report(y, preds, target_names=names,
                                    output_dict=True, zero_division=0)

    print(f"\n  Overall Accuracy: {acc:.2f}%\n")
    print(classification_report(y, preds, target_names=names, zero_division=0))

    history_path = 'models/image_training_curve.png'
    p1 = plot_confusion_matrix(cm, names,
                                'Image Classifier — Confusion Matrix',
                                'image_confusion_matrix.png')
    p2 = plot_per_class_metrics(report, names,
                                 'Image Classifier — Per-Class Metrics',
                                 'image_per_class_metrics.png')
    generate_html_report(
        modality='image', class_names=names, report_dict=report,
        cm=cm, overall_acc=acc,
        plot_paths={
            'Confusion Matrix':  p1,
            'Per-Class Metrics': p2,
            'Training Curve':    history_path if os.path.exists(history_path) else None,
        }
    )
    return acc, names, report


# ══════════════════════════════════════════════════════════════════════════════
#  AUDIO EVALUATOR
# ══════════════════════════════════════════════════════════════════════════════
def evaluate_audio():
    print("\n" + "═"*58)
    print("  AUDIO MODULE EVALUATION")
    print("═"*58)

    try:
        import torch
        import librosa
        from torch.utils.data import Dataset, DataLoader
        from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
        from audio_module.audio_analyzer import AudioCNN
    except ImportError as e:
        print(f"  ✗  Missing: {e}"); return

    MODEL_PATH = 'models/audio_cnn.pt'
    AUDIO_ROOT = 'datasets/audio'
    DEVICE     = 'cuda' if torch.cuda.is_available() else 'cpu'

    if not os.path.exists(MODEL_PATH):
        print(f"  ✗  Model not found at {MODEL_PATH}. Run training first."); return

    class AudioEvalDataset(Dataset):
        LABEL_MAP = {'real':0,'ai':1,'cloned':2}
        EXTS      = {'.wav','.mp3','.flac','.ogg','.m4a'}
        def __init__(self, root, n_mfcc=40, max_len=128, sr=22050):
            self.samples = []
            self.n_mfcc, self.max_len, self.sr = n_mfcc, max_len, sr
            for cls, lbl in self.LABEL_MAP.items():
                folder = os.path.join(root, cls)
                if not os.path.exists(folder): continue
                for f in os.listdir(folder):
                    if os.path.splitext(f)[1].lower() in self.EXTS:
                        self.samples.append((os.path.join(folder,f), lbl))
        def __len__(self): return len(self.samples)
        def __getitem__(self, idx):
            path, label = self.samples[idx]
            try:
                y, sr = librosa.load(path, sr=self.sr, mono=True)
            except Exception:
                y, sr = np.zeros(self.sr, dtype=np.float32), self.sr
            mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=self.n_mfcc)
            if mfcc.shape[1] < self.max_len:
                mfcc = np.pad(mfcc, ((0,0),(0,self.max_len-mfcc.shape[1])))
            else:
                mfcc = mfcc[:,:self.max_len]
            mean, std = mfcc.mean(), mfcc.std()
            mfcc = (mfcc - mean) / (std + 1e-9)
            return torch.tensor(mfcc, dtype=torch.float32).unsqueeze(0), label

    ds     = AudioEvalDataset(AUDIO_ROOT)
    loader = DataLoader(ds, batch_size=16, shuffle=False)
    print(f"  ·  Eval samples: {len(ds)}")

    model = AudioCNN(n_mfcc=40, num_classes=3)
    model.load_state_dict(torch.load(MODEL_PATH, map_location='cpu'))
    model.eval().to(DEVICE)
    print(f"  ✓  AudioCNN loaded from {MODEL_PATH}")

    all_preds, all_true = [], []
    with torch.no_grad():
        for mfcc_b, labels in loader:
            preds = model(mfcc_b.to(DEVICE)).argmax(1).cpu().numpy()
            all_preds.extend(preds)
            all_true.extend(labels.numpy())

    y, preds   = np.array(all_true), np.array(all_preds)
    ALL_NAMES  = ['Real Voice','AI-Generated Voice','Cloned Voice']
    present    = sorted(set(y.tolist()) | set(preds.tolist()))
    names      = [ALL_NAMES[i] for i in present if i < len(ALL_NAMES)]
    acc        = accuracy_score(y, preds) * 100
    cm         = confusion_matrix(y, preds, labels=present)
    report     = classification_report(y, preds, labels=present,
                                        target_names=names, output_dict=True,
                                        zero_division=0)

    print(f"\n  Overall Accuracy: {acc:.2f}%\n")
    print(classification_report(y, preds, labels=present, target_names=names, zero_division=0))

    history_path = 'models/audio_training_curve.png'
    p1 = plot_confusion_matrix(cm, names,
                                'Audio Classifier — Confusion Matrix',
                                'audio_confusion_matrix.png')
    p2 = plot_per_class_metrics(report, names,
                                 'Audio Classifier — Per-Class Metrics',
                                 'audio_per_class_metrics.png')
    generate_html_report(
        modality='audio', class_names=names, report_dict=report,
        cm=cm, overall_acc=acc,
        plot_paths={
            'Confusion Matrix':  p1,
            'Per-Class Metrics': p2,
            'Training Curve':    history_path if os.path.exists(history_path) else None,
        }
    )
    return acc, names, report


# ══════════════════════════════════════════════════════════════════════════════
#  COMBINED SUMMARY
# ══════════════════════════════════════════════════════════════════════════════
def print_summary(results: dict):
    print("\n" + "═"*58)
    print("  EVALUATION SUMMARY — VERITAS AI")
    print("═"*58)
    for mod, acc in results.items():
        bar_len = int(acc / 2)
        bar     = '█' * bar_len + '░' * (50 - bar_len)
        print(f"  {mod:<8}  {bar}  {acc:.2f}%")
    print("═"*58)
    print(f"\n  All outputs saved to:  {OUTPUT_DIR}/")
    print("  Files generated:")
    for f in sorted(os.listdir(OUTPUT_DIR)):
        print(f"    · {f}")
    print()


# ══════════════════════════════════════════════════════════════════════════════
#  ENTRY POINT
# ══════════════════════════════════════════════════════════════════════════════
USAGE = """
VERITAS AI — Evaluation Script
Usage:
  python training/evaluate.py text     Evaluate text model
  python training/evaluate.py image    Evaluate image model
  python training/evaluate.py audio    Evaluate audio model
  python training/evaluate.py all      Evaluate all + summary chart
"""

if __name__ == '__main__':
    mode = sys.argv[1] if len(sys.argv) > 1 else 'help'

    if mode == 'help':
        print(USAGE); sys.exit(0)

    summary_accs = {}

    if mode in ('text', 'all'):
        result = evaluate_text()
        if result: summary_accs['Text'] = result[0]

    if mode in ('image', 'all'):
        result = evaluate_image()
        if result: summary_accs['Image'] = result[0]

    if mode in ('audio', 'all'):
        result = evaluate_audio()
        if result: summary_accs['Audio'] = result[0]

    if summary_accs:
        plot_accuracy_summary(summary_accs, 'accuracy_summary.png')
        print_summary(summary_accs)