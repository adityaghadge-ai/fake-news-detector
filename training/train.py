"""
VERITAS AI — Training Script
=============================
Trains all three ML models:
  1. HybridMLP    — Text classifier (DistilBERT + Stylometric)
  2. ResNet50     — Image classifier (Real / AI-Generated / Manipulated)
  3. AudioCNN     — Audio classifier (Real Voice / AI Voice / Cloned)

Usage:
  python training/train.py text
  python training/train.py image
  python training/train.py audio
  python training/train.py all

Dataset Setup:
  Text  → datasets/text_dataset.csv   (columns: text, label)
  Image → datasets/images/train/real/ datasets/images/train/ai/ datasets/images/train/manipulated/
          datasets/images/val/real/   datasets/images/val/ai/   datasets/images/val/manipulated/
  Audio → datasets/audio/real/  datasets/audio/ai/  datasets/audio/cloned/
"""

import os
import sys
import time
import numpy as np

# ── make sure project root is on path ─────────────────────────────────────────
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
os.chdir(ROOT)


# ══════════════════════════════════════════════════════════════════════════════
#  UTILITIES
# ══════════════════════════════════════════════════════════════════════════════

def banner(title):
    w = 58
    print("\n" + "═"*w)
    print(f"  {title}")
    print("═"*w)

def section(title):
    print(f"\n── {title} " + "─"*(52-len(title)))

def ok(msg):   print(f"  ✓  {msg}")
def info(msg): print(f"  ·  {msg}")
def warn(msg): print(f"  ⚠  {msg}")
def err(msg):  print(f"  ✗  {msg}")

def save_model(state_dict, path):
    import torch
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(state_dict, path)
    ok(f"Model saved → {path}")

def plot_history(train_losses, val_accs, title, save_path):
    """Save a training curve plot."""
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
        fig.patch.set_facecolor('#0f1117')
        for ax in [ax1, ax2]:
            ax.set_facecolor('#131826')
            ax.tick_params(colors='#8590ab')
            for sp in ax.spines.values(): sp.set_edgecolor('#1c2235')

        ax1.plot(train_losses, color='#7c6ff7', linewidth=2)
        ax1.set_title('Training Loss', color='white', fontsize=10)
        ax1.set_xlabel('Epoch', color='#8590ab')
        ax1.set_ylabel('Loss',  color='#8590ab')
        ax1.grid(True, color='#1c2235', alpha=0.6)

        ax2.plot(val_accs, color='#3ae8a0', linewidth=2)
        ax2.set_title('Validation Accuracy', color='white', fontsize=10)
        ax2.set_xlabel('Epoch', color='#8590ab')
        ax2.set_ylabel('Accuracy', color='#8590ab')
        ax2.grid(True, color='#1c2235', alpha=0.6)

        fig.suptitle(title, color='white', fontsize=11, fontweight='bold')
        plt.tight_layout()
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=100, bbox_inches='tight', facecolor='#0f1117')
        plt.close()
        ok(f"Training curve saved → {save_path}")
    except Exception as e:
        warn(f"Could not save training curve: {e}")


# ══════════════════════════════════════════════════════════════════════════════
#  MODULE 1 — TEXT MODEL TRAINING
# ══════════════════════════════════════════════════════════════════════════════

def train_text(epochs=25, batch_size=32, lr=2e-4):
    banner("TEXT MODEL TRAINING  —  HybridMLP (DistilBERT + Stylometric)")

    # ── imports ───────────────────────────────────────────────────────────────
    try:
        import torch
        import torch.nn as nn
        import torch.nn.functional as F
        import pandas as pd
        from sklearn.model_selection import train_test_split
        from sklearn.metrics import classification_report, confusion_matrix
        from transformers import DistilBertTokenizer, DistilBertModel
        from text_module.text_analyzer import StylometricExtractor, HybridMLP
    except ImportError as e:
        err(f"Missing dependency: {e}")
        err("Run: pip install -r requirements.txt")
        return

    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    info(f"Device: {DEVICE}")

    # ── load dataset ──────────────────────────────────────────────────────────
    section("Loading Dataset")
    CSV_PATH = 'datasets/text_dataset.csv'

    if os.path.exists(CSV_PATH):
        df = pd.read_csv(CSV_PATH)
        # Normalise column names
        df.columns = [c.lower().strip() for c in df.columns]
        if 'label' not in df.columns:
            err("CSV must have a 'label' column (0=Human, 1=AI-Generated, 2=Fake News)")
            return
        if 'text' not in df.columns:
            err("CSV must have a 'text' column")
            return
        df = df[['text','label']].dropna()
        df['label'] = df['label'].astype(int)
        info(f"Loaded {len(df)} samples from {CSV_PATH}")
        info(f"Label distribution:\n{df['label'].value_counts().to_string()}")
    else:
        warn(f"Dataset not found at {CSV_PATH}")
        warn("Creating synthetic demo dataset (50 samples)...")
        df = _make_demo_text_df()
        os.makedirs('datasets', exist_ok=True)
        df.to_csv(CSV_PATH, index=False)
        ok(f"Demo dataset saved → {CSV_PATH}  (replace with real WELFake data!)")

    # ── load DistilBERT ───────────────────────────────────────────────────────
    section("Loading DistilBERT")
    try:
        tokenizer  = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
        bert_model = DistilBertModel.from_pretrained('distilbert-base-uncased')
        bert_model.eval().to(DEVICE)
        ok("DistilBERT loaded")
    except Exception as e:
        err(f"Could not load DistilBERT: {e}")
        err("Try: pip install transformers  or check your internet connection")
        return

    # ── extract features ──────────────────────────────────────────────────────
    section("Extracting Features (this may take a few minutes)")
    extractor = StylometricExtractor()
    embeddings, style_vecs, labels = [], [], []

    for i, (_, row) in enumerate(df.iterrows()):
        text  = str(row['text'])[:512]
        label = int(row['label'])

        # DistilBERT CLS embedding
        inputs = tokenizer(text, return_tensors='pt', truncation=True,
                           max_length=512, padding=True)
        inputs = {k: v.to(DEVICE) for k, v in inputs.items()}
        with torch.no_grad():
            out = bert_model(**inputs)
        emb = out.last_hidden_state[:, 0, :].squeeze().cpu().numpy()

        # Stylometric
        sty = extractor.feature_vector(text)

        embeddings.append(emb)
        style_vecs.append(sty)
        labels.append(label)

        if (i+1) % 100 == 0:
            info(f"  Processed {i+1}/{len(df)} samples...")

    ok(f"Feature extraction complete — {len(labels)} samples")

    X_emb   = torch.tensor(np.array(embeddings),  dtype=torch.float32)
    X_sty   = torch.tensor(np.array(style_vecs),  dtype=torch.float32)
    y       = torch.tensor(labels,                 dtype=torch.long)

    # ── train/val split ───────────────────────────────────────────────────────
    section("Splitting Data")
    n_total = len(y)
    n_train = int(0.80 * n_total)
    perm    = torch.randperm(n_total)

    tr_emb, va_emb = X_emb[perm[:n_train]],  X_emb[perm[n_train:]]
    tr_sty, va_sty = X_sty[perm[:n_train]],  X_sty[perm[n_train:]]
    tr_y,   va_y   = y[perm[:n_train]],       y[perm[n_train:]]

    info(f"Train: {n_train} samples  |  Val: {n_total - n_train} samples")

    # ── model ─────────────────────────────────────────────────────────────────
    section("Building HybridMLP")
    model     = HybridMLP(embed_dim=768, style_dim=9, num_classes=3).to(DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    criterion = nn.CrossEntropyLoss()

    # Class weights (handle imbalance)
    label_counts = np.bincount(labels, minlength=3).astype(float)
    label_counts = np.where(label_counts == 0, 1, label_counts)
    weights = torch.tensor(1.0 / label_counts, dtype=torch.float32).to(DEVICE)
    criterion = nn.CrossEntropyLoss(weight=weights)

    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    info(f"HybridMLP trainable parameters: {total_params:,}")

    # ── training loop ─────────────────────────────────────────────────────────
    section(f"Training  ({epochs} epochs, batch={batch_size}, lr={lr})")
    tr_emb, tr_sty, tr_y = tr_emb.to(DEVICE), tr_sty.to(DEVICE), tr_y.to(DEVICE)
    va_emb, va_sty, va_y = va_emb.to(DEVICE), va_sty.to(DEVICE), va_y.to(DEVICE)

    best_val_acc = 0.0
    train_losses, val_accs = [], []

    for epoch in range(1, epochs+1):
        model.train()
        perm  = torch.randperm(n_train)
        total_loss = 0.0
        n_batches  = 0

        for i in range(0, n_train, batch_size):
            idx    = perm[i : i+batch_size]
            logits = model(tr_emb[idx], tr_sty[idx])
            loss   = criterion(logits, tr_y[idx])
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total_loss += loss.item()
            n_batches  += 1

        scheduler.step()
        avg_loss = total_loss / max(n_batches, 1)

        # Validation
        model.eval()
        with torch.no_grad():
            val_logits = model(va_emb, va_sty)
            val_preds  = val_logits.argmax(dim=1)
            val_acc    = (val_preds == va_y).float().mean().item()

        train_losses.append(avg_loss)
        val_accs.append(val_acc)

        # Save best
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            save_model(model.state_dict(), 'models/hybrid_mlp_text.pt')

        if epoch % 5 == 0 or epoch == 1:
            print(f"  Epoch {epoch:3d}/{epochs}  |  Loss: {avg_loss:.4f}  |  Val Acc: {val_acc:.4f}  |  Best: {best_val_acc:.4f}")

    # ── final evaluation ──────────────────────────────────────────────────────
    section("Final Evaluation")
    model.eval()
    with torch.no_grad():
        preds = model(va_emb, va_sty).argmax(dim=1).cpu().numpy()
    true = va_y.cpu().numpy()

    # Detect which classes are actually present in the validation set
    ALL_NAMES  = ['Human-Written', 'AI-Generated', 'Fake News']
    present    = sorted(set(true.tolist()) | set(preds.tolist()))
    tgt_names  = [ALL_NAMES[i] for i in present if i < len(ALL_NAMES)]

    print("\n" + classification_report(
        true, preds,
        labels=present,
        target_names=tgt_names,
        zero_division=0
    ))

    cm = confusion_matrix(true, preds, labels=present)
    info(f"Classes present: {tgt_names}")
    info(f"Confusion Matrix:\n{cm}")

    plot_history(train_losses, val_accs,
                 'HybridMLP — Text Classifier Training',
                 'models/text_training_curve.png')

    ok(f"Best validation accuracy: {best_val_acc:.4f} ({best_val_acc*100:.2f}%)")
    ok("Text model training complete!")


def _make_demo_text_df():
    """50-sample synthetic dataset for smoke-testing the pipeline."""
    import pandas as pd

    human = [
        "The city council approved the new budget plan after a lengthy debate yesterday afternoon.",
        "Researchers at MIT have published a study on the effects of sleep deprivation on memory.",
        "The local sports team won the championship after a tense final match last weekend.",
        "A new restaurant opened downtown offering a blend of Mediterranean and Asian cuisines.",
        "The school board voted to extend the academic year by two weeks starting next semester.",
        "Traffic on the highway was disrupted this morning due to an accident near the overpass.",
        "The annual music festival will be held in the city park over the coming long weekend.",
        "Scientists have found a new species of beetle in the tropical rainforests of Brazil.",
        "The hospital announced an expansion of its emergency department to handle more patients.",
        "Local farmers are adopting new irrigation techniques to cope with seasonal droughts.",
        "The university library has extended its operating hours during the examination period.",
        "A charity run raised over fifty thousand dollars for children with rare diseases.",
        "The mayor announced a new initiative to improve public transport across the city.",
        "Engineers completed repairs on the bridge ahead of the original scheduled deadline.",
        "Community volunteers planted over three hundred trees along the riverbank last weekend.",
    ]
    ai = [
        "The optimal approach to neural network training involves careful hyperparameter selection. "
        "This systematic methodology ensures convergence. Explainable AI improves transparency.",
        "Machine learning models demonstrate significant potential across various domains. "
        "The systematic evaluation of these models provides valuable insights into their performance.",
        "Natural language processing techniques enable sophisticated text analysis. "
        "These methods provide consistent results across diverse linguistic contexts.",
        "Deep learning architectures offer powerful feature extraction capabilities. "
        "The utilization of transfer learning significantly reduces computational requirements.",
        "Artificial intelligence systems demonstrate remarkable performance in classification tasks. "
        "The implementation of attention mechanisms enhances model accuracy substantially.",
        "The integration of multimodal data improves detection accuracy considerably. "
        "Each modality contributes complementary information to the classification pipeline.",
        "Gradient-based optimization methods converge efficiently on well-structured datasets. "
        "Regularization techniques prevent overfitting during the training process.",
        "Convolutional neural networks extract hierarchical features from visual data. "
        "Pooling operations reduce spatial dimensions while preserving essential information.",
        "Transformer architectures leverage self-attention to capture long-range dependencies. "
        "The encoder-decoder framework facilitates sequence-to-sequence learning tasks.",
        "Ensemble methods combine multiple weak learners to produce robust predictions. "
        "Cross-validation ensures reliable estimation of generalization performance.",
        "Recurrent networks model sequential dependencies in time-series data effectively. "
        "Gating mechanisms address the vanishing gradient problem in deep architectures.",
        "Autoencoders learn compressed representations of high-dimensional input data. "
        "The latent space captures the most salient features for downstream tasks.",
        "Generative adversarial networks produce realistic synthetic samples through adversarial training. "
        "The discriminator guides the generator toward increasingly realistic outputs.",
        "Reinforcement learning agents optimize policies through interaction with environments. "
        "Reward shaping accelerates learning in sparse reward settings considerably.",
        "Semi-supervised learning leverages unlabeled data to improve generalization performance. "
        "Pseudo-labeling iteratively assigns labels to high-confidence unlabeled examples.",
    ]
    fake = [
        "SHOCKING: Government scientists SECRETLY admit that 5G towers are designed to control your mind! "
        "They don't want you to know the truth. Share before this gets deleted!",
        "BREAKING: Leaked documents EXPOSE the global vaccine conspiracy that mainstream media is HIDING. "
        "Doctors are being silenced. Wake up people! The truth is being suppressed.",
        "You won't BELIEVE what they found on the moon! NASA is covering up alien structures. "
        "Exclusive footage proves everything. The deep state controls what you see!",
        "BOMBSHELL: The cure for cancer has existed for 30 years but Big Pharma is keeping it secret. "
        "Share this immediately before it gets censored by the globalist elites!",
        "EXPOSED: Elite politicians caught on camera in secret underground bunker planning world domination. "
        "The mainstream media refuses to cover this shocking revelation. Spread the truth!",
        "URGENT: Scientists reveal that chemtrails contain mind-control chemicals approved by secret government. "
        "Whistleblowers risk their lives to expose this horrifying conspiracy. Wake up!",
        "BREAKING NEWS: Billionaires are funding a secret project to implant microchips in the food supply. "
        "This is not a conspiracy theory — it is happening RIGHT NOW. Share before banned!",
        "SHOCKING REVELATION: The election was stolen using a satellite hack that changed millions of votes. "
        "Patriots are fighting back against the deep state corruption. Spread the word!",
        "EXCLUSIVE: Insider leaks proof that the moon landing was filmed in a Hollywood studio. "
        "Stanley Kubrick admitted everything on his deathbed. They are hiding the truth!",
        "BOMBSHELL REPORT: Drinking bleach cures all viruses according to suppressed medical research. "
        "Doctors who reveal this miracle cure are being threatened by pharmaceutical companies!",
        "BREAKING: Secret elite group controls all world governments through hidden financial networks. "
        "The truth is finally coming out. They tried to silence the whistleblowers but failed!",
        "SHOCKING EXCLUSIVE: Famous celebrity is actually a reptilian shapeshifter confirmed by leaked video. "
        "The evidence is undeniable. Share this before the social media giants delete it!",
        "URGENT WARNING: New law will make it ILLEGAL to own gold and cash by next month! "
        "The government wants to control every transaction you make. Prepare now before it is too late!",
        "EXPOSED: The sun is actually artificial and controlled by a secret space agency. "
        "Ancient texts confirm this hidden knowledge that they do not want you to discover!",
        "BREAKING: Water fluoridation is a mind control experiment proven by newly declassified CIA files. "
        "Thousands of scientists are being paid to keep this bombshell secret from the public!",
    ]

    rows = (
        [(t, 0) for t in human] +
        [(t, 1) for t in ai]    +
        [(t, 2) for t in fake]
    )
    import random
    random.shuffle(rows)
    return pd.DataFrame(rows, columns=['text','label'])


# ══════════════════════════════════════════════════════════════════════════════
#  MODULE 2 — IMAGE MODEL TRAINING
# ══════════════════════════════════════════════════════════════════════════════

def train_image(epochs=5, batch_size=16, lr=1e-3):

    banner("IMAGE MODEL TRAINING — MobileNetV2")

    try:

        import torch
        import torch.nn as nn
        import torchvision.models as models
        import torchvision.transforms as T

        from torch.utils.data import DataLoader
        from torchvision.datasets import ImageFolder

        from sklearn.metrics import classification_report

    except ImportError as e:

        err(f"Missing dependency: {e}")

        return

    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

    info(f"Device: {DEVICE}")

    TRAIN_DIR = 'datasets/images/train'
    VAL_DIR = 'datasets/images/val'

    # =====================================================
    # TRANSFORMS
    # =====================================================

    transform_train = T.Compose([

        T.Resize((224,224)),

        T.RandomHorizontalFlip(),

        T.RandomRotation(10),

        T.ToTensor(),

        T.Normalize(
            [0.485,0.456,0.406],
            [0.229,0.224,0.225]
        )
    ])

    transform_val = T.Compose([

        T.Resize((224,224)),

        T.ToTensor(),

        T.Normalize(
            [0.485,0.456,0.406],
            [0.229,0.224,0.225]
        )
    ])

    # =====================================================
    # DATASETS
    # =====================================================

    train_ds = ImageFolder(
        TRAIN_DIR,
        transform=transform_train
    )

    val_ds = ImageFolder(
        VAL_DIR,
        transform=transform_val
    )

    info(f"Classes: {train_ds.classes}")

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size
    )

    # =====================================================
    # MODEL
    # =====================================================

    model = models.mobilenet_v2(

        weights=models.MobileNet_V2_Weights.DEFAULT
    )

    model.classifier = nn.Sequential(

        nn.Dropout(0.3),

        nn.Linear(
            model.last_channel,
            len(train_ds.classes)
        )
    )

    model = model.to(DEVICE)

    criterion = nn.CrossEntropyLoss()

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=lr
    )

    best_acc = 0

    train_losses = []
    val_accs = []

    # =====================================================
    # TRAIN LOOP
    # =====================================================

    for epoch in range(epochs):

        model.train()

        running_loss = 0

        correct = 0
        total = 0

        for imgs, labels in train_loader:

            imgs = imgs.to(DEVICE)
            labels = labels.to(DEVICE)

            outputs = model(imgs)

            loss = criterion(outputs, labels)

            optimizer.zero_grad()

            loss.backward()

            optimizer.step()

            running_loss += loss.item()

            preds = outputs.argmax(1)

            correct += (preds == labels).sum().item()

            total += len(labels)

        train_acc = correct / total

        # ================= VALIDATION =================

        model.eval()

        v_correct = 0
        v_total = 0

        with torch.no_grad():

            for imgs, labels in val_loader:

                imgs = imgs.to(DEVICE)
                labels = labels.to(DEVICE)

                outputs = model(imgs)

                preds = outputs.argmax(1)

                v_correct += (preds == labels).sum().item()

                v_total += len(labels)

        val_acc = v_correct / v_total

        train_losses.append(running_loss)
        val_accs.append(val_acc)

        print(f"\nEpoch {epoch+1}/{epochs}")

        print(f"Loss: {running_loss:.4f}")

        print(f"Train Accuracy: {train_acc:.4f}")

        print(f"Validation Accuracy: {val_acc:.4f}")

        # ================= SAVE BEST MODEL =================

        if val_acc > best_acc:

            best_acc = val_acc

            save_model(
                model.state_dict(),
                'models/mobilenet_image.pt'
            )

            print("Best Model Saved!")

    print("\nTraining Complete!")

    print(f"Best Validation Accuracy: {best_acc:.4f}")


# ══════════════════════════════════════════════════════════════════════════════
#  MODULE 3 — AUDIO MODEL TRAINING
# ══════════════════════════════════════════════════════════════════════════════

def train_audio(epochs=30, batch_size=16, lr=1e-3):
    banner("AUDIO MODEL TRAINING  —  AudioCNN (MFCC Spectrogram)")

    try:
        import torch
        import torch.nn as nn
        import librosa
        import numpy as np
        from torch.utils.data import Dataset, DataLoader
        from sklearn.metrics import classification_report
        from audio_module.audio_analyzer import AudioCNN
    except ImportError as e:
        err(f"Missing dependency: {e}")
        return

    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    info(f"Device: {DEVICE}")

    # ── audio dataset class ───────────────────────────────────────────────────
    class AudioMFCCDataset(Dataset):
        """
        Loads WAV/MP3/FLAC audio files and converts them to
        fixed-length MFCC spectrograms (shape: 1 × n_mfcc × max_len).
        """
        LABEL_MAP = {'real': 0, 'ai': 1, 'cloned': 2}
        EXTS      = {'.wav', '.mp3', '.flac', '.ogg', '.m4a'}

        def __init__(self, root, n_mfcc=40, max_len=128, sr=22050, augment=False):
            self.n_mfcc  = n_mfcc
            self.max_len = max_len
            self.sr      = sr
            self.augment = augment
            self.samples = []

            for cls, lbl in self.LABEL_MAP.items():
                folder = os.path.join(root, cls)
                if not os.path.exists(folder):
                    warn(f"Folder not found: {folder}  (skipping '{cls}' class)")
                    continue
                files = [f for f in os.listdir(folder)
                         if os.path.splitext(f)[1].lower() in self.EXTS]
                for f in files:
                    self.samples.append((os.path.join(folder, f), lbl))

        def __len__(self): return len(self.samples)

        def __getitem__(self, idx):
            path, label = self.samples[idx]
            try:
                y, sr = librosa.load(path, sr=self.sr, mono=True)
            except Exception:
                y = np.zeros(self.sr, dtype=np.float32)
                sr = self.sr

            # Time-stretch augmentation
            if self.augment and np.random.rand() < 0.4:
                rate = np.random.uniform(0.85, 1.15)
                y    = librosa.effects.time_stretch(y, rate=rate)

            # MFCC
            mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=self.n_mfcc)

            # Pad or trim to fixed length
            if mfcc.shape[1] < self.max_len:
                mfcc = np.pad(mfcc, ((0,0),(0, self.max_len - mfcc.shape[1])),
                              mode='constant')
            else:
                mfcc = mfcc[:, :self.max_len]

            # Normalise per sample
            mean, std = mfcc.mean(), mfcc.std()
            mfcc = (mfcc - mean) / (std + 1e-9)

            return torch.tensor(mfcc, dtype=torch.float32).unsqueeze(0), label

    # ── check dataset ─────────────────────────────────────────────────────────
    section("Checking Dataset")
    AUDIO_ROOT = 'datasets/audio'

    if not os.path.exists(AUDIO_ROOT):
        warn(f"Audio dataset not found at {AUDIO_ROOT}")
        info("Expected folder structure:")
        info("  datasets/audio/real/     ← real human voice WAV/MP3 files")
        info("  datasets/audio/ai/       ← TTS / AI-synthesised voice files")
        info("  datasets/audio/cloned/   ← voice-cloned audio files")
        info("")
        info("Recommended datasets:")
        info("  Real voice  : LJSpeech  — https://keithito.com/LJ-Speech-Dataset/")
        info("  AI voice    : LibriTTS (TTS samples) or Mozilla TTS output")
        info("  Anti-spoof  : ASVspoof 2019 — https://www.asvspoof.org/")
        info("")
        warn("Creating silent demo audio files for smoke-testing...")
        _make_demo_audio_dataset()

    # ── load datasets ─────────────────────────────────────────────────────────
    section("Loading Audio Datasets")
    full_ds = AudioMFCCDataset(AUDIO_ROOT, n_mfcc=40, max_len=128, augment=True)

    if len(full_ds) == 0:
        err("No audio files found in datasets/audio/. "
            "Add .wav/.mp3 files to real/, ai/, and cloned/ subfolders.")
        return

    info(f"Total audio samples: {len(full_ds)}")

    # Train/val split
    n_train = int(0.80 * len(full_ds))
    n_val   = len(full_ds) - n_train
    train_ds, val_ds = torch.utils.data.random_split(full_ds, [n_train, n_val])
    info(f"Train: {n_train}  |  Val: {n_val}")

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,  num_workers=2)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False, num_workers=2)

    # ── model ─────────────────────────────────────────────────────────────────
    section("Building AudioCNN")
    model     = AudioCNN(n_mfcc=40, num_classes=3).to(DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    criterion = nn.CrossEntropyLoss()

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    info(f"Trainable parameters: {trainable:,}")

    # ── training loop ─────────────────────────────────────────────────────────
    section(f"Training  ({epochs} epochs, batch={batch_size})")
    best_val_acc = 0.0
    train_losses, val_accs = [], []

    for epoch in range(1, epochs+1):
        # ── train ──
        model.train()
        total_loss = correct = total = 0
        for mfcc_batch, labels in train_loader:
            mfcc_batch = mfcc_batch.to(DEVICE)
            labels     = labels.to(DEVICE)
            logits     = model(mfcc_batch)
            loss       = criterion(logits, labels)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total_loss += loss.item()
            correct    += (logits.argmax(1) == labels).sum().item()
            total      += len(labels)

        scheduler.step()
        train_acc = correct / max(total, 1)
        avg_loss  = total_loss / max(len(train_loader), 1)

        # ── validate ──
        model.eval()
        v_correct = v_total = 0
        with torch.no_grad():
            for mfcc_batch, labels in val_loader:
                mfcc_batch = mfcc_batch.to(DEVICE)
                labels     = labels.to(DEVICE)
                preds      = model(mfcc_batch).argmax(1)
                v_correct += (preds == labels).sum().item()
                v_total   += len(labels)

        val_acc = v_correct / max(v_total, 1)
        train_losses.append(avg_loss)
        val_accs.append(val_acc)

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            save_model(model.state_dict(), 'models/audio_cnn.pt')

        if epoch % 5 == 0 or epoch == 1:
            print(f"  Epoch {epoch:3d}/{epochs}  |  Loss: {avg_loss:.4f}  |  "
                  f"Train Acc: {train_acc:.4f}  |  Val Acc: {val_acc:.4f}  |  Best: {best_val_acc:.4f}")

    # ── final evaluation ──────────────────────────────────────────────────────
    section("Final Evaluation")
    model.eval()
    all_preds, all_true = [], []
    with torch.no_grad():
        for mfcc_batch, labels in val_loader:
            preds = model(mfcc_batch.to(DEVICE)).argmax(1).cpu().numpy()
            all_preds.extend(preds)
            all_true.extend(labels.numpy())

    print("\n" + classification_report(
        all_true, all_preds,
        target_names=['Real Voice','AI-Generated Voice','Cloned Voice'],
        zero_division=0
    ))

    plot_history(train_losses, val_accs,
                 'AudioCNN — Voice Classifier Training',
                 'models/audio_training_curve.png')

    ok(f"Best validation accuracy: {best_val_acc:.4f} ({best_val_acc*100:.2f}%)")
    ok("Audio model training complete!")


def _make_demo_audio_dataset():
    """Creates tiny silent WAV files for smoke-testing the audio pipeline."""
    try:
        import soundfile as sf
    except ImportError:
        try:
            import scipy.io.wavfile as wavfile
            _sf = None
        except ImportError:
            warn("soundfile / scipy not installed — demo audio creation skipped")
            return

    SR = 22050
    classes = ['real', 'ai', 'cloned']
    for cls in classes:
        path = os.path.join('datasets', 'audio', cls)
        os.makedirs(path, exist_ok=True)
        for i in range(10):
            # Generate a simple sine wave at different frequencies
            freq   = 220 * (1 + classes.index(cls)) + i * 10
            t      = np.linspace(0, 1.0, SR, dtype=np.float32)
            signal = (0.3 * np.sin(2 * np.pi * freq * t)).astype(np.float32)
            out    = os.path.join(path, f'{cls}_{i:04d}.wav')
            try:
                import soundfile as sf
                sf.write(out, signal, SR)
            except Exception:
                try:
                    import scipy.io.wavfile as wf
                    wf.write(out, SR, (signal * 32767).astype(np.int16))
                except Exception:
                    pass
    ok("Demo audio dataset created (replace with real ASVspoof / LJSpeech data!)")


# ══════════════════════════════════════════════════════════════════════════════
#  ENTRY POINT
# ══════════════════════════════════════════════════════════════════════════════

USAGE = """
VERITAS AI — Training Script
Usage:
  python training/train.py text     Train HybridMLP text classifier
  python training/train.py image    Fine-tune ResNet50 image classifier
  python training/train.py audio    Train AudioCNN voice classifier
  python training/train.py all      Train all three models sequentially

Optional flags (append after mode):
  --epochs N       Number of training epochs  (default: 25/20/30)
  --batch N        Batch size                 (default: 32/32/16)
  --lr F           Learning rate              (default: 2e-4/1e-3/1e-3)

Examples:
  python training/train.py text  --epochs 50 --batch 16 --lr 1e-4
  python training/train.py image --epochs 30
  python training/train.py all
"""

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument('mode', nargs='?', default='help',
                        choices=['text','image','audio','all','help'])
    parser.add_argument('--epochs', type=int, default=None)
    parser.add_argument('--batch',  type=int, default=None)
    parser.add_argument('--lr',     type=float, default=None)
    args = parser.parse_args()

    if args.mode == 'help' or args.mode is None:
        print(USAGE)
        sys.exit(0)

    start = time.time()

    if args.mode in ('text', 'all'):
        train_text(
            epochs     = args.epochs or 25,
            batch_size = args.batch  or 32,
            lr         = args.lr     or 2e-4,
        )

    if args.mode in ('image', 'all'):
        train_image(
            epochs     = args.epochs or 20,
            batch_size = args.batch  or 32,
            lr         = args.lr     or 1e-3,
        )

    if args.mode in ('audio', 'all'):
        train_audio(
            epochs     = args.epochs or 30,
            batch_size = args.batch  or 16,
            lr         = args.lr     or 1e-3,
        )

    elapsed = time.time() - start
    print(f"\n{'═'*58}")
    print(f"  All done!  Total time: {elapsed/60:.1f} min")
    print(f"  Saved models → models/")
    print(f"{'═'*58}\n")