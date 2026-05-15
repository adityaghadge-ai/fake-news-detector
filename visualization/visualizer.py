"""
VISUALIZATION MODULE
====================
All plots returned as base64-encoded PNG strings (dark theme).
Functions:
  plot_confidence()        — horizontal bar chart
  plot_stylometric_radar() — radar / spider chart
  plot_sentence_lengths()  — sentence-length bar chart
  plot_word_freq()         — top-N word frequency bar
  plot_dashboard()         — 2×2 summary panel
"""

import numpy as np
import io
import base64
import re
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec
from collections import Counter

# ── Dark theme palette ─────────────────────────────────────────
BG      = '#0f1117'
SURFACE = '#131826'
BORDER  = '#1e2438'
VIOLET  = '#7c6ff7'
CORAL   = '#ff6b6b'
GREEN   = '#39e8a6'
YELLOW  = '#ffd166'
CYAN    = '#00d4ff'
TEXT    = '#dde1f0'
TEXT2   = '#8890aa'
GRID    = '#1e2438'

ACCENT_CYCLE = [VIOLET, CORAL, GREEN, YELLOW, CYAN]


def _theme(fig, axes):
    """Apply dark theme to figure and a list of axes."""
    fig.patch.set_facecolor(BG)
    for ax in axes:
        ax.set_facecolor(SURFACE)
        ax.tick_params(colors=TEXT2, labelsize=8)
        ax.xaxis.label.set_color(TEXT2)
        ax.yaxis.label.set_color(TEXT2)
        ax.title.set_color(TEXT)
        for sp in ax.spines.values():
            sp.set_edgecolor(BORDER)
        ax.grid(True, color=GRID, alpha=0.6, linestyle='--', linewidth=0.5)


def _to_b64(fig) -> str:
    buf = io.BytesIO()
    fig.savefig(buf, format='png', dpi=110, bbox_inches='tight',
                facecolor=fig.get_facecolor())
    plt.close(fig)
    buf.seek(0)
    return base64.b64encode(buf.read()).decode()


# ──────────────────────────────────────────────────────────────
#  1. CONFIDENCE BAR CHART
# ──────────────────────────────────────────────────────────────
def plot_confidence(scores: dict, title: str = "Classification Confidence") -> str:
    labels = list(scores.keys())
    values = list(scores.values())
    colors = ACCENT_CYCLE[:len(labels)]

    fig, ax = plt.subplots(figsize=(7, 3.2))
    _theme(fig, [ax])

    bars = ax.barh(labels, values, color=colors, edgecolor='none', height=0.45)
    for bar, val in zip(bars, values):
        ax.text(min(val + 1.5, 96), bar.get_y() + bar.get_height()/2,
                f'{val:.1f}%', va='center', color=TEXT, fontsize=9, fontweight='bold')

    ax.set_xlim(0, 105)
    ax.set_xlabel('Confidence (%)', fontsize=8)
    ax.set_title(title, fontsize=10, fontweight='bold')
    ax.invert_yaxis()
    plt.tight_layout()
    return _to_b64(fig)


# ──────────────────────────────────────────────────────────────
#  2. STYLOMETRIC RADAR CHART
# ──────────────────────────────────────────────────────────────
def plot_stylometric_radar(features: dict) -> str:
    MAX_VALS = {
        "lexical_diversity":        1.0,
        "sentence_length_variance": 200.0,
        "repetition_score":         1.0,
        "burstiness":               1.0,
        "readability_score":        100.0,
        "avg_word_length":          15.0,
        "unique_bigram_ratio":      1.0,
        "punctuation_density":      0.10,
    }
    keys   = [k for k in features if k in MAX_VALS][:8]
    if len(keys) < 3:
        return ""

    norms  = [min(abs(float(features[k])) / MAX_VALS[k], 1.0) for k in keys]
    labels = [k.replace('_', ' ').title() for k in keys]
    N      = len(keys)
    angles = np.linspace(0, 2*np.pi, N, endpoint=False).tolist()
    angles += angles[:1]
    vals   = norms + norms[:1]

    fig, ax = plt.subplots(figsize=(5.5, 5.5), subplot_kw=dict(polar=True))
    fig.patch.set_facecolor(BG)
    ax.set_facecolor(SURFACE)
    ax.plot(angles, vals, color=VIOLET, linewidth=2)
    ax.fill(angles, vals, color=VIOLET, alpha=0.22)
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(labels, color=TEXT, fontsize=7.5)
    ax.set_yticks([0.2, 0.4, 0.6, 0.8])
    ax.set_yticklabels(['0.2','0.4','0.6','0.8'], color=TEXT2, fontsize=6)
    ax.grid(color=GRID, alpha=0.7)
    ax.set_title('Stylometric Feature Radar', color=TEXT, fontsize=10, fontweight='bold', pad=15)
    plt.tight_layout()
    return _to_b64(fig)


# ──────────────────────────────────────────────────────────────
#  3. SENTENCE LENGTH DISTRIBUTION
# ──────────────────────────────────────────────────────────────
def plot_sentence_lengths(text: str) -> str:
    sents   = [s.strip() for s in re.split(r'[.!?]+', text) if len(s.strip()) > 3]
    lengths = [len(s.split()) for s in sents]
    if len(lengths) < 2:
        return ""

    fig, ax = plt.subplots(figsize=(8, 3))
    _theme(fig, [ax])
    ax.bar(range(len(lengths)), lengths, color=GREEN, alpha=0.8, edgecolor='none', width=0.75)
    mean = np.mean(lengths)
    ax.axhline(mean, color=CORAL, linestyle='--', linewidth=1.5, label=f'Mean: {mean:.1f}')
    ax.set_xlabel('Sentence #', fontsize=8)
    ax.set_ylabel('Word Count', fontsize=8)
    ax.set_title('Sentence Length Distribution', fontsize=10, fontweight='bold')
    ax.legend(facecolor=SURFACE, edgecolor=BORDER, labelcolor=TEXT, fontsize=8)
    plt.tight_layout()
    return _to_b64(fig)


# ──────────────────────────────────────────────────────────────
#  4. WORD FREQUENCY
# ──────────────────────────────────────────────────────────────
STOPWORDS = {
    'the','a','an','and','or','but','in','on','at','to','for','of',
    'with','is','was','it','that','this','are','be','have','had',
    'has','they','their','from','by','not','as','we','you','he','she',
}

def plot_word_freq(text: str, top_n: int = 15) -> str:
    words = re.findall(r'\b[a-z]{4,}\b', text.lower())
    words = [w for w in words if w not in STOPWORDS]
    freq  = Counter(words).most_common(top_n)
    if not freq:
        return ""

    labs  = [f[0] for f in freq]
    cnts  = [f[1] for f in freq]
    colors = [VIOLET if c == max(cnts) else CYAN for c in cnts]

    fig, ax = plt.subplots(figsize=(8, 3.5))
    _theme(fig, [ax])
    ax.bar(labs, cnts, color=colors, edgecolor='none')
    ax.set_xticklabels(labs, rotation=38, ha='right', fontsize=8)
    ax.set_ylabel('Frequency', fontsize=8)
    ax.set_title('Top Word Frequencies', fontsize=10, fontweight='bold')
    plt.tight_layout()
    return _to_b64(fig)


# ──────────────────────────────────────────────────────────────
#  5. SUMMARY DASHBOARD  (2×2 panel)
# ──────────────────────────────────────────────────────────────
def plot_dashboard(result: dict, modality: str = 'text') -> str:
    fig = plt.figure(figsize=(12, 5.5))
    fig.patch.set_facecolor(BG)
    gs  = GridSpec(2, 2, figure=fig, hspace=0.52, wspace=0.38)

    scores = result.get('scores', {})
    label  = result.get('label', '?')
    conf   = result.get('confidence', 0)
    risk   = result.get('risk_level', 'Low')

    # ── Panel 1: Donut ─────────────────────────────────────────
    ax1 = fig.add_subplot(gs[0, 0])
    if scores:
        vals   = list(scores.values())
        lbls   = list(scores.keys())
        colors = ACCENT_CYCLE[:len(vals)]
        wedges, _, autotexts = ax1.pie(
            vals, colors=colors, autopct='%1.1f%%', startangle=90,
            wedgeprops=dict(width=0.52, edgecolor=BG, linewidth=2),
            textprops=dict(color=TEXT, fontsize=7),
        )
        for at in autotexts: at.set_fontsize(7)
        ax1.set_facecolor(SURFACE)
        patches = [mpatches.Patch(color=c, label=l) for c,l in zip(colors,lbls)]
        ax1.legend(handles=patches, fontsize=6, facecolor=SURFACE,
                   edgecolor=BORDER, labelcolor=TEXT, loc='lower center',
                   bbox_to_anchor=(0.5,-0.12), ncol=1)
    ax1.set_title('Class Distribution', color=TEXT, fontsize=9)

    # ── Panel 2: Confidence gauge ──────────────────────────────
    ax2 = fig.add_subplot(gs[0, 1])
    _theme(fig, [ax2])
    bar_color = CORAL if label not in ("Human-Written","Real Voice","Real Video","Real Image") else GREEN
    ax2.barh([''], [conf],       color=bar_color, height=0.5)
    ax2.barh([''], [100 - conf], left=[conf], color=BORDER, height=0.5)
    ax2.set_xlim(0, 100)
    ax2.set_xlabel('Confidence %', fontsize=8)
    ax2.set_title(f'Prediction: {label}', fontsize=9)
    ax2.text(conf/2, 0, f'{conf:.1f}%', ha='center', va='center',
             color='white', fontsize=11, fontweight='bold')

    # ── Panel 3: Risk level ────────────────────────────────────
    ax3 = fig.add_subplot(gs[1, 0])
    _theme(fig, [ax3])
    risk_colors = {'Low': GREEN, 'Medium': YELLOW, 'High': CORAL}
    for r, v in [('Low',1),('Medium',2),('High',3)]:
        ax3.bar([r], [v],
                color=risk_colors[r] if r==risk else BORDER,
                alpha=1.0 if r==risk else 0.25, edgecolor='none')
    ax3.set_ylim(0, 4)
    ax3.set_title('Risk Level', fontsize=9)

    # ── Panel 4: Explanation ───────────────────────────────────
    ax4 = fig.add_subplot(gs[1, 1])
    ax4.set_facecolor(SURFACE)
    ax4.axis('off')
    ax4.set_title('Key Findings', color=TEXT, fontsize=9)
    exps = result.get('explanation', [])
    body = '\n'.join(
        f"• {e[:62]}..." if len(e) > 62 else f"• {e}" for e in exps[:3]
    )
    ax4.text(0.04, 0.88, body, transform=ax4.transAxes, color=TEXT2,
             fontsize=7, va='top', family='monospace',
             wrap=True, linespacing=1.55)

    return _to_b64(fig)