"""
VERITAS AI — Flask Backend
==========================
Routes:
  GET  /                     → Main UI
  POST /api/analyze/text     → Text analysis
  POST /api/analyze/image    → Image analysis
  POST /api/analyze/audio    → Audio analysis
  POST /api/analyze/video    → Video analysis
  POST /api/report           → PDF report download
  GET  /api/health           → Health check
"""

import os
import sys
import json
import tempfile
import traceback
from pathlib import Path

from flask import Flask, request, jsonify, render_template, send_file
from flask_cors import CORS
from werkzeug.utils import secure_filename

# ── path setup ─────────────────────────────────────────────────
ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))

app = Flask(__name__)
CORS(app)
app.config['MAX_CONTENT_LENGTH'] = 300 * 1024 * 1024   # 300 MB
app.config['UPLOAD_FOLDER']      = tempfile.gettempdir()

# ── lazy-load analyzers (avoids long cold-start) ───────────────
_analyzers = {}

def get(name):
    if name not in _analyzers:
        if name == 'text':
            from text_module.text_analyzer import TextAnalyzer
            _analyzers[name] = TextAnalyzer()
        elif name == 'image':
            from image_module.image_analyzer import ImageAnalyzer
            _analyzers[name] = ImageAnalyzer()
        elif name == 'audio':
            from audio_module.audio_analyzer import AudioAnalyzer
            _analyzers[name] = AudioAnalyzer()
        elif name == 'video':
            from video_module.video_analyzer import VideoAnalyzer
            _analyzers[name] = VideoAnalyzer()
    return _analyzers[name]


# ────────────────────────────────────────────────────────────────
#  HELPER — generate all plots for a result
# ────────────────────────────────────────────────────────────────
def build_plots(result: dict, modality: str, raw_text: str = "") -> dict:
    from visualization.visualizer import (
        plot_confidence, plot_dashboard,
        plot_stylometric_radar, plot_sentence_lengths, plot_word_freq,
    )
    plots = {}
    plots["confidence"] = plot_confidence(result.get("scores", {}),
                                           f"{modality.title()} Classification Confidence")
    plots["dashboard"]  = plot_dashboard(result, modality)

    if modality == "text" and raw_text:
        plots["radar"]        = plot_stylometric_radar(result.get("stylometric_features", {}))
        plots["sent_lengths"] = plot_sentence_lengths(raw_text)
        plots["word_freq"]    = plot_word_freq(raw_text)
    return plots


# ────────────────────────────────────────────────────────────────
#  ROUTES
# ────────────────────────────────────────────────────────────────

@app.route('/')
def index():
    return render_template('index.html')


@app.route('/api/analyze/text', methods=['POST'])
def analyze_text():
    try:
        data = request.get_json(force=True)
        text = (data.get('text') or '').strip()
        if len(text) < 20:
            return jsonify({"error": "Text must be at least 20 characters."}), 400

        result         = get('text').analyze(text)
        result["plots"] = build_plots(result, "text", raw_text=text)
        return jsonify(result)

    except Exception:
        traceback.print_exc()
        return jsonify({"error": "Text analysis failed. Check server logs."}), 500


@app.route('/api/analyze/image', methods=['POST'])
def analyze_image():
    try:
        if 'file' not in request.files or request.files['file'].filename == '':
            return jsonify({"error": "No image file received."}), 400

        f      = request.files['file']
        suffix = Path(secure_filename(f.filename)).suffix or '.jpg'
        tmp    = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
        f.save(tmp.name)

        result          = get('image').analyze(tmp.name)
        result["plots"] = build_plots(result, "image")
        os.unlink(tmp.name)
        return jsonify(result)

    except Exception:
        traceback.print_exc()
        return jsonify({"error": "Image analysis failed. Check server logs."}), 500


@app.route('/api/analyze/audio', methods=['POST'])
def analyze_audio():
    try:
        if 'file' not in request.files or request.files['file'].filename == '':
            return jsonify({"error": "No audio file received."}), 400

        f      = request.files['file']
        suffix = Path(secure_filename(f.filename)).suffix or '.wav'
        tmp    = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
        f.save(tmp.name)

        result          = get('audio').analyze(tmp.name)
        result["plots"] = build_plots(result, "audio")
        os.unlink(tmp.name)
        return jsonify(result)

    except Exception:
        traceback.print_exc()
        return jsonify({"error": "Audio analysis failed. Check server logs."}), 500


@app.route('/api/analyze/video', methods=['POST'])
def analyze_video():
    try:
        if 'file' not in request.files or request.files['file'].filename == '':
            return jsonify({"error": "No video file received."}), 400

        f      = request.files['file']
        suffix = Path(secure_filename(f.filename)).suffix or '.mp4'
        tmp    = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
        f.save(tmp.name)

        result          = get('video').analyze(tmp.name)
        result["plots"] = build_plots(result, "video")
        os.unlink(tmp.name)
        return jsonify(result)

    except Exception:
        traceback.print_exc()
        return jsonify({"error": "Video analysis failed. Check server logs."}), 500


@app.route('/api/report', methods=['POST'])
def generate_report():
    try:
        data      = request.get_json(force=True)
        result    = data.get("result", {})
        modality  = data.get("modality", "text")
        filename  = data.get("filename", "input")
        plots     = data.get("plots", [])

        from reporting.report_generator import ReportGenerator
        pdf_bytes = ReportGenerator().generate(result, modality, plots, filename)

        tmp = tempfile.NamedTemporaryFile(delete=False, suffix='.pdf')
        tmp.write(pdf_bytes); tmp.close()

        return send_file(
            tmp.name,
            mimetype='application/pdf',
            as_attachment=True,
            download_name=f'veritas_report_{modality}.pdf',
        )
    except Exception:
        traceback.print_exc()
        return jsonify({"error": "Report generation failed. Check server logs."}), 500


@app.route('/api/health')
def health():
    return jsonify({"status": "ok", "version": "1.0", "name": "VERITAS AI"})


# ────────────────────────────────────────────────────────────────
if __name__ == '__main__':
    banner = """
  ╔══════════════════════════════════════════════╗
  ║   VERITAS AI — Fake & AI Content Detector   ║
  ║   http://127.0.0.1:5000                      ║
  ╚══════════════════════════════════════════════╝
    """
    print(banner)
    app.run(debug=True, port=5000, host='0.0.0.0')