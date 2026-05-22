"""
VERITAS AI — Flask Backend with Authentication
===============================================
Routes:
  GET/POST /login      Login + Register page
  POST     /register   Process registration
  GET      /logout     Logout
  GET      /dashboard  User dashboard + history
  GET      /           Main detector UI (login required)
  POST     /api/analyze/text|image|audio|video
  POST     /api/report
  GET      /api/health
"""

import os, sys, tempfile, traceback
from pathlib import Path

from flask import (Flask, request, jsonify, render_template,
                   send_file, session, redirect, url_for, flash)
from flask_cors import CORS
from werkzeug.utils import secure_filename

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))

app = Flask(__name__)
app.secret_key = os.environ.get('SECRET_KEY', 'veritas-ai-dev-secret-key-2024')
CORS(app)
app.config['MAX_CONTENT_LENGTH'] = 300 * 1024 * 1024

from auth import (init_db, close_db, login_required, current_user,
                  register_user, login_user, get_user_history,
                  get_user_stats, save_analysis)

with app.app_context():
    init_db()

app.teardown_appcontext(close_db)

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


def build_plots(result, modality, raw_text=''):
    from visualization.visualizer import (
        plot_confidence, plot_dashboard,
        plot_stylometric_radar, plot_sentence_lengths, plot_word_freq,
    )
    plots = {}
    plots['confidence'] = plot_confidence(result.get('scores', {}),
                                           f'{modality.title()} Confidence')
    plots['dashboard']  = plot_dashboard(result, modality)
    if modality == 'text' and raw_text:
        plots['radar']        = plot_stylometric_radar(result.get('stylometric_features', {}))
        plots['sent_lengths'] = plot_sentence_lengths(raw_text)
        plots['word_freq']    = plot_word_freq(raw_text)
    return plots


@app.context_processor
def inject_user():
    return {'current_user': current_user()}


# ── AUTH ROUTES ───────────────────────────────────────────────

@app.route('/login', methods=['GET', 'POST'])
def login():
    if 'user_id' in session:
        return redirect(url_for('index'))

    mode = request.args.get('mode', 'login')

    if request.method == 'POST':
        form_type = request.form.get('form_type', 'login')

        if form_type == 'login':
            ue       = request.form.get('username_or_email', '').strip()
            password = request.form.get('password', '')
            if not ue or not password:
                flash('Please fill in all fields.', 'error')
                return render_template('login.html', mode='login')
            success, message, user = login_user(ue, password)
            if success:
                session.permanent = True
                session['user_id']  = user['id']
                session['username'] = user['username']
                flash(f'Welcome back, {user["username"]}!', 'success')
                nxt = request.args.get('next', url_for('index'))
                return redirect(nxt)
            else:
                flash(message, 'error')
                return render_template('login.html', mode='login')

    return render_template('login.html', mode=mode)


@app.route('/register', methods=['POST'])
def register():
    username  = request.form.get('username', '').strip()
    email     = request.form.get('email', '').strip()
    password  = request.form.get('password', '')
    confirm   = request.form.get('confirm_password', '')
    agree     = request.form.get('agree')

    if not agree:
        flash('You must agree to the Terms of Use.', 'error')
        return render_template('login.html', mode='register')
    if password != confirm:
        flash('Passwords do not match.', 'error')
        return render_template('login.html', mode='register')

    success, message = register_user(username, email, password)
    if success:
        _, _, user = login_user(username, password)
        if user:
            session['user_id']  = user['id']
            session['username'] = user['username']
            flash(f'Welcome to VERITAS AI, {username}!', 'success')
            return redirect(url_for('index'))
    else:
        flash(message, 'error')
        return render_template('login.html', mode='register')


@app.route('/logout')
def logout():
    session.clear()
    flash('You have been logged out successfully.', 'success')
    return redirect(url_for('login'))


@app.route('/dashboard')
@login_required
def dashboard():
    user    = current_user()
    history = get_user_history(session['user_id'], limit=15)
    stats   = get_user_stats(session['user_id'])
    return render_template('dashboard.html', user=user, history=history, stats=stats)


@app.route('/forgot')
def forgot():
    flash('Password reset is not available in this demo. Contact your admin.', 'error')
    return redirect(url_for('login'))


# ── MAIN ROUTE ────────────────────────────────────────────────

@app.route('/')
@login_required
def index():
    return render_template('index.html', user=current_user())


# ── ANALYSIS API ──────────────────────────────────────────────

@app.route('/api/analyze/text', methods=['POST'])
@login_required
def analyze_text():
    try:
        data = request.get_json(force=True)
        text = (data.get('text') or '').strip()
        if len(text) < 20:
            return jsonify({'error': 'Text must be at least 20 characters.'}), 400
        result = get('text').analyze(text)
        result['plots'] = build_plots(result, 'text', raw_text=text)
        save_analysis(session['user_id'], 'text', result.get('label','?'),
                      result.get('confidence',0), result.get('risk_level','Low'),
                      f'{len(text)} chars')
        return jsonify(result)
    except Exception:
        traceback.print_exc()
        return jsonify({'error': 'Text analysis failed.'}), 500


@app.route('/api/analyze/image', methods=['POST'])
@login_required
def analyze_image():
    tmp = None
    try:
        if 'file' not in request.files or not request.files['file'].filename:
            return jsonify({'error': 'No image file received.'}), 400
        f   = request.files['file']
        sfx = Path(secure_filename(f.filename)).suffix or '.jpg'
        tmp = tempfile.NamedTemporaryFile(delete=False, suffix=sfx)
        f.save(tmp.name)
        result = get('image').analyze(tmp.name)
        result['plots'] = build_plots(result, 'image')
        save_analysis(session['user_id'], 'image', result.get('label','?'),
                      result.get('confidence',0), result.get('risk_level','Low'),
                      secure_filename(f.filename))
        return jsonify(result)
    except Exception:
        traceback.print_exc()
        return jsonify({'error': 'Image analysis failed.'}), 500
    finally:
        if tmp is not None:
            try:
                tmp.close()
            except:
                pass
            try:
                os.unlink(tmp.name)
            except:
                pass


@app.route('/api/analyze/audio', methods=['POST'])
@login_required
def analyze_audio():
    try:
        if 'file' not in request.files or not request.files['file'].filename:
            return jsonify({'error': 'No audio file received.'}), 400
        f   = request.files['file']
        sfx = Path(secure_filename(f.filename)).suffix or '.wav'
        tmp = tempfile.NamedTemporaryFile(delete=False, suffix=sfx)
        f.save(tmp.name)
        result = get('audio').analyze(tmp.name)
        result['plots'] = build_plots(result, 'audio')
        os.unlink(tmp.name)
        save_analysis(session['user_id'], 'audio', result.get('label','?'),
                      result.get('confidence',0), result.get('risk_level','Low'),
                      secure_filename(f.filename))
        return jsonify(result)
    except Exception:
        traceback.print_exc()
        return jsonify({'error': 'Audio analysis failed.'}), 500


@app.route('/api/analyze/video', methods=['POST'])
@login_required
def analyze_video():
    try:
        if 'file' not in request.files or not request.files['file'].filename:
            return jsonify({'error': 'No video file received.'}), 400
        f   = request.files['file']
        sfx = Path(secure_filename(f.filename)).suffix or '.mp4'
        tmp = tempfile.NamedTemporaryFile(delete=False, suffix=sfx)
        f.save(tmp.name)
        result = get('video').analyze(tmp.name)
        result['plots'] = build_plots(result, 'video')
        os.unlink(tmp.name)
        save_analysis(session['user_id'], 'video', result.get('label','?'),
                      result.get('confidence',0), result.get('risk_level','Low'),
                      secure_filename(f.filename))
        return jsonify(result)
    except Exception:
        traceback.print_exc()
        return jsonify({'error': 'Video analysis failed.'}), 500


@app.route('/api/report', methods=['POST'])
@login_required
def generate_report():
    try:
        data     = request.get_json(force=True)
        result   = data.get('result', {})
        modality = data.get('modality', 'text')
        filename = data.get('filename', 'input')
        plots    = data.get('plots', [])
        from reporting.report_generator import ReportGenerator
        pdf_bytes = ReportGenerator().generate(result, modality, plots, filename)
        tmp = tempfile.NamedTemporaryFile(delete=False, suffix='.pdf')
        tmp.write(pdf_bytes); tmp.close()
        return send_file(tmp.name, mimetype='application/pdf',
                         as_attachment=True,
                         download_name=f'veritas_report_{modality}.pdf')
    except Exception:
        traceback.print_exc()
        return jsonify({'error': 'Report generation failed.'}), 500


@app.route('/api/health')
def health():
    u = current_user()
    return jsonify({'status':'ok','authenticated': u is not None,
                    'user': u['username'] if u else None})


if __name__ == '__main__':
    print("""
  ╔══════════════════════════════════════════════╗
  ║   VERITAS AI — With Authentication           ║
  ║   http://127.0.0.1:5000                      ║
  ╚══════════════════════════════════════════════╝
    """)
    app.run(debug=True, port=5000, host='0.0.0.0')