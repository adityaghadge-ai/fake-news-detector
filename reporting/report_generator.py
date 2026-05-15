"""
PDF REPORT GENERATOR
====================
Generates a downloadable, styled PDF report for each modality analysis.
Uses ReportLab — no LaTeX or external dependencies.
"""

import io
import base64
import datetime


class ReportGenerator:

    def generate(self, result: dict, modality: str,
                 plot_b64s: list = None,
                 input_filename: str = "input") -> bytes:
        try:
            from reportlab.lib.pagesizes import A4
            from reportlab.lib import colors
            from reportlab.lib.units import cm
            from reportlab.platypus import (
                SimpleDocTemplate, Paragraph, Spacer, Table,
                TableStyle, Image as RLImage, HRFlowable, PageBreak
            )
            from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
            from reportlab.lib.enums import TA_CENTER
        except ImportError:
            return b"[ERROR] ReportLab not installed. Run: pip install reportlab"

        buf = io.BytesIO()
        doc = SimpleDocTemplate(buf, pagesize=A4,
                                 rightMargin=2*cm, leftMargin=2*cm,
                                 topMargin=2*cm, bottomMargin=2*cm)
        styles = getSampleStyleSheet()

        # ── Custom styles ──────────────────────────────────────
        S = {
            'title': ParagraphStyle('t', parent=styles['Title'],
                                     fontSize=20, spaceAfter=4,
                                     textColor=colors.HexColor('#6C63FF')),
            'h1':    ParagraphStyle('h', parent=styles['Heading1'],
                                     fontSize=12, spaceAfter=4,
                                     textColor=colors.HexColor('#1a1d2e')),
            'body':  ParagraphStyle('b', parent=styles['Normal'],
                                     fontSize=9, leading=14, spaceAfter=5,
                                     textColor=colors.HexColor('#1a1d2e')),
            'mono':  ParagraphStyle('m', parent=styles['Code'],
                                     fontSize=8, leading=12,
                                     backColor=colors.HexColor('#f4f4f8')),
            'small': ParagraphStyle('s', parent=styles['Normal'],
                                     fontSize=7, leading=11,
                                     textColor=colors.HexColor('#888899')),
        }
        PRP = '#6C63FF'
        story = []

        # ── Header ────────────────────────────────────────────
        story.append(Paragraph("VERITAS AI", S['title']))
        story.append(Paragraph(
            "Explainable Multi-Modal Fake &amp; AI Content Detection Framework — Analysis Report",
            S['h1']))
        story.append(HRFlowable(width="100%", thickness=2, color=colors.HexColor(PRP)))
        story.append(Spacer(1, 10))

        # ── Meta table ────────────────────────────────────────
        meta = [
            ["Report Generated",  datetime.datetime.now().strftime("%Y-%m-%d  %H:%M:%S")],
            ["Analysis Modality", modality.upper()],
            ["Input File",        input_filename],
            ["Framework",         "DistilBERT + ResNet50 + Librosa + OpenCV + GradCAM"],
            ["Version",           "VERITAS AI v1.0  —  Research / Educational Use"],
        ]
        story.append(self._table(meta, [4.5*cm, 12*cm], header=False))
        story.append(Spacer(1, 14))

        # ── Prediction result ─────────────────────────────────
        label = result.get("label", "Unknown")
        conf  = result.get("confidence", 0)
        risk  = result.get("risk_level", "?")
        story.append(Paragraph("CLASSIFICATION RESULT", S['h1']))
        res_data = [
            ["Prediction Label", label],
            ["Confidence Score", f"{conf:.2f}%"],
            ["Risk Level",       risk],
            ["Word / File Info", f"Words: {result.get('word_count','—')}  |  "
                                  f"Chars: {result.get('char_count','—')}"],
        ]
        story.append(self._table(res_data, [5*cm, 11.5*cm], header=False,
                                  bg='#eeeeff', border='#b0b0ee'))
        story.append(Spacer(1, 12))

        # ── Per-class scores ──────────────────────────────────
        scores = result.get("scores", {})
        if scores:
            story.append(Paragraph("PER-CLASS CONFIDENCE SCORES", S['h1']))
            sc_data = [["Class", "Score (%)"]] + \
                      [[k, f"{v:.2f}"] for k, v in scores.items()]
            story.append(self._table(sc_data, [9*cm, 7.5*cm], header=True))
            story.append(Spacer(1, 12))

        # ── Explanation ───────────────────────────────────────
        exps = result.get("explanation", [])
        if exps:
            story.append(Paragraph("EXPLAINABILITY — KEY FINDINGS", S['h1']))
            for i, e in enumerate(exps, 1):
                story.append(Paragraph(f"<b>{i}.</b> {e}", S['body']))
            story.append(Spacer(1, 10))

        # ── Feature table ─────────────────────────────────────
        feat_key = {
            'text':  'stylometric_features',
            'image': 'forensic_analysis',
            'audio': 'audio_features',
            'video': 'temporal_analysis',
        }.get(modality, '')
        feat_title = {
            'text':  'STYLOMETRIC FEATURES',
            'image': 'FORENSIC IMAGE ANALYSIS',
            'audio': 'AUDIO FEATURES',
            'video': 'TEMPORAL ANALYSIS',
        }.get(modality, 'FEATURE ANALYSIS')

        feats = result.get(feat_key, {})
        if feats:
            story.append(Paragraph(feat_title, S['h1']))
            rows = [["Feature", "Value"]] + \
                   [[k.replace('_',' ').title(), str(v)[:40]] for k, v in feats.items()]
            story.append(self._table(rows, [9*cm, 7.5*cm], header=True))
            story.append(Spacer(1, 12))

        # ── Plots ─────────────────────────────────────────────
        if plot_b64s:
            story.append(PageBreak())
            story.append(Paragraph("VISUALIZATIONS & ANALYTICS", S['h1']))
            story.append(HRFlowable(width="100%", thickness=1,
                                     color=colors.HexColor(PRP)))
            story.append(Spacer(1, 8))
            for b64 in (plot_b64s or []):
                if not b64:
                    continue
                try:
                    img_buf = io.BytesIO(base64.b64decode(b64))
                    story.append(RLImage(img_buf, width=15*cm, height=7*cm))
                    story.append(Spacer(1, 10))
                except Exception:
                    pass

        # ── Footer ────────────────────────────────────────────
        story.append(Spacer(1, 20))
        story.append(HRFlowable(width="100%", thickness=1,
                                  color=colors.HexColor('#aaaacc')))
        story.append(Spacer(1, 6))
        story.append(Paragraph(
            "Generated by VERITAS AI — Explainable Multi-Modal Fake &amp; AI Content Detection. "
            "This report is for research and educational purposes only. "
            "Predictions are probabilistic — always apply human judgement.",
            S['small']))

        doc.build(story)
        return buf.getvalue()

    # ── helper ────────────────────────────────────────────────

    def _table(self, data, col_widths, header=True,
               bg='#f0efff', border='#c0bfee'):
        from reportlab.platypus import Table, TableStyle
        from reportlab.lib import colors

        t = Table(data, colWidths=col_widths)
        style = [
            ('FONTSIZE',      (0,0), (-1,-1), 8),
            ('TOPPADDING',    (0,0), (-1,-1), 5),
            ('BOTTOMPADDING', (0,0), (-1,-1), 5),
            ('GRID',          (0,0), (-1,-1), 0.3, colors.HexColor(border)),
            ('ROWBACKGROUNDS',(0,0),(-1,-1),
             [colors.HexColor(bg), colors.white]),
        ]
        if header:
            style += [
                ('BACKGROUND', (0,0), (-1,0), colors.HexColor('#1a1d2e')),
                ('TEXTCOLOR',  (0,0), (-1,0), colors.white),
                ('FONTNAME',   (0,0), (-1,0), 'Helvetica-Bold'),
                ('ALIGN',      (1,0), (1,-1), 'CENTER'),
            ]
        else:
            style += [
                ('FONTNAME',   (0,0), (0,-1), 'Helvetica-Bold'),
                ('TEXTCOLOR',  (0,0), (0,-1), colors.HexColor('#6C63FF')),
            ]
        t.setStyle(TableStyle(style))
        return t