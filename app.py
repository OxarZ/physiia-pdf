import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import io
import os
import json
import traceback

from flask import Flask, request, jsonify, send_file
try:
    from flask_cors import CORS
    has_cors = True
except ImportError:
    has_cors = False

from reportlab.lib.pagesizes import A4
from reportlab.lib.units import cm
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_JUSTIFY
from reportlab.platypus import (SimpleDocTemplate, Paragraph, Spacer, Table,
                                 TableStyle, Image, HRFlowable, PageBreak)
from reportlab.platypus.flowables import Flowable
from reportlab.lib.colors import HexColor, white, black

app = Flask(__name__)
if has_cors:
    CORS(app, origins=["*"])

@app.after_request
def add_cors(response):
    response.headers['Access-Control-Allow-Origin'] = '*'
    response.headers['Access-Control-Allow-Methods'] = 'GET, POST, OPTIONS'
    response.headers['Access-Control-Allow-Headers'] = 'Content-Type'
    return response

@app.route('/health', methods=['GET'])
def health():
    return jsonify({'status': 'ok', 'service': 'PhysiIA PDF Generator'})

@app.route('/generate-pdf', methods=['POST', 'OPTIONS'])
def generate_pdf():
    if request.method == 'OPTIONS':
        return jsonify({}), 200
    data = request.get_json(force=True, silent=True) or {}
    sujet = data.get('sujet', {})
    prefs = data.get('prefs', {})
    if not sujet:
        return jsonify({'error': 'Sujet manquant'}), 400
    try:
        if isinstance(sujet, str):
            sujet = json.loads(sujet)
        buf = build_pdf(sujet, prefs)
        theme = str(sujet.get('theme', 'sujet'))[:30].replace(' ', '_')
        return send_file(buf, mimetype='application/pdf',
                         as_attachment=True,
                         download_name='PhysiIA_{}.pdf'.format(theme))
    except Exception as e:
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500

# ── COULEURS ──────────────────────────────────────────────────────
C_BLEU   = HexColor('#1a3a6b')
C_BLEU_L = HexColor('#dce8f5')
C_GRIS   = HexColor('#f5f5f5')
C_OR     = HexColor('#e67e22')
C_VERT   = HexColor('#1a6b3a')
C_ROUGE  = HexColor('#c0392b')
C_VIOLET = HexColor('#6b1a6b')
C_WHITE  = HexColor('#ffffff')
C_GRAY2  = HexColor('#888888')
C_GRID   = HexColor('#cccccc')

PART_COLORS = {'1': C_BLEU, '2': C_VERT, '3': C_ROUGE, '4': C_VIOLET}

# ── STYLES ────────────────────────────────────────────────────────
def make_styles():
    base = {'fontName': 'Helvetica', 'fontSize': 10, 'leading': 14, 'textColor': black}
    return {
        'title':  ParagraphStyle('s1', fontName='Helvetica-Bold', fontSize=12, textColor=C_WHITE, alignment=TA_CENTER, leading=16),
        'sub':    ParagraphStyle('s2', fontName='Helvetica', fontSize=9, textColor=C_WHITE, alignment=TA_CENTER, leading=12),
        'partT':  ParagraphStyle('s3', fontName='Helvetica-Bold', fontSize=10, textColor=C_WHITE, leading=14),
        'subH':   ParagraphStyle('s4', fontName='Helvetica-Bold', fontSize=10, textColor=C_BLEU, leading=14),
        'body':   ParagraphStyle('s5', fontName='Helvetica', fontSize=10, textColor=black, leading=14, spaceAfter=5, alignment=TA_JUSTIFY),
        'q':      ParagraphStyle('s6', fontName='Helvetica', fontSize=10, textColor=black, leading=14, spaceAfter=4, leftIndent=8),
        'data':   ParagraphStyle('s7', fontName='Helvetica', fontSize=9, textColor=black, leading=13, wordWrap='CJK'),
        'dataB':  ParagraphStyle('s8', fontName='Helvetica-Bold', fontSize=9, textColor=C_BLEU, leading=13),
        'note':   ParagraphStyle('s9', fontName='Helvetica-Oblique', fontSize=8, textColor=C_GRAY2, leading=12),
        'annex':  ParagraphStyle('sa', fontName='Helvetica-Bold', fontSize=10, textColor=C_BLEU, alignment=TA_CENTER, spaceAfter=4),
        'footer': ParagraphStyle('sf', fontName='Helvetica', fontSize=7, textColor=C_GRAY2, alignment=TA_CENTER),
        'ctx':    ParagraphStyle('sc', fontName='Helvetica-Bold', fontSize=10, textColor=C_BLEU, spaceAfter=4),
        'cons':   ParagraphStyle('sk', fontName='Helvetica-Bold', fontSize=9, textColor=C_GRAY2, spaceAfter=3),
    }

# ── FLOWABLE : CASE RÉPONSE ───────────────────────────────────────
class AnswerBox(Flowable):
    def __init__(self, lines=4, label=''):
        super().__init__()
        self._lines = min(lines, 8)
        self._label = label

    def wrap(self, aw, ah):
        self._aw = aw
        self._ah = self._lines * 16 + 20
        return self._aw, self._ah

    def draw(self):
        c = self.canv
        c.setStrokeColor(HexColor('#cccccc'))
        c.setFillColor(HexColor('#fafafa'))
        c.roundRect(0, 0, self._aw, self._ah, 3, fill=1, stroke=1)
        if self._label:
            c.setFillColor(HexColor('#aaaaaa'))
            c.setFont('Helvetica-Oblique', 7)
            c.drawString(5, self._ah - 11, self._label)
        c.setStrokeColor(HexColor('#e5e5e5'))
        for i in range(1, self._lines):
            y = i * 16 + 4
            c.line(6, y, self._aw - 6, y)

# ── GRAPHIQUES ────────────────────────────────────────────────────
def fig_to_img(fig, w_cm=13, h_cm=6):
    buf = io.BytesIO()
    fig.savefig(buf, format='png', dpi=110, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    buf.seek(0)
    return Image(buf, width=w_cm * cm, height=h_cm * cm)

def make_graph(spec):
    gtype = spec.get('type', 'courbe')
    title = spec.get('title', '')
    xlabel = spec.get('xlabel', '')
    ylabel = spec.get('ylabel', '')
    blank = spec.get('blank', False)
    w = min(float(spec.get('width_cm', 13)), 16)
    h = min(float(spec.get('height_cm', 6)), 10)

    fig, ax = plt.subplots(figsize=(w / 2.54, h / 2.54))

    if blank:
        xmin = float(spec.get('xmin', 0)); xmax = float(spec.get('xmax', 10))
        ymin = float(spec.get('ymin', 0)); ymax = float(spec.get('ymax', 10))
        ax.set_xlim(xmin, xmax); ax.set_ylim(ymin, ymax)
        xt = spec.get('xticks'); yt = spec.get('yticks')
        if xt: ax.set_xticks(xt)
        if yt: ax.set_yticks(yt)
        ax.grid(True, alpha=0.4)
        ax.minorticks_on(); ax.grid(True, which='minor', alpha=0.15)
        ax.text((xmin + xmax) / 2, (ymin + ymax) / 2, 'A COMPLETER',
                ha='center', va='center', fontsize=14, color='lightgray',
                alpha=0.6, fontweight='bold', rotation=10)

    elif gtype == 'rc_charge':
        tau = float(spec.get('tau', 1.0))
        E = float(spec.get('E', 6.0))
        t = np.linspace(0, 5 * tau, 400)
        uc = E * (1 - np.exp(-t / tau))
        ur = E * np.exp(-t / tau)
        ax.plot(t * 1000, uc, 'b-', lw=2, label='u_C(t)')
        ax.plot(t * 1000, ur, 'r--', lw=1.8, label='u_R(t)')
        ax.axhline(E, color='gray', ls=':', lw=1, alpha=0.6)
        ax.axvline(tau * 1000, color='blue', ls=':', lw=0.8, alpha=0.5)
        ax.axhline(0.632 * E, color='blue', ls=':', lw=0.8, alpha=0.4)
        ax.text(tau * 1000 * 1.03, 0.15, 'tau', fontsize=10, color='blue')
        ax.legend(fontsize=8)

    elif gtype == 'titrage':
        Ca = float(spec.get('Ca', 0.1))
        Va = float(spec.get('Va', 20.0))
        Cb = float(spec.get('Cb', 0.1))
        pKa = float(spec.get('pKa', 4.75))
        v = np.linspace(0, Va * 1.5, 500)

        def ph(vb):
            nb = Cb * vb / 1000
            na = Ca * Va / 1000
            if abs(vb - Va) < 0.2:
                return 7.0 + (pKa - 4.75) * 0.3
            if vb < Va:
                exc = max(na - nb, 1e-10)
                return -np.log10(exc / ((Va + vb) / 1000))
            else:
                exc = max(nb - na, 1e-10)
                poh = -np.log10(exc / ((Va + vb) / 1000))
                return 14 - poh

        phs = np.array([ph(vi) for vi in v])
        phs = np.clip(phs, 0, 14)
        ax.plot(v, phs, 'b-', lw=2.5)
        ax.axvline(Va, color='red', ls='--', lw=1.5, alpha=0.8)
        ax.plot(Va, ph(Va), 'ro', ms=7, zorder=5)
        ax.annotate('Ve={:.1f}mL'.format(Va), xy=(Va, ph(Va)),
                    xytext=(Va + 2, 5),
                    arrowprops=dict(arrowstyle='->', color='red', lw=1),
                    fontsize=8, color='darkred')
        ax.set_ylim(0, 14)
        ax.set_xlim(0, Va * 1.5)
        ax.grid(True, alpha=0.3)

    elif gtype == 'oscillateur':
        A = float(spec.get('amplitude', 5.0))
        T = float(spec.get('periode', 2.0))
        tau_a = spec.get('tau_amorti', None)
        t = np.linspace(0, 4 * T, 400)
        if tau_a:
            tau_a = float(tau_a)
            y = A * np.exp(-t / tau_a) * np.cos(2 * np.pi * t / T)
            env = A * np.exp(-t / tau_a)
            ax.plot(t, env, 'r--', lw=0.8, alpha=0.5)
            ax.plot(t, -env, 'r--', lw=0.8, alpha=0.5)
        else:
            y = A * np.cos(2 * np.pi * t / T)
        ax.plot(t, y, 'b-', lw=2, label='x(t)')
        ax.axhline(0, color='black', lw=0.5)
        ax.legend(fontsize=8)

    elif gtype == 'spectre':
        raies = spec.get('raies', [])
        for r in raies:
            lam = float(r.get('lambda', 500))
            inten = float(r.get('intensite', 0.8))
            col = r.get('color', 'blue')
            ax.vlines(lam, 0, inten, colors=col, lw=7, alpha=0.85)
            ax.text(lam, inten + 0.05, '{:.0f}nm'.format(lam),
                    ha='center', fontsize=7, color=col)
        ax.set_xlim(350, 750)
        ax.set_ylim(0, 1.4)
        ax.grid(True, alpha=0.2, axis='y')

    elif gtype == 'courbe':
        points = spec.get('points', [])
        if points:
            xs = [float(p[0]) for p in points]
            ys = [float(p[1]) for p in points]
            ax.plot(xs, ys, spec.get('style', 'b-'), lw=2,
                    label=spec.get('label', ''))
            if spec.get('label'):
                ax.legend(fontsize=8)
        for k, setter in [('xmin', None), ('xmax', None), ('ymin', None), ('ymax', None)]:
            pass
        xmin = spec.get('xmin'); xmax = spec.get('xmax')
        ymin = spec.get('ymin'); ymax = spec.get('ymax')
        if xmin is not None and xmax is not None:
            ax.set_xlim(float(xmin), float(xmax))
        if ymin is not None and ymax is not None:
            ax.set_ylim(float(ymin), float(ymax))
        xt = spec.get('xticks'); yt = spec.get('yticks')
        if xt: ax.set_xticks(xt)
        if yt: ax.set_yticks(yt)

    if xlabel: ax.set_xlabel(xlabel, fontsize=9)
    if ylabel: ax.set_ylabel(ylabel, fontsize=9)
    if title: ax.set_title(title, fontsize=8, pad=4)
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    return fig_to_img(fig, w, h)

def make_schema(spec):
    stype = spec.get('type', '')
    w = min(float(spec.get('width_cm', 10)), 14)
    h = min(float(spec.get('height_cm', 7)), 10)
    title = spec.get('title', '')
    fig, ax = plt.subplots(figsize=(w / 2.54, h / 2.54))
    ax.set_xlim(0, 10); ax.set_ylim(0, 10); ax.axis('off')

    if stype == 'circuit_rc':
        lw = 2; col = 'black'
        E_val = spec.get('E', 6)
        R_val = spec.get('R', 100)
        C_val = str(spec.get('C', '12 uF'))
        ax.plot([1, 9], [8, 8], color=col, lw=lw)
        ax.plot([1, 9], [2, 2], color=col, lw=lw)
        ax.plot([1, 1], [2, 3.2], color=col, lw=lw)
        ax.plot([1, 1], [5.8, 8], color=col, lw=lw)
        circ = plt.Circle((1, 4.5), 1.3, fill=False, color=col, lw=lw)
        ax.add_patch(circ)
        ax.text(1, 5, '+', ha='center', va='center', fontsize=13, fontweight='bold')
        ax.text(1, 4, '-', ha='center', va='center', fontsize=15, fontweight='bold')
        ax.text(0.0, 4.5, 'E\n{}V'.format(E_val), ha='center', va='center',
                fontsize=8, color='darkblue')
        ax.plot([3.5, 3.5], [8, 7.2], color=col, lw=lw)
        ax.plot([3.5, 4.5], [7.2, 8.1], color=col, lw=lw, ls='--')
        ax.plot([4.5, 4.5], [8, 8], color=col, lw=lw)
        ax.text(4.0, 8.6, 'K', ha='center', fontsize=10, fontweight='bold', color='darkred')
        ax.plot([6, 6], [8, 6.8], color=col, lw=lw)
        ax.plot([6, 6], [4.6, 3.5], color=col, lw=lw)
        rr = patches.Rectangle((5.4, 4.6), 1.2, 2.2, lw=lw,
                                edgecolor=col, facecolor='lightyellow')
        ax.add_patch(rr)
        ax.text(6, 5.7, 'R', ha='center', va='center', fontsize=11, fontweight='bold')
        ax.text(7.0, 5.7, '{} ohm'.format(R_val), ha='left', va='center',
                fontsize=7.5, color='darkblue')
        ax.plot([8, 8], [8, 7], color=col, lw=lw)
        ax.plot([8, 8], [2, 3], color=col, lw=lw)
        ax.plot([7.2, 8.8], [7, 7], color=col, lw=3)
        ax.plot([7.2, 8.8], [3, 3], color=col, lw=3)
        ax.text(9.1, 5, 'C\n{}'.format(C_val), ha='left', va='center',
                fontsize=7.5, color='darkblue')
        ax.plot([6, 8], [8, 8], color=col, lw=lw)
        ax.plot([6, 8], [2, 2], color=col, lw=lw)

    elif stype == 'titrage':
        ax.text(5, 9, 'Burette (NaOH)', ha='center', fontsize=9,
                fontweight='bold', color='#1a5fa8')
        ax.plot([5, 5], [8.5, 7.2], color='black', lw=2)
        ax.text(5, 6.8, 'v', ha='center', fontsize=14, color='black')
        becher = patches.FancyBboxPatch((2.5, 2.5), 5, 3.5,
                                        boxstyle='round,pad=0.1', lw=1.5,
                                        edgecolor='black', facecolor='#fff9e6')
        ax.add_patch(becher)
        liq = patches.FancyBboxPatch((2.6, 2.6), 4.8, 2.5,
                                     boxstyle='round,pad=0.05', lw=0,
                                     facecolor='#d4edff', alpha=0.7)
        ax.add_patch(liq)
        ax.text(5, 3.9, 'Solution a titrer', ha='center', va='center',
                fontsize=9, color='#8B4513')
        ax.text(8.5, 5.5, 'pH-metre', ha='center', fontsize=8,
                fontweight='bold', color='darkgreen')
        ax.plot([7.5, 5.8], [5.2, 5.0], color='darkgreen', lw=1, ls='--')
        ax.plot([5.8, 5.8], [5.0, 3.2], color='darkgreen', lw=1.5)
        ax.add_patch(plt.Circle((5.8, 3.2), 0.18, fill=True,
                                facecolor='darkgreen'))
        ax.add_patch(patches.Ellipse((5, 2.0), 3.5, 0.5, lw=1,
                                     edgecolor='gray', facecolor='#e0e0e0'))
        ax.text(5, 2.0, 'Agitateur magnetique', ha='center', va='center',
                fontsize=7.5, color='gray')

    elif stype == 'onde':
        lambda_val = float(spec.get('lambda', 2.0))
        A_val = float(spec.get('A', 1.0))
        ax.axis('on')
        x = np.linspace(0, 4 * lambda_val, 400)
        y = A_val * np.sin(2 * np.pi * x / lambda_val)
        ax.set_xlim(0, 4 * lambda_val)
        ax.set_ylim(-A_val * 2, A_val * 2)
        ax.plot(x, y, 'b-', lw=2)
        ax.axhline(0, color='black', lw=0.5)
        ax.annotate('', xy=(lambda_val * 2, A_val * 1.5),
                    xytext=(lambda_val, A_val * 1.5),
                    arrowprops=dict(arrowstyle='<->', color='red', lw=1.5))
        ax.text(lambda_val * 1.5, A_val * 1.7, 'lambda', ha='center',
                fontsize=10, color='red')
        ax.set_xlabel('x (m)'); ax.set_ylabel('y (m)')

    if title: ax.set_title(title, fontsize=8, pad=3)
    fig.tight_layout(pad=0.2)
    return fig_to_img(fig, w, h)

# ── CONSTRUCTION PDF ──────────────────────────────────────────────
def safe_str(val, maxlen=120):
    return str(val or '')[:maxlen]

def build_pdf(sujet, prefs):
    buf = io.BytesIO()
    ST = make_styles()
    W_PAGE = A4[0] - 4 * cm  # largeur utile

    couleur = prefs.get('couleur', 'bleu')
    color_map = {'bleu': C_BLEU, 'vert': C_VERT, 'rouge': C_ROUGE, 'violet': C_VIOLET}
    ACCENT = color_map.get(couleur, C_BLEU)

    doc = SimpleDocTemplate(buf, pagesize=A4,
                            leftMargin=2 * cm, rightMargin=2 * cm,
                            topMargin=1.5 * cm, bottomMargin=2 * cm,
                            allowSplitting=1)
    story = []

    def hr():
        return HRFlowable(width='100%', thickness=0.6, color=ACCENT,
                          spaceAfter=4, spaceBefore=4)

    def safe_para(text, style, maxlen=200):
        return Paragraph(safe_str(text, maxlen), style)

    def part_header(num, title_text, pts):
        col = PART_COLORS.get(str(num), ACCENT)
        t = Table([[safe_para('PARTIE {}'.format(num), ST['partT']),
                    safe_para(title_text, ST['partT']),
                    safe_para('{} pts'.format(pts), ST['partT'])]],
                  colWidths=[2.5 * cm, 12 * cm, 2.5 * cm])
        t.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, -1), col),
            ('TOPPADDING', (0, 0), (-1, -1), 6),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 6),
            ('LEFTPADDING', (0, 0), (-1, -1), 8),
            ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
        ]))
        return t

    def sub_header(letter, title_text, pts):
        t = Table([[safe_para('  {} — {}'.format(letter, title_text), ST['subH']),
                    safe_para('({} pts)'.format(pts), ST['note'])]],
                  colWidths=[14 * cm, 3 * cm])
        t.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, -1), C_BLEU_L),
            ('LINEBELOW', (0, 0), (-1, -1), 0.5, ACCENT),
            ('TOPPADDING', (0, 0), (-1, -1), 5),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 5),
            ('LEFTPADDING', (0, 0), (0, 0), 8),
        ]))
        return t

    def donnees_box(items):
        if not items:
            return Spacer(1, 0.1 * cm)
        rows = []
        for item in items:
            txt = safe_str(item, 250)
            rows.append([Paragraph('•', ST['data']),
                         Paragraph(txt, ST['data'])])
        t = Table(rows, colWidths=[0.4 * cm, 14.6 * cm],
                  style=TableStyle([
                      ('BACKGROUND', (0, 0), (-1, -1), C_GRIS),
                      ('BOX', (0, 0), (-1, -1), 0.5, ACCENT),
                      ('TOPPADDING', (0, 0), (-1, -1), 2),
                      ('BOTTOMPADDING', (0, 0), (-1, -1), 2),
                      ('LEFTPADDING', (0, 0), (0, 0), 6),
                      ('LEFTPADDING', (1, 0), (1, 0), 4),
                      ('VALIGN', (0, 0), (-1, -1), 'TOP'),
                  ]))
        return t

    def question_p(num, text, pts=None):
        suffix = ''
        if pts:
            suffix = '  <i>({} pt{})</i>'.format(pts, 's' if pts > 1 else '')
        return Paragraph('<b>{}.</b>  {}{}'.format(
            safe_str(num, 10), safe_str(text, 400), suffix), ST['q'])

    # ── PAGE DE GARDE ──────────────────────────────────────────────
    h1 = Table([[safe_para('BACCALAUREAT GENERAL', ST['title']),
                 safe_para('EPREUVE DE PHYSIQUE-CHIMIE', ST['title'])]],
               colWidths=[8.5 * cm, 8.5 * cm])
    h1.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, -1), ACCENT),
        ('TOPPADDING', (0, 0), (-1, -1), 9),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 9),
        ('LINEAFTER', (0, 0), (0, 0), 1, C_WHITE),
    ]))
    story.append(h1)

    duree = safe_str(sujet.get('duree', '3h30'), 20)
    coeff = safe_str(sujet.get('coefficient', '6'), 10)
    h2 = Table([[safe_para('Serie Generale - Specialite', ST['sub']),
                 safe_para('Duree : {}  -  Coefficient : {}'.format(duree, coeff), ST['sub']),
                 safe_para('Session 2025', ST['sub'])]],
               colWidths=[6 * cm, 7 * cm, 4 * cm])
    h2.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, -1), HexColor('#2c5282')),
        ('TOPPADDING', (0, 0), (-1, -1), 4),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 4),
    ]))
    story.append(h2)
    story.append(Spacer(1, 0.3 * cm))

    theme = safe_str(sujet.get('theme', ''), 80)
    theme_t = Table([[Paragraph('<b>THEME : {}</b>'.format(theme.upper()),
                                ParagraphStyle('th', fontName='Helvetica-Bold',
                                               fontSize=10, textColor=ACCENT,
                                               alignment=TA_CENTER))]],
                    colWidths=[17 * cm])
    theme_t.setStyle(TableStyle([
        ('BOX', (0, 0), (-1, -1), 1.5, ACCENT),
        ('TOPPADDING', (0, 0), (-1, -1), 7),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 7),
        ('BACKGROUND', (0, 0), (-1, -1), C_BLEU_L),
    ]))
    story.append(theme_t)
    story.append(Spacer(1, 0.25 * cm))

    consignes = [
        "L'usage de la calculatrice est autorise.",
        "Le candidat doit traiter toutes les parties. L'ordre est libre.",
        "La qualite de la redaction est prise en compte dans la notation.",
        "Les resultats doivent etre exprimes avec les unites et chiffres significatifs adaptes.",
        "Les annexes sont a rendre avec la copie.",
    ]
    rows_c = [[Paragraph('o  ' + c, ST['data'])] for c in consignes]
    tc = Table(rows_c, colWidths=[17 * cm])
    tc.setStyle(TableStyle([
        ('BOX', (0, 0), (-1, -1), 0.5, C_GRAY2),
        ('INNERGRID', (0, 0), (-1, -1), 0.3, C_GRID),
        ('TOPPADDING', (0, 0), (-1, -1), 3),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 3),
        ('LEFTPADDING', (0, 0), (-1, -1), 8),
        ('BACKGROUND', (0, 0), (-1, -1), HexColor('#fafafa')),
    ]))
    story.append(Paragraph('<b>CONSIGNES</b>', ST['cons']))
    story.append(tc)
    story.append(Spacer(1, 0.2 * cm))

    # Tableau récap parties
    parties = sujet.get('parties', [])
    recap = [
        [safe_para('Partie', ST['dataB']),
         safe_para('Theme', ST['dataB']),
         safe_para('Notions', ST['dataB']),
         safe_para('Points', ST['dataB'])],
    ]
    for p in parties:
        recap.append([
            safe_para('Partie {}'.format(p.get('numero', '')), ST['data']),
            safe_para(p.get('titre', ''), ST['data']),
            safe_para(p.get('notions', ''), ST['data']),
            safe_para('{} pts'.format(p.get('points', '')), ST['data']),
        ])
    tr = Table(recap, colWidths=[2.8 * cm, 6 * cm, 5.7 * cm, 2.5 * cm])
    tr.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), ACCENT),
        ('TEXTCOLOR', (0, 0), (-1, 0), C_WHITE),
        ('ROWBACKGROUNDS', (0, 1), (-1, -1), [C_BLEU_L, C_WHITE]),
        ('GRID', (0, 0), (-1, -1), 0.4, C_GRID),
        ('TOPPADDING', (0, 0), (-1, -1), 4),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 4),
        ('LEFTPADDING', (0, 0), (-1, -1), 5),
        ('VALIGN', (0, 0), (-1, -1), 'TOP'),
    ]))
    story.append(tr)
    story.append(Spacer(1, 0.25 * cm))

    contexte = safe_str(sujet.get('contexte', ''), 600)
    if contexte:
        story.append(Paragraph('<b>CONTEXTE GENERAL</b>', ST['ctx']))
        story.append(Paragraph(contexte, ST['body']))
    story.append(hr())

    # ── PARTIES ────────────────────────────────────────────────────
    annexes = []

    for partie in parties:
        num = str(partie.get('numero', '1'))
        titr = safe_str(partie.get('titre', ''), 80)
        pts = partie.get('points', 0)
        intro = safe_str(partie.get('intro', ''), 500)

        story.append(part_header(num, titr, pts))
        story.append(Spacer(1, 0.2 * cm))
        if intro:
            story.append(Paragraph(intro, ST['body']))
            story.append(Spacer(1, 0.1 * cm))

        for sec in partie.get('sections', []):
            sec_letter = safe_str(sec.get('lettre', 'A'), 2)
            sec_title = safe_str(sec.get('titre', ''), 80)
            sec_pts = sec.get('points', 0)

            story.append(sub_header(sec_letter, sec_title, sec_pts))
            story.append(Spacer(1, 0.1 * cm))

            donnees = sec.get('donnees', [])
            if donnees:
                story.append(Paragraph('<b>Donnees :</b>', ST['dataB']))
                story.append(donnees_box(donnees))
                story.append(Spacer(1, 0.15 * cm))

            for sch in sec.get('schemas', []):
                try:
                    img = make_schema(sch)
                    story.append(img)
                    story.append(Spacer(1, 0.1 * cm))
                except Exception as e:
                    story.append(Paragraph('[Schema: {}]'.format(sch.get('title', '')), ST['note']))

            for gr in sec.get('graphiques', []):
                try:
                    img = make_graph(gr)
                    story.append(img)
                    story.append(Spacer(1, 0.1 * cm))
                    if gr.get('aussi_annexe'):
                        gr_blank = dict(gr)
                        gr_blank['blank'] = True
                        gr_blank['title'] = 'Annexe - ' + gr.get('title', '')
                        annexes.append({'type': 'graph', 'spec': gr_blank,
                                        'question': gr.get('question_annexe', '')})
                except Exception as e:
                    story.append(Paragraph('[Graphique: {}]'.format(gr.get('title', '')), ST['note']))

            for tab in sec.get('tableaux', []):
                try:
                    tab_title = safe_str(tab.get('title', ''), 100)
                    headers = tab.get('headers', [])
                    rows_t = tab.get('rows', [])
                    if tab_title:
                        story.append(Paragraph('<b>{}</b>'.format(tab_title), ST['note']))
                    all_rows = []
                    if headers:
                        all_rows.append([safe_para(str(h)[:40], ST['dataB']) for h in headers])
                    for row in rows_t:
                        all_rows.append([safe_para(str(c)[:80], ST['data']) for c in row])
                    if all_rows:
                        ncols = len(all_rows[0])
                        cw = (17 * cm) / ncols
                        tt = Table(all_rows, colWidths=[cw] * ncols)
                        ts = [
                            ('GRID', (0, 0), (-1, -1), 0.4, C_GRID),
                            ('TOPPADDING', (0, 0), (-1, -1), 4),
                            ('BOTTOMPADDING', (0, 0), (-1, -1), 4),
                            ('LEFTPADDING', (0, 0), (-1, -1), 5),
                            ('VALIGN', (0, 0), (-1, -1), 'TOP'),
                        ]
                        if headers:
                            ts.append(('BACKGROUND', (0, 0), (-1, 0), ACCENT))
                            ts.append(('TEXTCOLOR', (0, 0), (-1, 0), C_WHITE))
                            ts.append(('ROWBACKGROUNDS', (0, 1), (-1, -1), [C_WHITE, C_BLEU_L]))
                        tt.setStyle(TableStyle(ts))
                        story.append(tt)
                        story.append(Spacer(1, 0.15 * cm))
                    if tab.get('aussi_annexe'):
                        annexes.append({'type': 'tableau', 'spec': tab,
                                        'question': tab.get('question_annexe', '')})
                except Exception as e:
                    story.append(Paragraph('[Tableau: {}]'.format(tab.get('title', '')), ST['note']))

            for q in sec.get('questions', []):
                q_num = safe_str(q.get('numero', ''), 10)
                q_text = safe_str(q.get('texte', ''), 400)
                q_pts = q.get('points', None)
                q_lines = min(int(q.get('lignes_reponse', 3)), 8)
                story.append(question_p(q_num, q_text, q_pts))
                story.append(AnswerBox(q_lines, 'Reponse {}'.format(q_num)))
                story.append(Spacer(1, 0.1 * cm))

        story.append(hr())

    # ── ANNEXES ────────────────────────────────────────────────────
    if annexes:
        story.append(PageBreak())
        ah = Table([[Paragraph('ANNEXES - A RENDRE AVEC LA COPIE',
                               ParagraphStyle('ah', fontName='Helvetica-Bold',
                                              fontSize=11, textColor=C_WHITE,
                                              alignment=TA_CENTER))]],
                   colWidths=[17 * cm])
        ah.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, -1), C_OR),
            ('TOPPADDING', (0, 0), (-1, -1), 9),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 9),
        ]))
        story.append(ah)
        story.append(Spacer(1, 0.3 * cm))

        nom = safe_str(prefs.get('nom', ''), 40)
        prenom = safe_str(prefs.get('prenom', ''), 40)
        cand = Table([
            [safe_para('NOM : {}'.format(nom or '_' * 30), ST['data']),
             safe_para('Prenom : {}'.format(prenom or '_' * 30), ST['data'])],
            [safe_para('N candidat : __________', ST['data']),
             safe_para('Classe : {}'.format(safe_str(prefs.get('classe', 'Terminale'), 20)), ST['data'])],
        ], colWidths=[8.5 * cm, 8.5 * cm])
        cand.setStyle(TableStyle([
            ('BOX', (0, 0), (-1, -1), 1, ACCENT),
            ('INNERGRID', (0, 0), (-1, -1), 0.5, C_GRID),
            ('TOPPADDING', (0, 0), (-1, -1), 7),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 7),
            ('LEFTPADDING', (0, 0), (-1, -1), 8),
        ]))
        story.append(cand)
        story.append(Spacer(1, 0.4 * cm))

        for i, ann in enumerate(annexes, 1):
            q_label = safe_str(ann.get('question', ''), 80)
            story.append(Paragraph(
                'ANNEXE {}{}'.format(i, ' - ' + q_label if q_label else ''),
                ST['annex']))
            story.append(Spacer(1, 0.1 * cm))
            if ann['type'] == 'graph':
                try:
                    story.append(make_graph(ann['spec']))
                except Exception:
                    story.append(Paragraph('[Graphe a completer]', ST['note']))
            elif ann['type'] == 'tableau':
                try:
                    spec = ann['spec']
                    headers = spec.get('headers', [])
                    rows_v = spec.get('rows_vides', [[''] * len(headers)] * 4)
                    all_rows = []
                    if headers:
                        all_rows.append([safe_para(str(h)[:40], ST['dataB']) for h in headers])
                    for row in rows_v:
                        all_rows.append([safe_para(str(c)[:60], ST['data']) for c in row])
                    if all_rows:
                        ncols = len(all_rows[0])
                        ta = Table(all_rows, colWidths=[17 * cm / ncols] * ncols)
                        ta.setStyle(TableStyle([
                            ('GRID', (0, 0), (-1, -1), 0.5, C_GRID),
                            ('BACKGROUND', (0, 0), (-1, 0), ACCENT),
                            ('TEXTCOLOR', (0, 0), (-1, 0), C_WHITE),
                            ('TOPPADDING', (0, 0), (-1, -1), 12),
                            ('BOTTOMPADDING', (0, 0), (-1, -1), 12),
                            ('LEFTPADDING', (0, 0), (-1, -1), 5),
                        ]))
                        story.append(ta)
                except Exception:
                    story.append(Paragraph('[Tableau a completer]', ST['note']))
            story.append(Spacer(1, 0.3 * cm))

    # Footer
    story.append(HRFlowable(width='100%', thickness=0.5,
                             color=C_GRAY2, spaceAfter=3, spaceBefore=3))
    story.append(Paragraph(
        'Baccalaureat General - Epreuve de Physique-Chimie - Session 2025 - '
        'Les annexes sont a rendre avec la copie.',
        ST['footer']))

    doc.build(story)
    buf.seek(0)
    return buf


if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False)
