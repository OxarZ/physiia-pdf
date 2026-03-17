"""
PhysiIA — Backend PDF (Render.com)
Reçoit un JSON de contenu de sujet bac et génère un PDF professionnel imprimable
"""

from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
import anthropic
import json
import io
import os
import traceback

# PDF libs
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
from reportlab.lib.pagesizes import A4
from reportlab.lib import colors
from reportlab.lib.units import cm
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_JUSTIFY
from reportlab.platypus import (SimpleDocTemplate, Paragraph, Spacer, Table,
                                 TableStyle, Image, HRFlowable, PageBreak, KeepTogether)
from reportlab.platypus.flowables import Flowable
from reportlab.lib.colors import HexColor

app = Flask(__name__)
CORS(app, origins=["*"])

ANTHROPIC_KEY = os.environ.get("ANTHROPIC_API_KEY", "")
client = anthropic.Anthropic(api_key=ANTHROPIC_KEY)

# ── COULEURS ──────────────────────────────────────────────────────
BLEU      = HexColor('#1a3a6b')
BLEU_L    = HexColor('#dce8f5')
GRIS      = HexColor('#f5f5f5')
ROUGE     = HexColor('#c0392b')
OR        = HexColor('#e67e22')
VERT      = HexColor('#1a6b3a')
VIOLET    = HexColor('#6b1a6b')

PART_COLORS = {'1': BLEU, '2': VERT, '3': ROUGE, '4': VIOLET}

# ── STYLES ────────────────────────────────────────────────────────
def make_styles():
    return {
        'title':    ParagraphStyle('t1', fontName='Helvetica-Bold',   fontSize=13, textColor=HexColor('#ffffff'), alignment=TA_CENTER, leading=16),
        'sub':      ParagraphStyle('t2', fontName='Helvetica',        fontSize=9,  textColor=HexColor('#ffffff'), alignment=TA_CENTER, leading=12),
        'partT':    ParagraphStyle('t3', fontName='Helvetica-Bold',   fontSize=11, textColor=HexColor('#ffffff'), alignment=TA_LEFT,   leading=14),
        'subH':     ParagraphStyle('t4', fontName='Helvetica-Bold',   fontSize=10, textColor=BLEU,               leading=14),
        'body':     ParagraphStyle('t5', fontName='Helvetica',        fontSize=10, textColor=HexColor('#000000'), leading=14, spaceAfter=5, alignment=TA_JUSTIFY),
        'q':        ParagraphStyle('t6', fontName='Helvetica',        fontSize=10, textColor=HexColor('#000000'), leading=14, spaceAfter=4, leftIndent=10),
        'data':     ParagraphStyle('t7', fontName='Helvetica',        fontSize=9.5,textColor=HexColor('#000000'), leading=13, spaceAfter=2),
        'dataB':    ParagraphStyle('t8', fontName='Helvetica-Bold',   fontSize=9.5,textColor=BLEU,               leading=13, spaceAfter=2),
        'note':     ParagraphStyle('t9', fontName='Helvetica-Oblique',fontSize=8.5,textColor=HexColor('#555555'),leading=12, spaceAfter=3),
        'annex':    ParagraphStyle('ta', fontName='Helvetica-Bold',   fontSize=10, textColor=BLEU,               alignment=TA_CENTER, spaceAfter=4),
        'footer':   ParagraphStyle('tf', fontName='Helvetica',        fontSize=7.5,textColor=HexColor('#888888'),alignment=TA_CENTER),
    }

# ── FLOWABLES CUSTOM ──────────────────────────────────────────────
class AnswerBox(Flowable):
    def __init__(self, lines=4, label='', w=None):
        super().__init__()
        self._lines = lines; self._label = label; self._w = w
    def wrap(self, aw, ah):
        self._aw = self._w or aw
        self._ah = self._lines * 18 + 22
        return self._aw, self._ah
    def draw(self):
        c = self.canv
        c.setStrokeColor(HexColor('#bbbbbb'))
        c.setFillColor(HexColor('#fafafa'))
        c.roundRect(0, 0, self._aw, self._ah, 3, fill=1, stroke=1)
        if self._label:
            c.setFillColor(HexColor('#999999'))
            c.setFont('Helvetica-Oblique', 7.5)
            c.drawString(6, self._ah - 13, self._label)
        c.setStrokeColor(HexColor('#e0e0e0'))
        for i in range(1, self._lines):
            y = i * 18 + 4
            c.line(8, y, self._aw - 8, y)

class DottedLine(Flowable):
    def wrap(self, aw, ah): self._aw = aw; return aw, 10
    def draw(self):
        c = self.canv
        c.setStrokeColor(HexColor('#bbbbbb'))
        c.setDash(2, 4)
        c.line(0, 5, self._aw, 5)
        c.setDash()

# ── GÉNÉRATION DE GRAPHIQUES ──────────────────────────────────────
def fig_to_img(fig, w_cm=13, h_cm=6):
    buf = io.BytesIO()
    fig.savefig(buf, format='png', dpi=130, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    buf.seek(0)
    return Image(buf, width=w_cm*cm, height=h_cm*cm)

def make_graph_generic(graph_spec):
    """Génère un graphique à partir des specs JSON fournies par Claude"""
    gtype = graph_spec.get('type', 'courbe')
    title = graph_spec.get('title', '')
    xlabel = graph_spec.get('xlabel', 'x')
    ylabel = graph_spec.get('ylabel', 'y')
    blank = graph_spec.get('blank', False)  # True = annexe à compléter
    w = graph_spec.get('width_cm', 13)
    h = graph_spec.get('height_cm', 6)

    fig, ax = plt.subplots(figsize=(w/2.54, h/2.54))

    if blank:
        # Graphe quadrillé vierge pour annexe
        xmin = graph_spec.get('xmin', 0)
        xmax = graph_spec.get('xmax', 10)
        ymin = graph_spec.get('ymin', 0)
        ymax = graph_spec.get('ymax', 10)
        xticks = graph_spec.get('xticks', None)
        yticks = graph_spec.get('yticks', None)
        ax.set_xlim(xmin, xmax); ax.set_ylim(ymin, ymax)
        if xticks: ax.set_xticks(xticks)
        if yticks: ax.set_yticks(yticks)
        ax.grid(True, alpha=0.4); ax.minorticks_on(); ax.grid(True, which='minor', alpha=0.15)
        ax.text((xmin+xmax)/2, (ymin+ymax)/2, 'À COMPLÉTER',
                ha='center', va='center', fontsize=16, color='lightgray',
                alpha=0.5, fontweight='bold', rotation=12)

    elif gtype == 'rc_charge':
        tau = graph_spec.get('tau', 1.0)
        E   = graph_spec.get('E', 6.0)
        t   = np.linspace(0, 5*tau, 500)
        uc  = E * (1 - np.exp(-t/tau))
        ur  = E * np.exp(-t/tau)
        ax.plot(t*1000, uc, 'b-',  lw=2,   label=r'$u_C(t)$')
        ax.plot(t*1000, ur, 'r--', lw=1.8, label=r'$u_R(t)$')
        ax.axhline(E, color='gray', ls=':', lw=1, alpha=0.6)
        ax.axvline(tau*1000, color='blue', ls=':', lw=0.8, alpha=0.5)
        ax.axhline(0.632*E, color='blue', ls=':', lw=0.8, alpha=0.5)
        ax.text(tau*1000*1.03, 0.2, 'τ', fontsize=11, color='blue')
        ax.text(5*tau*1000*1.01, E+0.1, f'E={E}V', fontsize=8, color='gray')
        ax.legend(fontsize=8, loc='right')
        ax.set_xlabel('t (ms)'); ax.set_ylabel('Tension (V)')

    elif gtype == 'titrage':
        Ca = graph_spec.get('Ca', 0.10)
        Va = graph_spec.get('Va', 20.0)
        Cb = graph_spec.get('Cb', 0.10)
        pKa = graph_spec.get('pKa', 4.75)
        v   = np.linspace(0, Va*1.5, 600)
        def ph(vb):
            nb = Cb * vb / 1000; na = Ca * Va / 1000
            if abs(vb - Va) < 0.15: return 7.0 + (pKa - 4.75)*0.5
            if vb < Va:
                exc = na - nb
                return -np.log10(max(exc/((Va+vb)/1000 if Va+vb>0 else 1), 1e-14))
            else:
                exc = nb - na
                poh = -np.log10(max(exc/((Va+vb)/1000 if Va+vb>0 else 1), 1e-14))
                return 14 - poh
        phs = np.array([ph(vi) for vi in v])
        ax.plot(v, phs, 'b-', lw=2.5, label='pH = f(V)')
        ax.axvline(Va, color='red', ls='--', lw=1.5, alpha=0.8)
        ax.plot(Va, 7.0, 'ro', ms=8, zorder=5)
        ax.annotate(f'V_E = {Va:.1f} mL', xy=(Va, 7.0), xytext=(Va+2, 5),
                    arrowprops=dict(arrowstyle='->', color='red', lw=1.2),
                    fontsize=8, color='darkred',
                    bbox=dict(boxstyle='round,pad=0.3', facecolor='lightyellow', alpha=0.8))
        ax.set_xlabel('V(solution titrante) (mL)'); ax.set_ylabel('pH')
        ax.set_xlim(0, Va*1.5); ax.set_ylim(0, 14)
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    elif gtype == 'spectre':
        raies = graph_spec.get('raies', [])
        for r in raies:
            ax.vlines(r['lambda'], 0, r['intensite'], colors=r.get('color','blue'), lw=8, alpha=0.85)
            ax.text(r['lambda'], r['intensite']+0.04, f"{r['lambda']} nm", ha='center', fontsize=7)
        ax.set_xlabel('λ (nm)'); ax.set_ylabel('Intensité relative')
        ax.set_xlim(350, 750); ax.set_ylim(0, 1.4)
        ax.grid(True, alpha=0.2, axis='y')

    elif gtype == 'courbe':
        # Courbe générique définie par points
        points = graph_spec.get('points', [])
        if points:
            xs = [p[0] for p in points]
            ys = [p[1] for p in points]
            style = graph_spec.get('style', 'b-')
            label = graph_spec.get('label', '')
            ax.plot(xs, ys, style, lw=2, label=label)
            if label: ax.legend(fontsize=8)
        xmin = graph_spec.get('xmin', None)
        xmax = graph_spec.get('xmax', None)
        ymin = graph_spec.get('ymin', None)
        ymax = graph_spec.get('ymax', None)
        if xmin is not None: ax.set_xlim(xmin, xmax)
        if ymin is not None: ax.set_ylim(ymin, ymax)
        xticks = graph_spec.get('xticks', None)
        yticks = graph_spec.get('yticks', None)
        if xticks: ax.set_xticks(xticks)
        if yticks: ax.set_yticks(yticks)

    elif gtype == 'oscillateur':
        A   = graph_spec.get('amplitude', 5.0)
        T   = graph_spec.get('periode', 2.0)
        tau_a = graph_spec.get('tau_amorti', None)
        t   = np.linspace(0, 4*T, 500)
        if tau_a:
            y = A * np.exp(-t/tau_a) * np.cos(2*np.pi*t/T)
            ax.plot([-A*np.exp(-ti/tau_a) for ti in t], 'r--', lw=0.8, alpha=0.5)
            ax.plot([A*np.exp(-ti/tau_a) for ti in t], 'r--', lw=0.8, alpha=0.5, label='enveloppe')
        else:
            y = A * np.cos(2*np.pi*t/T)
        ax.plot(t, y, 'b-', lw=2, label='x(t)')
        ax.axhline(0, color='black', lw=0.5)
        ax.set_xlabel('t (s)'); ax.set_ylabel('x (m)')
        ax.legend(fontsize=8)

    ax.set_xlabel(xlabel if xlabel else ax.get_xlabel())
    ax.set_ylabel(ylabel if ylabel else ax.get_ylabel())
    if title: ax.set_title(title, fontsize=9, pad=5)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    return fig_to_img(fig, w, h)

def make_schema(schema_spec):
    """Génère un schéma à partir des specs"""
    stype = schema_spec.get('type', '')
    w = schema_spec.get('width_cm', 10)
    h = schema_spec.get('height_cm', 7)
    fig, ax = plt.subplots(figsize=(w/2.54, h/2.54))
    ax.set_xlim(0, 10); ax.set_ylim(0, 10); ax.axis('off')

    if stype == 'circuit_rc':
        lw = 2; col = 'black'
        E_val = schema_spec.get('E', 6); R_val = schema_spec.get('R', 100); C_val = schema_spec.get('C', '12 μF')
        ax.plot([1,9],[8,8], color=col, lw=lw); ax.plot([1,9],[2,2], color=col, lw=lw)
        ax.plot([1,1],[2,3.2], color=col, lw=lw); ax.plot([1,1],[5.8,8], color=col, lw=lw)
        circ = plt.Circle((1,4.5), 1.3, fill=False, color=col, lw=lw); ax.add_patch(circ)
        ax.text(1,5,'＋',ha='center',va='center',fontsize=13,fontweight='bold')
        ax.text(1,4,'－',ha='center',va='center',fontsize=15,fontweight='bold')
        ax.text(0.0,4.5,f'E\n{E_val}V',ha='center',va='center',fontsize=8.5,color='darkblue')
        ax.plot([3.5,3.5],[8,7.2],color=col,lw=lw); ax.plot([3.5,4.5],[7.2,8.1],color=col,lw=lw,ls='--')
        ax.plot([4.5,4.5],[8,8],color=col,lw=lw)
        ax.text(4.0,8.6,'K',ha='center',fontsize=10,fontweight='bold',color='darkred')
        ax.plot([6,6],[8,6.8],color=col,lw=lw); ax.plot([6,6],[4.6,3.5],color=col,lw=lw)
        rr = patches.Rectangle((5.4,4.6),1.2,2.2,lw=lw,edgecolor=col,facecolor='lightyellow'); ax.add_patch(rr)
        ax.text(6,5.7,'R',ha='center',va='center',fontsize=11,fontweight='bold')
        ax.text(7.0,5.7,f'{R_val} Ω',ha='left',va='center',fontsize=8,color='darkblue')
        ax.plot([8,8],[8,7],color=col,lw=lw); ax.plot([8,8],[2,3],color=col,lw=lw)
        ax.plot([7.2,8.8],[7,7],color=col,lw=3); ax.plot([7.2,8.8],[3,3],color=col,lw=3)
        ax.text(9.1,5,f'C\n{C_val}',ha='left',va='center',fontsize=8,color='darkblue')
        ax.plot([6,8],[8,8],color=col,lw=lw); ax.plot([6,8],[2,2],color=col,lw=lw)
        ax.annotate('',xy=(7.5,8.5),xytext=(5,8.5),arrowprops=dict(arrowstyle='->',color='red',lw=1.5))
        ax.text(6.2,9.0,'i(t)',ha='center',fontsize=9,color='red')
        ax.annotate('',xy=(9.5,3.2),xytext=(9.5,7.2),arrowprops=dict(arrowstyle='<->',color='blue',lw=1.5))
        ax.text(9.8,5,'u_C',ha='left',fontsize=10,color='blue',style='italic')

    elif stype == 'titrage':
        circle_b = plt.Circle((5,8.5),1.0,fill=False,color='black',lw=1.5); ax.add_patch(circle_b)
        ax.plot([5,5],[7.5,6.8],color='black',lw=1.5)
        ax.text(5,8.5,'NaOH',ha='center',va='center',fontsize=8,color='#1a5fa8',fontweight='bold')
        ax.text(5,7.2,'▼',ha='center',fontsize=12,color='black')
        becher = patches.FancyBboxPatch((2,2.5),6,3.5,boxstyle='round,pad=0.1',lw=1.5,edgecolor='black',facecolor='#fff9e6'); ax.add_patch(becher)
        liq = patches.FancyBboxPatch((2.1,2.6),5.8,2.4,boxstyle='round,pad=0.05',lw=0,facecolor='#d4edff',alpha=0.7); ax.add_patch(liq)
        ax.text(5,3.8,'Solution à titrer',ha='center',va='center',fontsize=9,color='#8B4513')
        phm = patches.Rectangle((7.5,4),2,1.2,lw=1.5,edgecolor='darkgreen',facecolor='#e8fce8'); ax.add_patch(phm)
        ax.text(8.5,4.6,'pH-mètre',ha='center',va='center',fontsize=8,color='darkgreen',fontweight='bold')
        ax.plot([7.6,5.5],[4.0,4.2],color='darkgreen',lw=1,ls='--'); ax.plot([5.5,5.5],[4.2,2.9],color='darkgreen',lw=1.5)
        ax.add_patch(plt.Circle((5.5,2.9),0.18,fill=True,facecolor='darkgreen'))
        ax.add_patch(patches.Ellipse((5,2.2),3.5,0.5,lw=1,edgecolor='gray',facecolor='#e0e0e0'))
        ax.text(5,2.2,'Agitateur magnétique',ha='center',va='center',fontsize=7.5,color='gray')

    elif stype == 'optique':
        ax.plot([0.5,9.5],[5,5],color='black',lw=1,ls='--',alpha=0.5)
        ax.text(9.7,5,'axe\nopt.',ha='left',va='center',fontsize=7,color='gray')
        lenses = schema_spec.get('lenses', [])
        for lens in lenses:
            x = lens.get('x', 5); vergence = lens.get('vergence', 10); label = lens.get('label','L')
            ax.annotate('',xy=(x,6.5),xytext=(x,3.5),arrowprops=dict(arrowstyle='<->',color='blue',lw=2))
            ax.text(x+0.15,7,label,fontsize=9,color='blue',fontweight='bold')
            fp = 10/vergence
            ax.plot([x+fp,x+fp],[4.8,5.2],color='blue',lw=1)
            ax.text(x+fp,4.5,"F'",ha='center',fontsize=8,color='blue')
            ax.plot([x-fp,x-fp],[4.8,5.2],color='blue',lw=1)
            ax.text(x-fp,5.5,"F",ha='center',fontsize=8,color='blue')

    elif stype == 'onde':
        lambda_val = schema_spec.get('lambda', 2.0); A_val = schema_spec.get('A', 1.0)
        x = np.linspace(0, 4*lambda_val, 500)
        y = A_val * np.sin(2*np.pi*x/lambda_val)
        ax.set_xlim(0, 4*lambda_val); ax.set_ylim(-A_val*2, A_val*2); ax.axis('on')
        ax.plot(x, y, 'b-', lw=2)
        ax.axhline(0, color='black', lw=0.5)
        ax.annotate('', xy=(lambda_val*2, A_val*1.4), xytext=(lambda_val, A_val*1.4),
                    arrowprops=dict(arrowstyle='<->', color='red', lw=1.5))
        ax.text(lambda_val*1.5, A_val*1.6, 'λ', ha='center', fontsize=11, color='red')
        ax.set_xlabel('Position x (m)'); ax.set_ylabel('Déplacement (m)')

    title = schema_spec.get('title', '')
    if title: ax.set_title(title, fontsize=9, pad=4)
    fig.tight_layout(pad=0.3)
    return fig_to_img(fig, w, h)

# ── CONSTRUCTION PDF ──────────────────────────────────────────────
def build_pdf(sujet: dict, prefs: dict) -> io.BytesIO:
    buf = io.BytesIO()
    ST = make_styles()

    # Préférences élève
    nom       = prefs.get('nom', '')
    prenom    = prefs.get('prenom', '')
    classe    = prefs.get('classe', 'Terminale')
    niveau    = prefs.get('niveau', 'Standard')
    couleur_theme = prefs.get('couleur', 'bleu')  # 'bleu', 'vert', 'rouge'

    color_map = {'bleu': BLEU, 'vert': VERT, 'rouge': ROUGE, 'violet': VIOLET}
    ACCENT = color_map.get(couleur_theme, BLEU)

    doc = SimpleDocTemplate(buf, pagesize=A4,
                            leftMargin=2*cm, rightMargin=2*cm,
                            topMargin=1.5*cm, bottomMargin=2*cm)
    story = []

    def hr(col=None, t=0.7):
        return HRFlowable(width='100%', thickness=t, color=col or ACCENT,
                          spaceAfter=4, spaceBefore=4)

    def part_header(num, title_text, pts):
        col = PART_COLORS.get(str(num), ACCENT)
        data = [[Paragraph(f'PARTIE {num}', ST['partT']),
                 Paragraph(title_text, ST['partT']),
                 Paragraph(f'{pts} pts', ST['partT'])]]
        t = Table(data, colWidths=[3*cm, 11*cm, 3*cm])
        t.setStyle(TableStyle([
            ('BACKGROUND',(0,0),(-1,-1), col),
            ('TOPPADDING',(0,0),(-1,-1), 7),
            ('BOTTOMPADDING',(0,0),(-1,-1), 7),
            ('LEFTPADDING',(0,0),(0,0), 10),
        ]))
        return t

    def sub_header(letter, title_text, pts):
        data = [[Paragraph(f'  {letter} — {title_text}', ST['subH']),
                 Paragraph(f'({pts} pts)', ST['note'])]]
        t = Table(data, colWidths=[14*cm, 3*cm])
        t.setStyle(TableStyle([
            ('BACKGROUND',(0,0),(-1,-1), BLEU_L),
            ('LINEBELOW',(0,0),(-1,-1), 0.5, ACCENT),
            ('TOPPADDING',(0,0),(-1,-1), 5),
            ('BOTTOMPADDING',(0,0),(-1,-1), 5),
            ('LEFTPADDING',(0,0),(0,0), 8),
        ]))
        return t

    def question_p(num, text, pts=None):
        suffix = f'  <i>({pts} pt{"s" if pts and pts>1 else ""})</i>' if pts else ''
        return Paragraph(f'<b>{num}.</b>  {text}{suffix}', ST['q'])

    def donnees_box(items):
        rows = []
        for item in items:
            texte = str(item)[:300]
            rows.append([Paragraph('•', ST['data']), Paragraph(texte, ST['data'])])
        if not rows:
            return Spacer(1, 0.1*cm)
        t = Table(rows, colWidths=[0.4*cm, 15*cm], repeatRows=0)
        t.setStyle(TableStyle([
            ('BACKGROUND',(0,0),(-1,-1), HexColor('#f5f5f5')),
            ('BOX',(0,0),(-1,-1), 0.5, ACCENT),
            ('TOPPADDING',(0,0),(-1,-1), 3),
            ('BOTTOMPADDING',(0,0),(-1,-1), 3),
            ('LEFTPADDING',(0,0),(0,0), 8),
            ('LEFTPADDING',(1,0),(1,0), 4),
        ]))
        return t
    # ── PAGE DE GARDE ──
    h1 = Table([[Paragraph('BACCALAURÉAT GÉNÉRAL', ST['title']),
                 Paragraph('ÉPREUVE DE PHYSIQUE-CHIMIE', ST['title'])]],
               colWidths=[8.5*cm, 8.5*cm])
    h1.setStyle(TableStyle([
        ('BACKGROUND',(0,0),(-1,-1), ACCENT),
        ('TOPPADDING',(0,0),(-1,-1), 10),
        ('BOTTOMPADDING',(0,0),(-1,-1), 10),
        ('LINEAFTER',(0,0),(0,0), 1, HexColor('#ffffff')),
    ]))
    story.append(h1)

    duree = sujet.get('duree', '3h30'); coeff = sujet.get('coefficient', '6')
    h2 = Table([[Paragraph('Série Générale — Spécialité', ST['sub']),
                 Paragraph(f'Durée : {duree}  —  Coefficient : {coeff}', ST['sub']),
                 Paragraph('Session 2025', ST['sub'])]],
               colWidths=[6*cm, 7*cm, 4*cm])
    h2.setStyle(TableStyle([
        ('BACKGROUND',(0,0),(-1,-1), HexColor('#2c5282')),
        ('TOPPADDING',(0,0),(-1,-1), 5), ('BOTTOMPADDING',(0,0),(-1,-1), 5),
    ]))
    story.append(h2)
    story.append(Spacer(1, 0.3*cm))

    # Thème
    theme_text = sujet.get('theme', 'Sciences et technologies')
    theme_t = Table([[Paragraph(f'<b>THÈME : {theme_text.upper()}</b>',
                                ParagraphStyle('th', fontName='Helvetica-Bold', fontSize=11,
                                              textColor=ACCENT, alignment=TA_CENTER))]],
                    colWidths=[17*cm])
    theme_t.setStyle(TableStyle([
        ('BOX',(0,0),(-1,-1), 1.5, ACCENT),
        ('TOPPADDING',(0,0),(-1,-1), 8), ('BOTTOMPADDING',(0,0),(-1,-1), 8),
        ('BACKGROUND',(0,0),(-1,-1), BLEU_L),
    ]))
    story.append(theme_t)
    story.append(Spacer(1, 0.3*cm))

    # Consignes standard
    consignes = [
        "L'usage de la calculatrice est autorisé (conformément à la circulaire n°99-186).",
        "Le candidat doit traiter toutes les parties. L'ordre de traitement est libre.",
        "La qualité de la rédaction et la clarté des raisonnements sont prises en compte.",
        "Les résultats numériques doivent être exprimés avec les unités et le bon nombre de chiffres significatifs.",
        "Les annexes sont à rendre avec la copie.",
    ]
    rows_c = [[Paragraph(f'⚬  {c}', ST['data'])] for c in consignes]
    tc = Table(rows_c, colWidths=[17*cm])
    tc.setStyle(TableStyle([
        ('BOX',(0,0),(-1,-1), 0.5, HexColor('#888888')),
        ('INNERGRID',(0,0),(-1,-1), 0.3, HexColor('#cccccc')),
        ('TOPPADDING',(0,0),(-1,-1), 3), ('BOTTOMPADDING',(0,0),(-1,-1), 3),
        ('LEFTPADDING',(0,0),(-1,-1), 8),
        ('BACKGROUND',(0,0),(-1,-1), HexColor('#fafafa')),
    ]))
    story.append(Paragraph('<b>CONSIGNES</b>',
                           ParagraphStyle('cl', fontName='Helvetica-Bold', fontSize=9,
                                         textColor=HexColor('#555'), spaceAfter=3)))
    story.append(tc)
    story.append(Spacer(1, 0.25*cm))

    # Tableau récap parties
    parties = sujet.get('parties', [])
    recap_data = [
        [Paragraph('<b>Partie</b>', ST['dataB']),
         Paragraph('<b>Thème</b>', ST['dataB']),
         Paragraph('<b>Notions</b>', ST['dataB']),
         Paragraph('<b>Points</b>', ST['dataB'])],
    ]
    for p in parties:
        recap_data.append([
            Paragraph(f"Partie {p.get('numero','')}", ST['data']),
            Paragraph(str(p.get('titre',''))[:80], ST['data']),
            Paragraph(str(p.get('notions',''))[:80], ST['data']),
            Paragraph(f"{p.get('points','')} pts", ST['data']),
        ])
    tr = Table(recap_data, colWidths=[3*cm, 5.5*cm, 5.5*cm, 2*cm])
    tr.setStyle(TableStyle([
        ('BACKGROUND',(0,0),(-1,0), ACCENT),
        ('TEXTCOLOR',(0,0),(-1,0), HexColor('#ffffff')),
        ('ROWBACKGROUNDS',(0,1),(-1,-1), [BLEU_L, HexColor('#ffffff')]),
        ('GRID',(0,0),(-1,-1), 0.4, HexColor('#aaaaaa')),
        ('TOPPADDING',(0,0),(-1,-1), 5), ('BOTTOMPADDING',(0,0),(-1,-1), 5),
        ('LEFTPADDING',(0,0),(-1,-1), 6),
    ]))
    story.append(tr)
    story.append(Spacer(1, 0.3*cm))

    # Contexte
    contexte = sujet.get('contexte', '')
    if contexte:
        story.append(Paragraph('<b>CONTEXTE GÉNÉRAL</b>',
                               ParagraphStyle('ctxl', fontName='Helvetica-Bold', fontSize=10,
                                             textColor=ACCENT, spaceAfter=4)))
        story.append(Paragraph(contexte, ST['body']))
    story.append(hr())

    # ── PARTIES ──
    annexes = []

    for partie in parties:
        num  = str(partie.get('numero', '1'))
        titr = partie.get('titre', '')
        pts  = partie.get('points', 0)
        intro = partie.get('intro', '')

        story.append(part_header(num, titr, pts))
        story.append(Spacer(1, 0.2*cm))
        if intro:
            story.append(Paragraph(intro, ST['body']))
            story.append(Spacer(1, 0.15*cm))

        # Sections dans la partie
        sections = partie.get('sections', [])
        for sec in sections:
            sec_letter = sec.get('lettre', 'A')
            sec_title  = sec.get('titre', '')
            sec_pts    = sec.get('points', 0)
            story.append(sub_header(sec_letter, sec_title, sec_pts))
            story.append(Spacer(1, 0.1*cm))

            # Données
            donnees = sec.get('donnees', [])
            if donnees:
                story.append(Paragraph('<b>Données :</b>', ST['dataB']))
                story.append(donnees_box(donnees))
                story.append(Spacer(1, 0.15*cm))

            # Schémas
            schemas = sec.get('schemas', [])
            for sch in schemas:
                try:
                    img = make_schema(sch)
                    story.append(img)
                    story.append(Spacer(1, 0.1*cm))
                except Exception as e:
                    story.append(Paragraph(f'[Schéma : {sch.get("title","")}]', ST['note']))

            # Graphiques
            graphiques = sec.get('graphiques', [])
            for gr in graphiques:
                try:
                    img = make_graph_generic(gr)
                    story.append(img)
                    story.append(Spacer(1, 0.1*cm))
                    # Si c'est une annexe à compléter
                    if gr.get('aussi_annexe', False):
                        gr_blank = dict(gr)
                        gr_blank['blank'] = True
                        gr_blank['title'] = f"Annexe — {gr.get('title','')}"
                        annexes.append({'type':'graph', 'spec': gr_blank,
                                        'question': gr.get('question_annexe','')})
                except Exception as e:
                    story.append(Paragraph(f'[Graphique : {gr.get("title","")}]', ST['note']))

            # Tableaux
            tableaux = sec.get('tableaux', [])
            for tab in tableaux:
                tab_title = tab.get('title','')
                headers   = tab.get('headers', [])
                rows_t    = tab.get('rows', [])
                if tab_title:
                    story.append(Paragraph(f'<b>{tab_title}</b>', ST['note']))
                all_rows = []
                if headers:
                    all_rows.append([Paragraph(f'<b>{h}</b>', ST['dataB']) for h in headers])
                for row in rows_t:
                    all_rows.append([Paragraph(str(c), ST['data']) for c in row])
                if all_rows:
                    ncols = len(all_rows[0])
                    col_w = 17*cm / ncols
                    tt = Table(all_rows, colWidths=[col_w]*ncols)
                    ts_style = [
                        ('GRID',(0,0),(-1,-1), 0.4, HexColor('#aaaaaa')),
                        ('TOPPADDING',(0,0),(-1,-1), 4),
                        ('BOTTOMPADDING',(0,0),(-1,-1), 4),
                        ('LEFTPADDING',(0,0),(-1,-1), 5),
                    ]
                    if headers:
                        ts_style.append(('BACKGROUND',(0,0),(-1,0), ACCENT))
                        ts_style.append(('TEXTCOLOR',(0,0),(-1,0), HexColor('#ffffff')))
                        ts_style.append(('ROWBACKGROUNDS',(0,1),(-1,-1), [HexColor('#ffffff'), BLEU_L]))
                    tt.setStyle(TableStyle(ts_style))
                    story.append(tt)
                    story.append(Spacer(1, 0.15*cm))
                if tab.get('aussi_annexe', False):
                    annexes.append({'type':'tableau', 'spec': tab,
                                    'question': tab.get('question_annexe','')})

            # Questions
            questions = sec.get('questions', [])
            for q in questions:
                q_num  = q.get('numero', '')
                q_text = q.get('texte', '')
                q_pts  = q.get('points', None)
                q_lines = q.get('lignes_reponse', 3)
                story.append(question_p(q_num, q_text, q_pts))
                story.append(AnswerBox(q_lines, f'Réponse {q_num}'))
                story.append(Spacer(1, 0.1*cm))

        story.append(hr())

    # ── ANNEXES ──
    if annexes:
        story.append(PageBreak())
        annex_h = Table([[Paragraph('ANNEXES — À RENDRE AVEC LA COPIE',
                                    ParagraphStyle('ah', fontName='Helvetica-Bold', fontSize=12,
                                                  textColor=HexColor('#ffffff'), alignment=TA_CENTER))]],
                        colWidths=[17*cm])
        annex_h.setStyle(TableStyle([
            ('BACKGROUND',(0,0),(-1,-1), OR),
            ('TOPPADDING',(0,0),(-1,-1), 10), ('BOTTOMPADDING',(0,0),(-1,-1), 10),
        ]))
        story.append(annex_h)
        story.append(Spacer(1, 0.3*cm))

        # Cadre candidat
        cand = Table([
            [Paragraph(f'NOM : {nom if nom else "________________________________"}', ST['data']),
             Paragraph(f'Prénom : {prenom if prenom else "________________________________"}', ST['data'])],
            [Paragraph('N° candidat : ___________', ST['data']),
             Paragraph(f'Classe : {classe}', ST['data'])],
        ], colWidths=[8.5*cm, 8.5*cm])
        cand.setStyle(TableStyle([
            ('BOX',(0,0),(-1,-1), 1, ACCENT),
            ('INNERGRID',(0,0),(-1,-1), 0.5, HexColor('#aaaaaa')),
            ('TOPPADDING',(0,0),(-1,-1), 8), ('BOTTOMPADDING',(0,0),(-1,-1), 8),
            ('LEFTPADDING',(0,0),(-1,-1), 8),
        ]))
        story.append(cand)
        story.append(Spacer(1, 0.4*cm))

        for i, ann in enumerate(annexes, 1):
            q_label = ann.get('question','')
            story.append(Paragraph(f'ANNEXE {i}{" — "+q_label if q_label else ""}', ST['annex']))
            story.append(Spacer(1, 0.1*cm))
            if ann['type'] == 'graph':
                try:
                    img = make_graph_generic(ann['spec'])
                    story.append(img)
                except:
                    story.append(Paragraph('[Graphe à compléter]', ST['note']))
            elif ann['type'] == 'tableau':
                spec = ann['spec']
                headers = spec.get('headers',[])
                rows_ann = []
                if headers:
                    rows_ann.append([Paragraph(f'<b>{h}</b>', ST['dataB']) for h in headers])
                for row in spec.get('rows_vides',[]):
                    rows_ann.append([Paragraph(str(c), ST['data']) for c in row])
                if rows_ann:
                    ncols = len(rows_ann[0])
                    ta = Table(rows_ann, colWidths=[17*cm/ncols]*ncols)
                    ta.setStyle(TableStyle([
                        ('GRID',(0,0),(-1,-1), 0.5, HexColor('#aaaaaa')),
                        ('BACKGROUND',(0,0),(-1,0), ACCENT),
                        ('TEXTCOLOR',(0,0),(-1,0), HexColor('#ffffff')),
                        ('TOPPADDING',(0,0),(-1,-1), 8),
                        ('BOTTOMPADDING',(0,0),(-1,-1), 8),
                        ('LEFTPADDING',(0,0),(-1,-1), 6),
                    ]))
                    story.append(ta)
            story.append(DottedLine())
            story.append(Spacer(1, 0.3*cm))

    # Footer
    story.append(HRFlowable(width='100%', thickness=0.5, color=HexColor('#aaaaaa'),
                             spaceAfter=4, spaceBefore=4))
    n_pages = len(parties) + (2 if annexes else 0)
    story.append(Paragraph(
        f'Baccalauréat Général — Épreuve de Physique-Chimie — Session 2025 — '
        f'Ce sujet comporte {n_pages} pages. Les annexes sont à rendre avec la copie. '
        f'{"Niveau : "+niveau if niveau != "Standard" else ""}',
        ST['footer']))

    doc.build(story)
    buf.seek(0)
    return buf

# ── ENDPOINT PRINCIPAL ─────────────────────────────────────────────
@app.route('/generate-pdf', methods=['POST'])
def generate_pdf():
    data = request.get_json()
    if not data:
        return jsonify({'error': 'JSON manquant'}), 400

    sujet_json = data.get('sujet')       # JSON structuré du sujet
    prefs      = data.get('prefs', {})   # préférences élève

    if not sujet_json:
        return jsonify({'error': 'Sujet manquant'}), 400

    try:
        if isinstance(sujet_json, str):
            sujet_json = json.loads(sujet_json)

        pdf_buf = build_pdf(sujet_json, prefs)
        theme = sujet_json.get('theme', 'sujet')[:30].replace(' ', '_')
        filename = f"PhysiIA_{theme}.pdf"

        return send_file(
            pdf_buf,
            mimetype='application/pdf',
            as_attachment=True,
            download_name=filename
        )
    except Exception as e:
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500

@app.route('/health', methods=['GET'])
def health():
    return jsonify({'status': 'ok', 'service': 'PhysiIA PDF Generator'})

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False)
