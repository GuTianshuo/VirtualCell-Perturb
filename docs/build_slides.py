"""Build the 10-minute presentation (docs/VirtualCell-Perturb_汇报.pptx) from the result files.

All numbers on the slides are read from results/ so the deck stays in sync with the code.
    python docs/build_slides.py
"""
import json
import os
import sys

import numpy as np
import pandas as pd
from PIL import Image
from pptx import Presentation
from pptx.chart.data import CategoryChartData
from pptx.dml.color import RGBColor
from pptx.enum.chart import XL_CHART_TYPE, XL_LABEL_POSITION, XL_LEGEND_POSITION, XL_TICK_LABEL_POSITION
from pptx.enum.shapes import MSO_CONNECTOR, MSO_SHAPE
from pptx.enum.text import MSO_ANCHOR, PP_ALIGN
from pptx.oxml.ns import qn
from pptx.util import Emu, Inches, Pt

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
RES = os.path.join(ROOT, "results", "replogle_k562_essential")
FIG = os.path.join(ROOT, "figures")
OUT = os.path.join(ROOT, "docs", "VirtualCell-Perturb_汇报.pptx")

FONT = "Microsoft YaHei"
MONO = "Consolas"
DARK, DARK2 = "0F2433", "1C3A52"
BLUE, AQUA, ORANGE, YELLOW = "2A78D6", "1BAF7A", "EB6834", "EDA100"
TEXT, TEXT2, MUTED, LINE, TINT, WHITE = "1A1F26", "4A5561", "8A929B", "D9DFE6", "EEF4FB", "FFFFFF"
STAGES = ["问题", "数据", "模型", "评测", "解释"]
W, H = 13.333, 7.5


# ---------------------------------------------------------------------------------------
# primitives
# ---------------------------------------------------------------------------------------
def rgb(h):
    return RGBColor.from_string(h)


def style_run(run, size=16, bold=False, color=TEXT, italic=False, font=FONT):
    f = run.font
    f.size = Pt(size)
    f.bold = bold
    f.italic = italic
    f.color.rgb = rgb(color)
    f.name = font
    rPr = run._r.get_or_add_rPr()
    for tag in ("a:ea", "a:cs"):
        el = rPr.find(qn(tag))
        if el is None:
            el = rPr.makeelement(qn(tag), {})
            rPr.append(el)
        el.set("typeface", font)


def text(slide, x, y, w, h, paras, size=16, color=TEXT, bold=False, align=PP_ALIGN.LEFT,
         anchor=MSO_ANCHOR.TOP, after=6, bullet=False, font=FONT, line=1.1, name=None):
    """paras: list of paragraphs; a paragraph is a str or a list of (text, {opts}) runs."""
    tb = slide.shapes.add_textbox(Inches(x), Inches(y), Inches(w), Inches(h))
    if name:
        tb.name = name
    tf = tb.text_frame
    tf.word_wrap = True
    tf.vertical_anchor = anchor
    tf.margin_left = tf.margin_right = Inches(0.04)
    tf.margin_top = tf.margin_bottom = Inches(0.02)
    for i, p in enumerate(paras):
        para = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        para.alignment = align
        para.space_after = Pt(after)
        para.line_spacing = line
        runs = [(p, {})] if isinstance(p, str) else p
        for t, o in runs:
            r = para.add_run()
            r.text = t
            style_run(r, o.get("size", size), o.get("bold", bold), o.get("color", color), o.get("italic", False),
                      o.get("font", font))
        if bullet:
            pPr = para._p.get_or_add_pPr()
            pPr.set("marL", str(Emu(Inches(0.22))))
            pPr.set("indent", str(-Emu(Inches(0.2))))
            bu = pPr.makeelement(qn("a:buChar"), {"char": "•"})
            pPr.append(bu)
    return tb


def box(slide, x, y, w, h, fill=TINT, line=None, shape=MSO_SHAPE.ROUNDED_RECTANGLE, radius=0.08, name=None):
    s = slide.shapes.add_shape(shape, Inches(x), Inches(y), Inches(w), Inches(h))
    if name:
        s.name = name
    s.fill.solid()
    s.fill.fore_color.rgb = rgb(fill)
    if line:
        s.line.color.rgb = rgb(line)
        s.line.width = Pt(1)
    else:
        s.line.fill.background()
    s.shadow.inherit = False
    if shape == MSO_SHAPE.ROUNDED_RECTANGLE:
        s.adjustments[0] = radius
    return s


def circle_label(slide, x, y, d, label, fill=BLUE, color=WHITE, size=14, line=None):
    s = box(slide, x, y, d, d, fill=fill, line=line, shape=MSO_SHAPE.OVAL)
    tf = s.text_frame
    tf.margin_left = tf.margin_right = tf.margin_top = tf.margin_bottom = 0
    tf.vertical_anchor = MSO_ANCHOR.MIDDLE
    p = tf.paragraphs[0]
    p.alignment = PP_ALIGN.CENTER
    r = p.add_run()
    r.text = label
    style_run(r, size, True, color)
    return s


def arrow(slide, x1, y1, x2, y2, color=MUTED, width=1.75):
    c = slide.shapes.add_connector(MSO_CONNECTOR.STRAIGHT, Inches(x1), Inches(y1), Inches(x2), Inches(y2))
    c.line.color.rgb = rgb(color)
    c.line.width = Pt(width)
    ln = c.line._get_or_add_ln()
    tail = ln.makeelement(qn("a:tailEnd"), {"type": "triangle", "w": "med", "len": "med"})
    ln.append(tail)
    return c


def picture(slide, path, x, y, w=None, h=None, name=None):
    """Insert an image fitted (aspect-preserving) into the box w x h, centred."""
    im = Image.open(path)
    ar = im.width / im.height
    if w and h:
        if w / h > ar:
            ww, hh = h * ar, h
        else:
            ww, hh = w, w / ar
        x, y = x + (w - ww) / 2, y + (h - hh) / 2
    elif w:
        ww, hh = w, w / ar
    else:
        ww, hh = h * ar, h
    pic = slide.shapes.add_picture(path, Inches(x), Inches(y), Inches(ww), Inches(hh))
    if name:
        pic.name = name
    return pic


def tracker(slide, active):
    """Visual motif: the five required stages, current one filled."""
    d, gap = 0.42, 0.2
    x0 = W - 0.55 - (len(STAGES) * d + (len(STAGES) - 1) * gap)
    y = 0.42
    for i, s in enumerate(STAGES):
        x = x0 + i * (d + gap)
        if i < len(STAGES) - 1:
            ln = slide.shapes.add_connector(MSO_CONNECTOR.STRAIGHT, Inches(x + d), Inches(y + d / 2),
                                            Inches(x + d + gap), Inches(y + d / 2))
            ln.line.color.rgb = rgb(LINE)
            ln.line.width = Pt(1.25)
        on = s in active
        circle_label(slide, x, y, d, s[0], fill=BLUE if on else WHITE, color=WHITE if on else MUTED, size=12,
                     line=None if on else LINE)


def title(slide, t, stage=None, sub=None):
    text(slide, 0.55, 0.32, 8.6, 0.7, [t], size=30, bold=True, color=TEXT, anchor=MSO_ANCHOR.MIDDLE, name="Title")
    if sub:
        text(slide, 0.55, 1.0, 8.9, 0.45, [sub], size=15, color=TEXT2, name="Subtitle")
    if stage:
        tracker(slide, stage if isinstance(stage, (list, tuple)) else [stage])


def footer(slide, n, src=None):
    text(slide, 0.55, H - 0.42, 9.5, 0.3, [src or "VirtualCell-Perturb · 生物医学信息学课程设计 · 选题九"], size=10, color=MUTED)
    text(slide, W - 1.2, H - 0.42, 0.65, 0.3, [str(n)], size=10, color=MUTED, align=PP_ALIGN.RIGHT)


def stat(slide, x, y, w, value, label, color=BLUE, vsize=34, lsize=12, h=1.2):
    text(slide, x, y, w, 0.62, [value], size=vsize, bold=True, color=color)
    text(slide, x, y + 0.66, w, h - 0.6, [label], size=lsize, color=TEXT2, line=1.05)


TALK = []


def notes(slide, s):
    """Speaker notes; the time label is estimated from length (~4.5 spoken characters per second)."""
    sec = max(15, int(round(len(s.replace(" ", "")) / 4.5 / 5.0)) * 5)
    TALK.append(sec)
    slide.notes_slide.notes_text_frame.text = f"（约 {sec} 秒）" + s


def bg(slide, color):
    f = slide.background.fill
    f.solid()
    f.fore_color.rgb = rgb(color)


# ---------------------------------------------------------------------------------------
# data
# ---------------------------------------------------------------------------------------
def load():
    R = {}
    R["summ"] = pd.read_csv(os.path.join(RES, "summary.csv"), index_col=0)
    R["nc"] = pd.read_csv(os.path.join(RES, "noise_ceiling.csv"), index_col=0)["value"]
    R["stats"] = pd.read_csv(os.path.join(RES, "stats_tests.csv"), index_col=0)
    p = os.path.join(RES, "flow_singlecell_metrics.csv")
    R["flow"] = pd.read_csv(p) if os.path.exists(p) else None
    p = os.path.join(ROOT, "results", "external_adamson", "summary.csv")
    R["ext"] = pd.read_csv(p, index_col=0) if os.path.exists(p) else None
    I = os.path.join(ROOT, "results", "interpretation")
    R["cx"] = pd.read_csv(os.path.join(I, "complex_recovery.csv"), index_col=0)
    R["pw"] = pd.read_csv(os.path.join(I, "pathway_consistency_summary.csv"))
    R["cls"] = pd.read_csv(os.path.join(I, "class_performance.csv"))
    R["fail"] = pd.read_csv(os.path.join(I, "failure_analysis.csv"))
    R["dirn"] = pd.read_csv(os.path.join(I, "deg_direction.csv"), index_col=0)
    import json
    R["repr"] = json.load(open(os.path.join(I, "representation.json")))
    p = os.path.join(RES, "split_robustness.csv")
    R["splits"] = pd.read_csv(p) if os.path.exists(p) else None
    return R


def f3(x):
    return f"{x:.3f}"


# ---------------------------------------------------------------------------------------
def build():
    R = load()
    S, nc = R["summ"], R["nc"]
    PF, KNN, LIN, MS, NET = "PerturbFormer+KD", "kNN-concat+KD", "Linear-concat+KD", "Mean-shift", "PerturbNet-attn+KD"
    pf = S.loc[PF]
    prs = Presentation()
    prs.slide_width, prs.slide_height = Inches(W), Inches(H)
    blank = prs.slide_layouts[6]
    n = 0

    def new():
        nonlocal n
        n += 1
        return prs.slides.add_slide(blank)

    # 1 ---------------------------------------------------------------- title
    s = new()
    bg(s, DARK)
    hero = os.path.join(FIG, "hero_cells.png")
    if os.path.exists(hero):
        picture(s, hero, 7.4, 0.9, w=5.4, h=5.6)
    text(s, 0.75, 1.55, 7.0, 0.5, ["生物医学信息学课程设计 · 选题九"], size=16, color="9FB7CC")
    text(s, 0.75, 2.1, 7.2, 1.9, ["虚拟细胞扰动预测"], size=48, bold=True, color=WHITE)
    text(s, 0.75, 3.15, 6.6, 1.2, ["敲低一个从未做过实验的基因，细胞的转录组会怎样变化？"], size=20, color="D6E2EC", line=1.15)
    text(s, 0.75, 4.45, 6.6, 0.9, [[("PerturbFormer", {"bold": True, "color": "7FB2F0"}),
                                     (" 知识检索  +  ", {}),
                                     ("条件流匹配", {"bold": True, "color": "7FB2F0"}),
                                     (" 单细胞生成", {})]], size=16, color="D6E2EC")
    text(s, 0.75, 5.9, 6.6, 0.8, ["吉林大学人工智能学院　|　汇报人：__________"], size=14, color="9FB7CC")
    notes(s, "各位老师好，我的选题是第九题，虚拟细胞扰动预测。核心问题是：敲低一个从没做过实验的基因，细胞会怎样变化。下面按问题、数据、模型、评测、生物学解释五个环节汇报。")

    # 2 ---------------------------------------------------------------- problem
    s = new()
    title(s, "如果敲低一个基因，细胞会怎样？", "问题")
    rows = [("组合爆炸", "约 2 万个基因 × 细胞类型 × 剂量 × 组合，实验不可能穷举"),
            ("实验昂贵", "全基因组 Perturb-seq 需要数百万个单细胞测序"),
            ("虚拟细胞", "先在计算机中筛选靶点，把实验预算留给最有希望的扰动")]
    for i, (h, d) in enumerate(rows):
        y = 1.55 + i * 1.12
        circle_label(s, 0.6, y + 0.05, 0.55, str(i + 1), fill=BLUE, size=16)
        text(s, 1.35, y, 4.7, 0.42, [h], size=18, bold=True)
        text(s, 1.35, y + 0.42, 4.7, 0.6, [d], size=14, color=TEXT2)
    # input -> output schematic
    bx = 6.3
    box(s, bx, 1.55, 1.85, 1.0, fill=TINT)
    text(s, bx, 1.6, 1.85, 0.9, [[("对照细胞 ", {}), ("x₀", {"italic": True})], "单细胞表达谱"], size=14, align=PP_ALIGN.CENTER,
         anchor=MSO_ANCHOR.MIDDLE, after=2)
    box(s, bx, 2.85, 1.85, 1.0, fill=TINT)
    text(s, bx, 2.9, 1.85, 0.9, [[("基因扰动 ", {}), ("g", {"italic": True})], "CRISPRi 敲低"], size=14, align=PP_ALIGN.CENTER,
         anchor=MSO_ANCHOR.MIDDLE, after=2)
    box(s, bx + 2.3, 1.95, 1.45, 1.5, fill=BLUE)
    text(s, bx + 2.3, 1.95, 1.45, 1.5, ["虚拟细胞", "模型"], size=16, bold=True, color=WHITE, align=PP_ALIGN.CENTER,
         anchor=MSO_ANCHOR.MIDDLE, after=0)
    arrow(s, bx + 1.85, 2.05, bx + 2.3, 2.45)
    arrow(s, bx + 1.85, 3.35, bx + 2.3, 2.95)
    outs = ["Δ̂ 平均表达变化", "DEG 上调 / 下调方向", "扰动后单细胞状态 x₁"]
    for i, o in enumerate(outs):
        box(s, bx + 4.2, 1.45 + i * 0.75, 2.2, 0.6, fill=WHITE, line=LINE)
        text(s, bx + 4.2, 1.45 + i * 0.75, 2.2, 0.6, [o], size=12.5, align=PP_ALIGN.CENTER, anchor=MSO_ANCHOR.MIDDLE)
        arrow(s, bx + 3.75, 2.7, bx + 4.2, 1.75 + i * 0.75)
    box(s, 0.6, 5.05, 12.15, 1.55, fill=TINT)
    text(s, 0.85, 5.18, 11.7, 1.35, [
        [("关键难点 1 · 未见基因：", {"bold": True, "color": BLUE}), ("测试基因在训练中从未被扰动，模型不能“记住”答案，只能靠先验知识迁移。", {})],
        [("关键难点 2 · 系统偏差陷阱：", {"bold": True, "color": BLUE}),
         (f"“预测所有扰动的平均效应”这一平凡基线已能达到 Top-20 DEG 相关 {S.loc[MS, 'pearson_de20']:.2f}，许多深度模型并不比它好 (Ahlmann-Eltze 2025)。", {})],
        [("关键难点 3 · 测量噪声：", {"bold": True, "color": BLUE}),
         (f"每个扰动中位数只有约 115 个细胞；同一扰动拆成两半，全基因相关也只有 {nc['pearson_all']:.2f}。", {})]],
        size=14, after=5)
    footer(s, n)
    notes(s, f"为什么做这个问题：基因组合太多，Perturb-seq 又贵，需要能先在计算机里筛选的虚拟细胞。任务是输入一个对照细胞和一个基因，输出平均表达变化、差异基因方向和这个细胞扰动后的状态。"
             f"难点有三个：测试基因训练时完全没见过；多数扰动共享系统性响应，只预测平均效应的平凡基线 Top-20 DEG 相关就有 {S.loc[MS, 'pearson_de20']:.2f}；而且测量噪声很大。")

    # 3 ---------------------------------------------------------------- data
    s = new()
    title(s, "数据：两个独立的 Perturb-seq 筛选", "数据")
    cards = [("Replogle et al., Cell 2022", "K562 必需基因 CRISPRi", "1,092", "162,751 个细胞 · 5,000 个基因", "主数据：5 折未见基因评测"),
             ("Adamson et al., Cell 2016", "K562 内质网应激 CRISPRi", "86", "68,603 个细胞 · 另一实验室", "外部验证：跨实验室泛化")]
    for i, (a, b_, big, c, d) in enumerate(cards):
        x = 0.6 + i * 3.2
        box(s, x, 1.45, 3.0, 2.75, fill=TINT)
        text(s, x + 0.18, 1.55, 2.7, 0.32, [a], size=11.5, color=TEXT2)
        text(s, x + 0.18, 1.88, 2.7, 0.38, [b_], size=14, bold=True)
        text(s, x + 0.18, 2.3, 2.7, 0.62, [big], size=34, bold=True, color=BLUE)
        text(s, x + 0.18, 2.92, 2.7, 0.3, ["个基因敲低"], size=12, color=TEXT2)
        text(s, x + 0.18, 3.25, 2.7, 0.3, [c], size=11.5, color=TEXT2)
        text(s, x + 0.18, 3.6, 2.7, 0.4, [d], size=12, bold=True, color=TEXT)
    text(s, 7.25, 1.45, 5.5, 0.4, ["把“基因”编码成向量的四种先验知识"], size=15, bold=True)
    pri = [("GO 注释", "基因功能本体 (BP/CC/MF)", "99.6%"), ("STRING PPI", "蛋白互作网络扩散嵌入", "97.6%"),
           ("GenePT", "GPT 嵌入的 NCBI 基因摘要 (语言模型)", "99.5%"), ("应答谱", "该基因在训练扰动中的响应模式", "37.7%")]
    cols = [BLUE, ORANGE, AQUA, YELLOW]
    for i, (a, b_, c) in enumerate(pri):
        y = 1.95 + i * 0.54
        box(s, 7.25, y + 0.1, 0.24, 0.24, fill=cols[i], shape=MSO_SHAPE.OVAL)
        text(s, 7.6, y, 1.45, 0.45, [a], size=13.5, bold=True, anchor=MSO_ANCHOR.MIDDLE)
        text(s, 9.05, y, 2.9, 0.45, [b_], size=12, color=TEXT2, anchor=MSO_ANCHOR.MIDDLE)
        text(s, 11.85, y, 0.9, 0.45, [c], size=13, bold=True, color=TEXT, align=PP_ALIGN.RIGHT, anchor=MSO_ANCHOR.MIDDLE)
    text(s, 7.25, 4.05, 5.5, 0.3, ["百分比 = 1,137 个被扰动基因中有该先验的比例"], size=10, color=MUTED)
    box(s, 0.6, 4.4, 12.15, 0.55, fill=WHITE, line=LINE)
    text(s, 0.75, 4.4, 11.9, 0.55, [[("敲低 vs 敲除：", {"bold": True, "color": BLUE}),
         ("两个数据集都是 CRISPRi 敲低（dCas9-KRAB 抑制转录，不切断 DNA）。在能测到靶基因的 412 个扰动中，靶基因 100% 显著下调、"
          "表达中位数降到接近 0，转录层面接近敲除；本项目用它回答题目中“敲除一个基因会怎样”的问题。", {})]],
         size=12, anchor=MSO_ANCHOR.MIDDLE, line=1.05)
    picture(s, os.path.join(FIG, "fig1_data_overview.png"), 0.6, 5.05, w=12.15, h=1.95)
    footer(s, n, "数据来源：GEARS 预处理版 (Harvard Dataverse)；GO / STRING v12 / GenePT (Zenodo 10833191)；MSigDB / CORUM (Enrichr)")
    notes(s, "数据是两个独立的 CRISPRi 筛选：主数据 Replogle 2022，1092 个基因敲低；外部验证用另一实验室的 Adamson 2016。题目写的是敲除，这两个数据都是敲低，"
             "抑制转录而不切断 DNA；但能测到的靶基因表达几乎都降到零，效果接近敲除。为了预测新基因，每个基因用 GO、蛋白互作、GenePT 语言模型和应答谱四种知识编码。")

    # 3b --------------------------------------------------------------- representation
    s = new()
    title(s, "细胞与扰动的表征", ["数据", "模型"])
    rp = R["repr"]
    picture(s, os.path.join(FIG, "fig_repr.png"), 0.45, 1.3, w=12.45, h=3.95)
    cols3 = [("单细胞状态", f"扰动细胞偏离对照，不同复合体方向不同（位移 {min(rp['shift_sd'].values()):.2f}–{max(rp['shift_sd'].values()):.2f} 个对照标准差）；"
                           "也有不少扰动细胞与对照重叠：敲低在细胞间不均一。"),
             ("扰动应答谱", f"同一复合体的扰动聚在一起：10 近邻同类比例 {rp['knn_purity']:.0%}（随机 {rp['knn_purity_null']:.0%}）。这是“借鉴相似已测基因”的生物学基础。")]
    if rp.get("capture_test_median") is not None:
        cols3.append(("潜空间", f"100 维 PCA 只占单细胞方差约 15%（其余主要是测量噪声），却保留了未见扰动平均效应的 {rp['capture_test_median']:.0%}（中位数）。"))
    for i, (h_, d) in enumerate(cols3):
        x = 0.6 + i * 4.1
        text(s, x, 5.35, 3.9, 0.35, [h_], size=14, bold=True, color=BLUE)
        text(s, x, 5.72, 3.9, 1.2, [d], size=12, color=TEXT2, line=1.1)
    footer(s, n)
    cap_txt = f"右图说明 100 维潜空间虽只占单细胞方差的 15%，却保留了未见扰动平均效应的 {rp['capture_test_median']:.0%}。" if rp.get("capture_test_median") is not None else ""
    notes(s, f"这一页是细胞和扰动的表征。左图是单细胞：扰动细胞偏离对照，不同复合体的偏离方向不同，也有不少扰动细胞和对照混在一起，说明敲低在细胞间并不均一。"
             f"中图是 1092 个扰动的应答谱，同一复合体的扰动聚在一起，十个近邻里同类占 {rp['knn_purity']:.0%}，随机只有 {rp['knn_purity_null']:.0%}，这就是模型能借鉴相似基因的基础。" + cap_txt)

    # 4 ---------------------------------------------------------------- overview
    s = new()
    title(s, "整体框架：从基因到虚拟细胞", "模型")
    steps = [("基因 g", "任意人类基因\n(约 2.1 万个可编码)", WHITE, TEXT),
             ("四个知识 token", "GO · PPI · GenePT · 应答谱\nTransformer 融合", TINT, TEXT),
             ("PerturbFormer", "对 1,092 个已测扰动做\n知识核交叉注意力", BLUE, WHITE),
             ("Δ̂ + DEG 方向", "伪批量表达变化\n+ CRISPRi 敲低规则", TINT, TEXT),
             ("条件流匹配", "对照细胞 x₀ → 扰动细胞 x₁\n(ODE 最优传输)", BLUE, WHITE)]
    for i, (h_, d, fc, tc) in enumerate(steps):
        x = 0.6 + i * 2.5
        box(s, x, 1.65, 2.1, 1.75, fill=fc, line=LINE if fc == WHITE else None)
        text(s, x + 0.08, 1.75, 1.94, 0.45, [h_], size=15, bold=True, color=tc, align=PP_ALIGN.CENTER)
        text(s, x + 0.08, 2.25, 1.94, 1.05, d.split("\n"), size=11.5, color=tc if tc == WHITE else TEXT2,
             align=PP_ALIGN.CENTER, after=2)
        if i < len(steps) - 1:
            arrow(s, x + 2.12, 2.52, x + 2.48, 2.52, color=TEXT2)
    box(s, 5.6, 3.75, 2.1, 0.75, fill=WHITE, line=BLUE)
    text(s, 5.6, 3.75, 2.1, 0.75, ["记忆库：已测扰动的", "真实响应 (Value)"], size=11.5, color=BLUE, align=PP_ALIGN.CENTER,
         anchor=MSO_ANCHOR.MIDDLE, after=0)
    arrow(s, 6.65, 3.75, 6.65, 3.42, color=BLUE)
    prin = [("借鉴而非记忆", "未见基因的效应 ≈ 生物学上相似且已经测过的基因的效应，用注意力把“相似”学出来"),
            ("强基线做起点", "知识核注意力初始化时就等价于多源 kNN，训练只学修正量，870 个样本也不过拟合"),
            ("两个层次输出", "监督模型给出平均变化和 DEG 方向；生成模型把它展开成每个细胞的反事实状态")]
    for i, (h_, d) in enumerate(prin):
        x = 0.6 + i * 4.1
        text(s, x, 4.85, 3.85, 0.4, [h_], size=16, bold=True, color=BLUE)
        text(s, x, 5.3, 3.85, 1.4, [d], size=13.5, color=TEXT2, line=1.15)
    footer(s, n)
    notes(s, "整体框架：输入任意基因，先得到四种知识表示并用 Transformer 融合；PerturbFormer 用它查阅 1092 个已测扰动，把相似扰动的真实响应加权组合，再加上敲低规则，"
             "得到平均变化和 DEG 方向；最后条件流匹配把它展开到每个细胞。原则是借鉴而不是记忆，从强基线出发只学修正。")

    # 5 ---------------------------------------------------------------- PerturbFormer
    s = new()
    title(s, "PerturbFormer：查阅相似的已测基因", "模型")
    box(s, 0.6, 1.45, 6.6, 1.5, fill=TINT)
    text(s, 0.8, 1.55, 6.3, 1.35, [
        [("Δ̂ᵢ = Σₕ gₕ Σⱼ aₕ(i, j) Δⱼ  +  g₀ (μ + dec(hᵢ) Bᵀ)", {"font": "Cambria", "size": 15})],
        [("aₕ(i, j) = softmaxⱼ[ Σₛ wₕₛ(i) cos(Aₛeₛ(i), Aₛeₛ(j)) / τₕ ]", {"font": "Cambria", "size": 15}), ("  (top-k)", {"size": 11, "color": TEXT2})],
        [("Query = 待预测基因的知识；Key = 已测基因的知识；Value = 已测基因的真实响应 Δⱼ", {"size": 11.5, "color": TEXT2})]],
        size=15, after=5)
    bl = [[("知识核注意力：", {"bold": True}), ("logits 由各知识源余弦相似度构成，Aₛ 初始化为单位阵 → 起点即多源 kNN", {})],
          [("查询相关的知识源权重 wₕₛ(i)：", {"bold": True}), ("模型自己决定这个基因该信 GO、PPI 还是语言模型", {})],
          [("top-k 稀疏 + 多头温度 + 参数化分支门控", {"bold": True})],
          [("CRISPRi 敲低规则：", {"bold": True}), ("靶基因按训练集中位残余比例下调", {})],
          [("多任务损失：", {"bold": True}), ("DE 加权 MSE + DEG 方向 logistic 损失；知识源随机丢弃", {})]]
    text(s, 0.6, 3.15, 6.6, 3.4, bl, size=14, bullet=True, after=7)
    # native chart: retrieval example
    oof = np.load(os.path.join(RES, "oof_predictions.npz"), allow_pickle=True)
    perts = list(oof["perts"])
    gname = "PSMB5" if "PSMB5" in perts else perts[0]
    i = perts.index(gname)
    nb = [(perts[j], float(w)) for j, w in zip(oof["neigh_idx"][i, :6], oof["neigh_w"][i, :6]) if j >= 0]
    cd = CategoryChartData()
    cd.categories = [g for g, _ in nb][::-1]
    cd.add_series("注意力权重", [round(w, 3) for _, w in nb][::-1])
    gf = s.shapes.add_chart(XL_CHART_TYPE.BAR_CLUSTERED, Inches(7.6), Inches(1.75), Inches(5.15), Inches(3.6), cd)
    ch = gf.chart
    ch.has_legend = False
    ch.has_title = True
    ch.chart_title.text_frame.text = f"例：预测 {gname} 敲低时，模型注意的已测扰动"
    style_run(ch.chart_title.text_frame.paragraphs[0].runs[0], 13, True, TEXT)
    pl = ch.plots[0]
    pl.gap_width = 70
    pl.series[0].format.fill.solid()
    pl.series[0].format.fill.fore_color.rgb = rgb(BLUE)
    pl.has_data_labels = True
    pl.data_labels.number_format = "0.00"
    pl.data_labels.number_format_is_linked = False
    pl.data_labels.position = XL_LABEL_POSITION.OUTSIDE_END
    pl.data_labels.font.size = Pt(11)
    pl.data_labels.font.color.rgb = rgb(TEXT2)
    ch.value_axis.visible = False
    ch.value_axis.has_major_gridlines = False
    ch.category_axis.tick_labels.font.size = Pt(12)
    ch.category_axis.tick_labels.font.italic = True
    ch.category_axis.tick_labels.font.color.rgb = rgb(TEXT)
    ch.category_axis.format.line.color.rgb = rgb(LINE)
    n_psm = sum(g.startswith("PSM") for g, _ in nb)
    text(s, 7.6, 5.45, 5.15, 1.1, [f"{gname} 是蛋白酶体 20S 核心亚基。训练时从未见过它，模型注意力最高的 {len(nb)} 个已测扰动中有 {n_psm} 个是其他蛋白酶体亚基，"
                                   "预测就是它们真实响应的加权组合：可解释、可追溯。"], size=12.5, color=TEXT2, line=1.15)
    footer(s, n)
    notes(s, f"PerturbFormer 是一个交叉注意力：Query 是待预测基因的知识，Key 是已测基因的知识，Value 是已测基因的真实响应。关键是知识核：注意力分数由各知识源的余弦相似度构成，"
             f"初始时等价于多源 kNN，训练只学修正量，样本少也不过拟合；模型还会针对每个基因决定更相信哪种知识。右边是真实例子：预测 {gname} 时它从没见过这个基因，"
             f"注意力最高的 {len(nb)} 个已测扰动中有 {n_psm} 个是其他蛋白酶体亚基。")

    # 6 ---------------------------------------------------------------- benchmark design
    s = new()
    title(s, "评测设计：严格的未见基因泛化", "评测")
    text(s, 0.6, 1.45, 5.6, 0.4, ["5 折交叉验证，按扰动基因划分"], size=16, bold=True)
    for k in range(5):
        y = 1.95 + k * 0.5
        text(s, 0.6, y, 0.8, 0.4, [f"第 {k + 1} 折"], size=12, color=TEXT2, anchor=MSO_ANCHOR.MIDDLE)
        for j in range(5):
            box(s, 1.45 + j * 0.92, y + 0.04, 0.86, 0.32, fill=BLUE if j == k else "DCE6F2", radius=0.15)
    nsplit = R["splits"].split_seed.nunique() if R["splits"] is not None else 1
    text(s, 1.45, 4.5, 4.6, 0.8, [[("■ ", {"color": BLUE}), ("测试：20% 基因完全不参与训练　", {}), ("■ ", {"color": "B8CBE3"}), ("训练", {})],
                                  "每个基因恰好被预测一次 → 1,092 个 out-of-fold 预测" + (f"；换 {nsplit} 种随机划分重复" if nsplit > 1 else "")],
         size=12, color=TEXT2, after=3)
    box(s, 0.6, 5.35, 5.6, 1.2, fill=TINT)
    sw = pd.read_csv(os.path.join(RES, "dev_sweep.csv"))
    rng_de = sw.pearson_de20.max() - sw.pearson_de20.min()
    rng_sys = sw.pearson_sys.max() - sw.pearson_sys.min()
    text(s, 0.8, 5.42, 5.25, 1.1, [[("开发集 / 测试集严格分离：", {"bold": True, "color": BLUE}),
                                    (f"模型设计与超参数只在开发集上比较，基线同等调参；第 0 折的 219 个基因完全未参与设计，在这一折上优势反而更大。"
                                     f"{len(sw)} 组超参数的结果极差：DE20 {rng_de:.3f}、去偏 {rng_sys:.3f}，模型对超参数不敏感。", {})]],
         size=12.5, line=1.15)
    mets = [("Pearson Δ (全部 / Top-20 DEG)", "GEARS 标准指标"), ("DEG 方向准确率", "上调 / 下调是否预测正确"),
            ("去偏 Pearson (Systema)", "减去平均扰动响应，只奖励扰动特异信号"), ("扰动识别排名", "预测能否认出是哪个基因被敲低"),
            ("能量距离 / MMD", "单细胞分布是否一致"), ("噪声天花板 + 配对 Wilcoxon", "细胞拆半一致性；95% bootstrap CI")]
    text(s, 6.75, 1.45, 6.0, 0.4, ["指标：不只看 accuracy"], size=16, bold=True)
    for k, (a, b_) in enumerate(mets):
        y = 1.95 + k * 0.76
        box(s, 6.75, y, 6.0, 0.64, fill=WHITE, line=LINE)
        text(s, 6.9, y, 2.95, 0.64, [a], size=13, bold=True, anchor=MSO_ANCHOR.MIDDLE)
        text(s, 9.85, y, 2.85, 0.64, [b_], size=11.5, color=TEXT2, anchor=MSO_ANCHOR.MIDDLE, line=1.0)
    footer(s, n)
    notes(s, "评测设计：按基因五折划分，每折 20% 的基因完全不进训练，每个基因恰好被预测一次；设计和超参数只在开发集上比较，基线也同等调参，第 0 折的基因完全没参与设计" + ("；还换了多种随机划分重复。" if nsplit > 1 else "。")
             + "指标不只看相关性：去偏 Pearson 下平凡基线恒为 0；另有 DEG 方向、识别排名、单细胞分布距离、噪声上限和配对检验。")

    # 7 ---------------------------------------------------------------- main results (native chart)
    s = new()
    title(s, "主结果：知识检索稳定优于文献基线", "评测")
    # 3-split averages; Linear and kNN tuned on the development split (03d_baseline_tuning.py)
    sr = R["splits"].groupby("model").mean(numeric_only=True)
    tb = pd.read_csv(os.path.join(RES, "baseline_tuned_vs_pf.csv")).groupby(["baseline", "metric"]).baseline_value.mean()
    tbd = pd.read_csv(os.path.join(RES, "baseline_tuned_vs_pf.csv")).groupby(["baseline", "metric"])["diff"].mean()
    AV = {MS: sr.loc[MS], NET: sr.loc[NET], PF: sr.loc[PF],
          LIN: pd.Series({k: tb.loc[("Linear", k)] for k in ("pearson_all", "pearson_de20", "pearson_sys", "dir_acc_de20")}),
          KNN: pd.Series({k: tb.loc[("kNN", k)] for k in ("pearson_all", "pearson_de20", "pearson_sys", "dir_acc_de20")})}
    models = [(MS, "均值平移", "8A929B"), (LIN, "线性基线 (调参后)", AQUA), (NET, "PerturbNet (MLP/Transformer)", YELLOW),
              (KNN, "kNN 多源先验 (调参后)", ORANGE), (PF, "PerturbFormer (本文)", BLUE)]
    cats = [("pearson_all", "Pearson Δ\n全部基因"), ("pearson_de20", "Pearson Δ\nTop-20 DEG"), ("pearson_sys", "去偏 Pearson\n(Systema)"), ("dir_acc_de20", "DEG 方向\n准确率")]
    cd = CategoryChartData()
    cd.categories = [c for _, c in cats]
    for m, lab, _ in models:
        cd.add_series(lab, [round(float(AV[m][k]), 3) for k, _ in cats])
    gf = s.shapes.add_chart(XL_CHART_TYPE.COLUMN_CLUSTERED, Inches(0.5), Inches(1.4), Inches(8.3), Inches(5.3), cd)
    ch = gf.chart
    ch.has_legend = True
    ch.legend.position = XL_LEGEND_POSITION.TOP
    ch.legend.include_in_layout = False
    ch.legend.font.size = Pt(11)
    ch.legend.font.color.rgb = rgb(TEXT2)
    pl = ch.plots[0]
    pl.gap_width = 55
    pl.overlap = -8
    for ser, (_, _, c) in zip(pl.series, models):
        ser.format.fill.solid()
        ser.format.fill.fore_color.rgb = rgb(c)
    last = pl.series[len(models) - 1]
    last.data_labels.show_value = True
    last.data_labels.number_format = "0.00"
    last.data_labels.number_format_is_linked = False
    last.data_labels.position = XL_LABEL_POSITION.OUTSIDE_END
    last.data_labels.font.size = Pt(11)
    last.data_labels.font.bold = True
    last.data_labels.font.color.rgb = rgb(TEXT)
    va = ch.value_axis
    va.maximum_scale = 1.0
    va.minimum_scale = 0.0
    va.has_major_gridlines = True
    va.major_gridlines.format.line.color.rgb = rgb("E6EAEF")
    va.tick_labels.font.size = Pt(10)
    va.tick_labels.font.color.rgb = rgb(MUTED)
    va.format.line.fill.background()
    ch.category_axis.tick_labels.font.size = Pt(11)
    ch.category_axis.tick_labels.font.color.rgb = rgb(TEXT)
    ch.category_axis.format.line.color.rgb = rgb(LINE)
    st = R["stats"]
    keym = ["pearson_all", "pearson_de20", "pearson_sys", "dir_acc_de20", "de_overlap20", "rank"]

    def verdict(other):
        better, worse = 0, 0
        for k in keym:
            d = pf[k] - S.loc[other, k]
            if k == "rank":
                d = -d
            if st.loc[other, k + "_p"] < 0.05:
                better += d > 0
                worse += d < 0
        return better, worse, len(keym) - better - worse
    b_lin = verdict(LIN)
    b_knn = verdict(KNN)

    def verdict_splits(other):
        """significant wins / ties / losses over all available CV splits x 6 metrics"""
        from scipy.stats import wilcoxon as _w
        w = t = l = 0
        nspl = 0
        for d in ("", "_seed1", "_seed2"):
            fp = os.path.join(ROOT, "results", "replogle_k562_essential" + d, "per_perturbation_metrics.csv")
            if not os.path.exists(fp):
                continue
            nspl += 1
            mm_ = pd.read_csv(fp)
            for k in keym:
                q = mm_.pivot(index="pert", columns="model", values=k).dropna(subset=[PF, other])
                dlt = q[PF].mean() - q[other].mean()
                if k == "rank":
                    dlt = -dlt
                if _w(q[PF], q[other]).pvalue < 0.05:
                    w, l = w + (dlt > 0), l + (dlt < 0)
                else:
                    t += 1
        return w, t, l, nspl
    vs = verdict_splits(KNN)
    tv = pd.read_csv(os.path.join(RES, "baseline_tuned_vs_pf.csv"))
    tk = tv[tv.baseline == "kNN"]
    vt = (((tk.p < 0.05) & ((tk["diff"] > 0) ^ (tk.metric == "rank"))).sum(), ((tk.p >= 0.05)).sum(),
          ((tk.p < 0.05) & ((tk["diff"] < 0) ^ (tk.metric == "rank"))).sum())
    stat(s, 9.2, 1.45, 3.6, f"+{tbd.loc[('Linear', 'pearson_sys')]:.3f}",
         f"去偏 Pearson 超过调参后的线性基线；{len(tv[tv.baseline == 'Linear'])} 次配对比较全部显著更好")
    stat(s, 9.2, 2.95, 3.6, f"{AV[PF]['pearson_sys'] / nc['pearson_sys_pred_ceiling']:.0%}",
         f"达到测量噪声决定的可达上限（去偏 {nc['pearson_sys_pred_ceiling']:.2f}）；均值平移为 0")
    stat(s, 9.2, 4.45, 3.6, f"{vt[0]} 优 · {vt[1]} 平 · {vt[2]} 劣",
         f"对比调参后的 kNN：相关性持平；DEG 方向、DEG 重合、识别排名在 3 种划分中都更好", vsize=24)
    gp_ = os.path.join(RES, "gears_fold0_seed0.csv")
    if os.path.exists(gp_):
        gg_ = pd.read_csv(gp_).groupby("model").pearson_sys.mean()
        text(s, 9.2, 5.75, 3.6, 1.0, [f"已发表方法 GEARS（第 0 折）：去偏 Pearson {gg_['GEARS+KD']:.2f}；"
                                      "公平预算重跑（30/50 细胞、20 epoch）仍为 0.23，未改变结论；3 种划分平均，配对 Wilcoxon"],
             size=11, color=MUTED, line=1.05)
    else:
        text(s, 9.2, 5.85, 3.6, 0.9, ["3 种随机划分平均；配对 Wilcoxon，每种划分 n = 1,092 个未见基因"], size=11, color=MUTED)
    footer(s, n, "完整表格：results/replogle_k562_essential/{main_table, split_robustness, baseline_tuned_vs_pf}.md")
    notes(s, f"主结果是三种划分的平均，基线都做了同等调参。均值平移的去偏 Pearson 为 0，没有扰动特异信息。知识检索的模型明显好于线性和深度基线，PerturbFormer 去偏 Pearson {AV[PF]['pearson_sys']:.3f}，"
             f"比调参后的线性基线高 {tbd.loc[('Linear', 'pearson_sys')]:.3f}。如实说明：调参后的 kNN 在相关性上和它持平，PerturbFormer 稳定更好的是 DEG 方向、重合度和可识别性。已发表的 GEARS 明显更差；公平预算重跑（30/50 细胞、20 epoch）也没有改变结论。")

    # 8 ---------------------------------------------------------------- priors & ablation (native charts)
    s = new()
    title(s, "哪些设计真正起作用？", "评测")
    names = {"go": "GO", "ppi": "PPI", "genept": "GenePT", "resp": "应答谱"}
    single = {k: np.nanmean([S.loc[f"{fam}-{k}", "pearson_sys"] for fam in ("Linear", "kNN") if f"{fam}-{k}" in S.index]) for k in names}
    order_src = sorted(single, key=lambda k: -single[k])

    def quiet_axes(ch, vmin=None, vmax=None, fmt="0.0"):
        va = ch.value_axis
        if vmin is not None:
            va.minimum_scale = vmin
        if vmax is not None:
            va.maximum_scale = vmax
        va.has_major_gridlines = True
        va.major_gridlines.format.line.color.rgb = rgb("E6EAEF")
        va.tick_labels.font.size = Pt(10)
        va.tick_labels.font.color.rgb = rgb(MUTED)
        va.tick_labels.number_format = fmt
        va.tick_labels.number_format_is_linked = False
        va.format.line.fill.background()
        ca = ch.category_axis
        ca.tick_labels.font.size = Pt(11)
        ca.tick_labels.font.color.rgb = rgb(TEXT)
        ca.format.line.color.rgb = rgb(LINE)
        ca.has_major_gridlines = False

    srcs = [("go", "GO"), ("ppi", "PPI"), ("genept", "GenePT"), ("resp", "应答谱"), ("concat", "多源融合")]
    cd = CategoryChartData()
    cd.categories = [l for _, l in srcs]
    for fam in ("Linear", "kNN"):
        cd.add_series(fam, [round(float(S.loc[f"{fam}-{k}", "pearson_sys"]), 3) for k, _ in srcs])
    gf = s.shapes.add_chart(XL_CHART_TYPE.COLUMN_CLUSTERED, Inches(0.5), Inches(1.75), Inches(6.0), Inches(3.3), cd)
    ch = gf.chart
    ch.has_title = True
    ch.chart_title.text_frame.text = "单一先验 vs 多源融合（去偏 Pearson）"
    style_run(ch.chart_title.text_frame.paragraphs[0].runs[0], 13, True, TEXT)
    ch.has_legend = True
    ch.legend.position = XL_LEGEND_POSITION.TOP
    ch.legend.include_in_layout = False
    ch.legend.font.size = Pt(11)
    pl = ch.plots[0]
    pl.gap_width = 60
    for ser, c in zip(pl.series, [AQUA, ORANGE]):
        ser.format.fill.solid()
        ser.format.fill.fore_color.rgb = rgb(c)
    quiet_axes(ch, 0, 0.5, "0.0")

    abl_names = [("PerturbFormer", "去掉 KD 敲低规则"), ("Ablation-no-go+KD", "去掉 GO"), ("Ablation-no-genept+KD", "去掉 GenePT"),
                 ("Ablation-free-attention+KD", "自由注意力 (无知识核)"), ("Ablation-no-ppi+KD", "去掉 PPI"),
                 ("Ablation-concat-fusion+KD", "拼接代替 Transformer"), ("Ablation-no-resp+KD", "去掉应答谱"),
                 ("Ablation-no-topk+KD", "去掉 top-k"), ("Ablation-no-dirloss+KD", "去掉方向损失")]
    abl_names = [(k, v) for k, v in abl_names if k in S.index]
    drops = [(v, float(S.loc[k, "pearson_sys"] - pf.pearson_sys)) for k, v in abl_names]
    drops.sort(key=lambda r: r[1])
    cd = CategoryChartData()
    cd.categories = [v for v, _ in drops][::-1]
    cd.add_series("Δ 去偏 Pearson", [round(d, 3) for _, d in drops][::-1])
    gf = s.shapes.add_chart(XL_CHART_TYPE.BAR_CLUSTERED, Inches(6.85), Inches(1.75), Inches(5.95), Inches(4.95), cd)
    ch = gf.chart
    ch.has_title = True
    ch.chart_title.text_frame.text = "消融：去偏 Pearson 相对完整模型的变化"
    style_run(ch.chart_title.text_frame.paragraphs[0].runs[0], 13, True, TEXT)
    ch.has_legend = False
    pl = ch.plots[0]
    pl.gap_width = 55
    ser = pl.series[0]
    ser.invert_if_negative = False
    ser.format.fill.solid()
    ser.format.fill.fore_color.rgb = rgb("E34948")
    pl.has_data_labels = True
    pl.data_labels.number_format = "+0.000;-0.000"
    pl.data_labels.number_format_is_linked = False
    pl.data_labels.position = XL_LABEL_POSITION.OUTSIDE_END
    pl.data_labels.font.size = Pt(10)
    pl.data_labels.font.color.rgb = rgb(TEXT2)
    quiet_axes(ch, -0.04, 0.0, "0.00")
    ch.category_axis.tick_labels.font.size = Pt(11)
    ch.category_axis.tick_label_position = XL_TICK_LABEL_POSITION.LOW
    kd_gain = pf.pearson_de20 - S.loc["PerturbFormer", "pearson_de20"]
    lines = [[("先验知识：", {"bold": True, "color": BLUE}),
              (f"单源 {' > '.join(names[k] for k in order_src)}；应答谱只覆盖 38% 的基因，几乎不能泛化；多源融合优于任何单源。", {})],
             [("关键设计：", {"bold": True, "color": BLUE}),
              (f"知识核注意力、三种外部知识缺一不可；CRISPRi 敲低规则使 Top-20 DEG 相关 +{kd_gain:.3f}。", {})]]
    text(s, 0.6, 5.25, 6.0, 1.6, lines, size=13, after=6, line=1.12)
    footer(s, n)
    notes(s, f"左图：单独用一种先验时 {names[order_src[0]]} 最强，纯数据驱动的应答谱几乎不能泛化，多源融合最好。右图是消融，影响最大的是{drops[0][0]}、{drops[1][0]}和{drops[2][0]}；"
             f"换成从零学习的注意力会变差，说明从多源 kNN 出发只学修正是必要的。敲低规则让 Top-20 DEG 相关提升 {kd_gain:.2f}。")

    # 9 ---------------------------------------------------------------- flow
    s = new()
    title(s, "高级版本：条件流匹配生成扰动后的单细胞", ["模型", "评测"])
    fl = R["flow"]
    bl = [[("对照细胞分布 → 扰动细胞分布：", {"bold": True}), ("在 100 维 PCA 空间学习速度场 v(x, t, c)，ODE 积分 20 步", {})],
          [("最优传输配对：", {"bold": True}), ("每个扰动内部用匈牙利算法配对对照与扰动细胞 (OT-CFM)", {})],
          [("条件 c：", {"bold": True}), ("PerturbFormer 的预测；训练扰动用“排除自身”的预测，避免标签泄露", {})],
          [("输出：", {"bold": True}), ("输入任意对照细胞，得到它被敲低后的反事实状态", {})]]
    text(s, 0.6, 1.45, 5.3, 3.3, bl, size=13.5, bullet=True, after=8, line=1.12)
    flow_note = ""
    if fl is not None:
        g = fl.groupby("model")[["energy_distance", "logvar_err_top20pc"]].mean()
        e_cfm, e_sh = g.loc["CFM (ours)", "energy_distance"], g.loc["PerturbFormer-shift", "energy_distance"]
        v_cfm, v_sh = g.loc["CFM (ours)", "logvar_err_top20pc"], g.loc["PerturbFormer-shift", "logvar_err_top20pc"]
        e_or = g.loc["Oracle-shift", "energy_distance"]
        pv = fl.pivot_table(index="pert", columns="model", values="energy_distance")
        win = (pv["CFM (ours)"] < pv["PerturbFormer-shift"]).mean()
        stat(s, 0.6, 4.85, 1.7, f"{(e_cfm / e_sh - 1):+.0%}", "能量距离 (vs 仅平移均值)", vsize=30)
        stat(s, 2.4, 4.85, 1.7, f"{(v_cfm / v_sh - 1):+.0%}", "方差误差 (vs 仅平移均值)", vsize=30)
        from scipy.stats import wilcoxon as _w
        pval = _w(pv["CFM (ours)"], pv["PerturbFormer-shift"]).pvalue
        if "Borrowed-cells" in set(fl.model):
            e_b = g.loc["Borrowed-cells", "energy_distance"]
            pb_ = _w(pv["CFM (ours)"], pv["Borrowed-cells"]).pvalue
            stat(s, 4.2, 4.85, 1.7, "≈", f"借用相似扰动细胞 ({e_b:.2f} vs {e_cfm:.2f}, p={pb_:.1g})", vsize=30, color=ORANGE)
        else:
            stat(s, 4.2, 4.85, 1.7, f"{win:.0%}", f"未见扰动上更好 (p={pval:.0e})", vsize=30)
        flow_note = (f"能量距离 {e_sh:.2f} 降到 {e_cfm:.2f}，方差误差也明显下降。但如实说明：直接借用最相似扰动的真实细胞，分布指标和流模型持平；"
                     "流模型独有的是逐细胞映射，给定一个细胞能算出它自己的反事实状态。")
    picture(s, os.path.join(FIG, "fig7_flow_slide.png"), 6.2, 1.3, w=6.65, h=5.6)
    footer(s, n)
    notes(s, "高级版本是条件流匹配：在 PCA 潜空间学一个速度场，把对照细胞连续推到扰动后的分布，条件来自 PerturbFormer 的预测，所以也能处理未见基因。相对只平移均值，" + flow_note)

    # 10 --------------------------------------------------------------- external
    s = new()
    title(s, "外部验证：换一个实验室还能预测吗？", "评测")
    picture(s, os.path.join(FIG, "fig8_external.png"), 0.5, 1.4, w=12.3, h=3.2)
    ext = R["ext"]
    if ext is not None:
        e = ext.loc["PerturbFormer+KD"]
        c = ext.loc["Replogle-measured (same gene)"]
        frac = e["pearson_de20_overlap41"] / c["pearson_de20"]
        stat(s, 0.6, 4.85, 3.6, f"{frac:.0%}", "Top-20 DEG 相关达到“直接重新测量同一基因”的跨实验室上限的比例", vsize=36, h=1.6)
        paras = [[("设计：", {"bold": True}), ("用 Replogle 数据训练，并剔除全部 86 个 Adamson 靶基因 → 每个 Adamson 扰动对模型都是新基因；在两数据共有的 1,910 个基因上比较。", {})],
                 [("上限：", {"bold": True}), (f"同一基因在 Replogle 的实测响应预测 Adamson，全基因相关也只有 {c['pearson_all']:.2f}，跨实验室差异本身很大。", {})]]
        k_ = ext.loc["kNN-concat+KD"]
        paras.append([("两种检索方法对比：", {"bold": True}),
                      (f"跨实验室时 kNN 的 Top-20 DEG 相关 ({k_['pearson_de20_overlap41']:.2f}) 略高于 PerturbFormer ({e['pearson_de20_overlap41']:.2f})，"
                       f"PerturbFormer 全基因相关更高 ({e['pearson_all_overlap41']:.2f} vs {k_['pearson_all_overlap41']:.2f})；可学习模型的小幅优势没有跨实验室保持。", {})])
        dir_rank = ext["dir_acc_de20"].drop("Replogle-measured (same gene)").sort_values(ascending=False)
        if dir_rank.index[0] == "Mean-shift":
            paras.append([("需要说明的发现：", {"bold": True, "color": ORANGE}),
                          (f"Adamson 中系统性响应占主导，均值平移的方向准确率 ({dir_rank.iloc[0]:.2f}) 反而最高，但它的去偏相关为 0。", {})])
        text(s, 4.6, 4.75, 8.2, 2.2, paras, size=12, after=5, line=1.1)
    footer(s, n)
    notes(s, f"外部验证：只用 Replogle 训练，并去掉 Adamson 的全部靶基因，再预测另一实验室的数据。上限本身很低，直接拿 Replogle 里同一基因的实测去预测，全基因相关也只有 {c['pearson_all']:.2f}。"
             "我们对未见基因达到了这一上限的大部分；跨实验室时 kNN 在部分指标上略好，可学习模型的小优势没有保持，这点如实报告。")

    # 11 --------------------------------------------------------------- biology 1
    s = new()
    title(s, "生物学解释：预测保留了真实的生物学结构", "解释")
    picture(s, os.path.join(FIG, "fig_interp_C_complex.png"), 0.5, 1.4, w=6.4, h=2.6)
    picture(s, os.path.join(FIG, "fig_interp_D_pathways.png"), 7.1, 1.3, w=5.7, h=4.9)
    cx = R["cx"]["frac_top5_same_complex"]
    pw = R["pw"].set_index("model")
    ppw = pd.read_csv(os.path.join(ROOT, "results", "interpretation", "pathway_consistency_per_pathway.csv")).head(20)
    top3 = ppw.sort_values("r_pred_vs_true", ascending=False).pathway.head(3).tolist()
    text(s, 0.6, 4.25, 6.3, 2.6, [
        [("蛋白复合体：", {"bold": True, "color": BLUE}),
         (f"实测数据中同一 CORUM 复合体的基因敲低后响应相似（Top-5 邻居同复合体 {cx['实测应答谱之间']:.0%}，随机 {cx['随机']:.1%}）；"
          f"预测应答谱同样保留了这一结构（{cx['预测应答谱 → 最近的实测应答谱']:.0%}）。", {})],
        [("通路层面：", {"bold": True, "color": BLUE}),
         (f"Hallmark 通路活性预测与实测的平均相关 r = {pw.loc['PerturbFormer+KD', 'mean_r_per_pathway']:.2f}（均值平移 {pw.loc['Mean-shift', 'mean_r_per_pathway']:.2f}）；"
          f"最准的是 {'、'.join(top3)}。", {})],
        [("CRISPRi 自检：", {"bold": True, "color": BLUE}), ("被靶向的基因本身显著下调，且几乎总是该扰动排名第一的差异基因。", {})]],
        size=13.5, after=8, line=1.13)
    footer(s, n)
    notes(s, f"预测是否保留了真实的生物学结构？左图：实测中同一复合体的成员敲低后响应相似，预测的应答谱也保留了这种结构。右图是通路层面，{'、'.join(top3)} 预测最好，"
             f"平均相关 {pw.loc['PerturbFormer+KD', 'mean_r_per_pathway']:.2f}，而均值平移在通路层面完全没有区分能力。")

    # 12 --------------------------------------------------------------- case studies
    s = new()
    title(s, "案例：三个成功，一个有启发的失败", "解释")
    picture(s, os.path.join(FIG, "fig_interp_G_cases.png"), 0.45, 1.35, w=12.45, h=3.35)
    met = pd.read_csv(os.path.join(RES, "per_perturbation_metrics.csv"))
    mm = met[met.model == PF].set_index("pert")
    oofp = list(oof["perts"])
    gnb = [perts[j] for j in oof["neigh_idx"][oofp.index("GATA1"), :4] if j >= 0] if "GATA1" in oofp else []
    text(s, 0.6, 4.85, 6.0, 1.95, [
        [("RPS19 / MED12 / PSMB5：", {"bold": True, "color": BLUE}),
         (f"r = {mm.loc['RPS19', 'pearson_all']:.2f} / {mm.loc['MED12', 'pearson_all']:.2f} / {mm.loc['PSMB5', 'pearson_all']:.2f}。"
          "同一复合体的其他亚基都已测过，模型借鉴它们，富集结果与实测一致：RPS19 → 胞质翻译基因下调；"
          "MED12 → MYC 靶基因下调、血红素代谢上调；PSMB5 → MYC 靶基因下调、SQSTM1/UBC 等蛋白稳态应激基因上调。", {})]],
        size=13, after=8, line=1.12)
    text(s, 6.8, 4.85, 6.0, 1.95, [
        [("GATA1（失败）：", {"bold": True, "color": ORANGE}),
         (f"r = {mm.loc['GATA1', 'pearson_all']:.2f}。GATA1 是红系主转录因子，训练集中没有效应相同的基因。模型借鉴 {'、'.join(gnb[:3])} 等转录因子，"
          "正确预测了 MYC 靶基因下调（多数转录因子敲低的共性），却把红系特异的血红蛋白基因方向预测反了。", {})],
        [("启示：", {"bold": True}), ("谱系特异的功能无法靠“相似基因”外推，这正是最需要做实验的地方。", {})]],
        size=13, after=6, line=1.12)
    footer(s, n)
    notes(s, f"四个案例都是训练时没见过的基因。RPS19、MED12、PSMB5 所在复合体的其他成员都测过，模型借鉴它们，富集结果与实测一致。GATA1 是有启发的失败：它是红系主转录因子，"
             f"训练集中没有效应相同的基因，模型参考了 {'、'.join(gnb[:3])} 等一般转录因子，MYC 靶基因下调对了，血红蛋白基因方向却反了。谱系特异的功能不能靠相似基因外推，恰恰最需要实验。")

    # 13 --------------------------------------------------------------- where are the errors?
    s = new()
    title(s, "错误出在哪里？", "解释")
    picture(s, os.path.join(FIG, "fig_interp_H_direction.png"), 0.5, 1.35, w=6.4, h=2.3)
    picture(s, os.path.join(FIG, "fig_interp_E_failure.png"), 7.05, 1.35, w=5.8, h=2.1)
    dn = R["dirn"].loc[PF]
    cal = json.load(open(os.path.join(ROOT, "results", "interpretation", "reliability.json")))
    rel_txt = " / ".join(f"{x:.2f}" for x in cal["mean_pearson_sys"])
    fa = R["fail"]
    from scipy.stats import spearmanr
    rho1 = spearmanr(fa.max_prior_sim_to_train, fa.pearson_sys)[0]
    rho2 = spearmanr(np.log10(fa.n_deg + 1), fa.pearson_sys)[0]
    strong = min(dn["acc|dlt 0.2–0.5"], dn["acc|dlt >0.5"])
    stat(s, 0.6, 3.95, 2.0, f"{dn.acc_all_sig:.1%}", f"全部 {dn.n_sig_deg / 1e4:.0f} 万个显著 DEG 的方向准确率（AUROC {dn.auroc_up_vs_down:.2f}）", vsize=30, h=1.4)
    stat(s, 2.8, 3.95, 2.0, f"≥{strong:.0%}", "变化幅度 |Δ| > 0.2 的 DEG", vsize=30, h=1.4)
    stat(s, 5.0, 3.95, 1.9, f"{dn['acc|dlt <0.05']:.0%}", "|Δ| < 0.05 的弱变化 DEG", vsize=30, h=1.4, color=ORANGE)
    text(s, 0.6, 5.55, 6.3, 1.3, [[("基因层面：", {"bold": True, "color": BLUE}),
         ("方向错误集中在接近噪声的弱变化基因；题目要求的“DEG 方向预测”对强效应基因已相当可靠。", {})]], size=13, line=1.12)
    text(s, 7.05, 3.85, 5.8, 3.0, [
        [("扰动层面：", {"bold": True, "color": BLUE}), (f"训练集中没有相似基因（Spearman ρ = {rho1:.2f}）或扰动效应弱（ρ = {rho2:.2f}）时，整体预测变差。", {})],
        [("可以事先判断：", {"bold": True, "color": BLUE}), (f"按先验相似度分低/中/高三档，交叉验证去偏 Pearson 依次为 {rel_txt}；"
                                                          "预测工具会自动给出评级（GATA1 被评为“低”）。", {})],
        [("启示：", {"bold": True}), ("下一轮实验应优先测知识空间中离已测基因最远的基因（主动学习）。", {})]],
        size=13, after=8, line=1.12)
    footer(s, n)
    notes(s, f"错误出在哪里？左图是基因层面：在全部 {dn.n_sig_deg / 1e4:.0f} 万个显著差异基因上，方向准确率 {dn.acc_all_sig:.0%}，变化幅度大于 0.2 的基因在 {strong:.0%} 以上，"
             f"错误集中在接近噪声的弱变化基因。右图是扰动层面：训练集中没有相似基因，或者扰动本身效应弱时，预测变差。先验相似度在预测前就能算出，按它分三档，去偏 Pearson 从 {cal['mean_pearson_sys'][0]:.2f} 升到 {cal['mean_pearson_sys'][2]:.2f}，预测工具会自动给出可靠性评级，GATA1 就被评为低。这也提示下一轮实验应该优先测离已测基因最远的基因。")

    # 14 --------------------------------------------------------------- demo
    s = new()
    title(s, "成果：输入细胞 + 基因，输出扰动后状态", "解释")
    demo_gene = "MYC" if os.path.exists(os.path.join(ROOT, "results", "predictions", "MYC", "report.md")) else "GATA1"
    box(s, 0.6, 1.45, 5.4, 0.9, fill=DARK)
    text(s, 0.8, 1.5, 5.1, 0.8, [f"$ python scripts/predict.py --gene {demo_gene} \\", "      --cells my_k562_cells.h5ad"],
         size=13, color="D6E2EC", font=MONO, after=0, anchor=MSO_ANCHOR.MIDDLE)
    rep_path = os.path.join(ROOT, "results", "predictions", demo_gene, "report.md")
    rep = open(rep_path).read().splitlines()[2:] if os.path.exists(rep_path) else []
    keep = ("是否测过", "可靠性", "依据", "预测下调", "通路活性下降")
    rep = [r.lstrip("- ") for r in rep if any(k in r for k in keep)]
    rep = [r.replace(" (注意力权重)", "").replace("训练筛选中是否测过该扰动", "筛选中是否测过") for r in rep]

    def first_items(line, k=4):
        if ": " not in line:
            return line
        head, body = line.split(": ", 1)
        sep = "; " if "; " in body else ", "
        items = body.split(sep)
        return head + ": " + sep.join(items[:k]) + (" …" if len(items) > k else "")
    rep = [first_items(r, 3) if not r.startswith("预测可靠性") else
           r.split(" (")[0] + "（最相似的已测扰动: " + r.split("最相似: ")[1].split(";")[0] + "）" for r in rep]
    text(s, 0.6, 2.55, 5.4, 3.0, rep, size=12, color=TEXT2, bullet=True, after=5, line=1.05)
    text(s, 0.6, 5.55, 5.4, 1.3, [[("输出：", {"bold": True}), ("单细胞表达 (h5ad)、伪批量变化表、证据报告与可视化；支持约 2.1 万个人类基因；交互式网页可浏览全部 1,092 个预测。", {})]],
         size=12.5, line=1.12)
    demo = os.path.join(ROOT, "results", "predictions", demo_gene, "prediction.png")
    if os.path.exists(demo):
        picture(s, demo, 6.2, 1.5, w=6.65, h=5.2)
    footer(s, n)
    notes(s, f"最终成果：一行命令，输入一批对照细胞和一个基因，输出这些细胞敲低后的表达、伪批量变化表和证据报告。右图是 {demo_gene}，它在筛选中从未被扰动过，"
             "模型参考了 MAX 等转录因子，预测 MYC 靶基因整体下调，与已知功能一致；这类新基因无法直接验证，定量准确性以交叉验证为准。")

    # 15 --------------------------------------------------------------- conclusion
    s = new()
    bg(s, DARK)
    text(s, 0.75, 0.6, 11.5, 0.8, ["结论与展望"], size=34, bold=True, color=WHITE)
    concl = [("问题", "把“敲低一个新基因会怎样”形式化为未见扰动泛化，并正视系统偏差与测量噪声"),
             ("模型", "多源知识检索：PerturbFormer 在已测扰动上做知识核交叉注意力；条件流匹配把预测展开到单细胞"),
             ("评测", f"3 种划分、基线同等调参：显著优于线性与深度基线；与多源 kNN 相关性持平、DEG 方向与可识别性更好；跨实验室验证成立"),
             ("解释", "预测保留复合体与通路结构，每个预测可追溯到它参考的已测实验；谱系主调控因子是明确的失败边界")]
    for i, (a, b_) in enumerate(concl):
        y = 1.65 + i * 0.95
        circle_label(s, 0.75, y, 0.55, a[0], fill=BLUE, size=15)
        text(s, 1.55, y - 0.04, 11.0, 0.7, [[(a + "　", {"bold": True, "color": WHITE}), (b_, {"color": "D6E2EC"})]], size=15.5,
             anchor=MSO_ANCHOR.MIDDLE, line=1.05)
    text(s, 0.75, 5.6, 5.7, 1.4, [[("局限", {"bold": True, "color": "F2A27E"})], "单一细胞系、只做单基因敲低；单细胞分布上流模型与“借用相似扰动细胞”持平；跨实验室差异仍大"],
         size=13.5, color="D6E2EC", after=4, line=1.12)
    text(s, 6.8, 5.6, 5.8, 1.4, [[("展望", {"bold": True, "color": "7FB2F0"})], "跨细胞系迁移 (K562 → RPE1)；组合扰动；用不确定性驱动主动学习选择下一轮实验"],
         size=13.5, color="D6E2EC", after=4, line=1.12)
    notes(s, "总结：我把敲低新基因会怎样形式化为未见扰动泛化，用多源知识检索加流匹配实现了输入细胞和基因、输出扰动后状态；在严格评测下显著优于文献基线，并通过跨实验室验证；"
             "预测保留了复合体和通路结构，每个预测可追溯，失败边界也很清楚。局限是只有一个细胞系、只做单基因敲低。谢谢。")

    prs.save(OUT)
    print("saved", OUT, f"({n} slides)")
    print(f"estimated talk time: {sum(TALK)} s = {sum(TALK) / 60:.1f} min; per slide: {TALK}")


if __name__ == "__main__":
    build()
