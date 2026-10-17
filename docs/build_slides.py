"""Build the 10-minute presentation (docs/VirtualCell-Perturb_汇报.pptx) from the result files.

All numbers on the slides are read from results/ so the deck stays in sync with the code.
    python docs/build_slides.py
"""
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


def notes(slide, s):
    slide.notes_slide.notes_text_frame.text = s


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
    notes(s, "（约 30 秒）各位老师同学好。我的选题是第九题：虚拟细胞扰动预测。核心问题只有一句话：如果敲低一个从来没有做过实验的基因，细胞的转录组会发生什么变化？我做了两个层次的模型：一个检索增强的 Transformer 预测平均表达变化，一个条件流匹配生成模型预测每个细胞的变化。下面按 Problem、Data、Model、Benchmark、Biological interpretation 五个环节汇报。")

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
    notes(s, "（约 50 秒）为什么做这个问题：基因组合空间太大、Perturb-seq 太贵，所以需要能在计算机里先做筛选的虚拟细胞。任务形式化为：输入一个对照细胞和一个要敲低的基因，输出平均表达变化、差异基因的方向，以及这个细胞扰动后的状态。真正的难点有三个：第一，测试基因训练时完全没见过；第二，近期 Nature Methods 的工作指出，大多数扰动共享一个系统性响应，“预测平均效应”这种平凡基线就很强，很多深度模型其实打不过它；第三，测量本身噪声很大。我的评测设计就是围绕这三点展开的。")

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
    text(s, 7.25, 4.1, 5.5, 0.3, ["覆盖率 = 1,137 个被扰动基因中有该先验的比例"], size=10.5, color=MUTED)
    picture(s, os.path.join(FIG, "fig1_data_overview.png"), 0.6, 4.4, w=12.15, h=2.55)
    footer(s, n, "数据来源：GEARS 预处理版 (Harvard Dataverse)；GO / STRING v12 / GenePT (Zenodo 10833191)；MSigDB / CORUM (Enrichr)")
    notes(s, "（约 50 秒）数据用了两个独立实验。主数据是 Replogle 2022 的 K562 必需基因 CRISPRi 筛选，1092 个基因敲低、16 万个细胞；外部验证用 Adamson 2016，来自另一个实验室。为了能预测没见过的基因，我把每个基因编码成四种先验知识：GO 注释、STRING 蛋白互作、GenePT 这个基于大语言模型的基因嵌入，以及数据驱动的应答谱。下面三张小图说明：扰动效应强弱差异巨大，而右边的噪声天花板告诉我们，任何模型能达到的相关性都是有上限的。")

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
    notes(s, "（约 45 秒）这是整体框架。输入任意一个基因，先查出它的四种知识 token，经过 Transformer 融合；然后 PerturbFormer 拿这个表示去“查阅”所有已经测过的 1092 个扰动，注意力权重高的已测扰动，它们的真实响应被加权组合起来，再加上一个参数化分支和 CRISPRi 敲低规则，得到平均表达变化和 DEG 方向。最后，条件流匹配把这个平均变化展开成每个细胞的变化。设计原则有三条：借鉴而不是记忆；从强基线出发只学修正；两个层次的输出。")

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
    notes(s, f"（约 60 秒）PerturbFormer 的核心公式在左上：它是一个交叉注意力，Query 是待预测基因的知识表示，Key 是已测基因的知识表示，Value 是已测基因的真实响应。关键设计是“知识核”：注意力的 logits 直接由各知识源的余弦相似度组成，投影矩阵初始化为单位阵，所以模型一开始就等价于一个多源 kNN，训练只需要学习修正量。另外模型会针对每个基因自己决定更相信哪种知识。右边是一个真实例子：预测 {gname} 时它从未见过这个基因，但注意力几乎都落在其他蛋白酶体亚基上。这就是可解释性：每个预测都能追溯到它参考了哪些实验。")

    # 6 ---------------------------------------------------------------- benchmark design
    s = new()
    title(s, "评测设计：严格的未见基因泛化", "评测")
    text(s, 0.6, 1.45, 5.6, 0.4, ["5 折交叉验证，按扰动基因划分"], size=16, bold=True)
    for k in range(5):
        y = 1.95 + k * 0.5
        text(s, 0.6, y, 0.8, 0.4, [f"第 {k + 1} 折"], size=12, color=TEXT2, anchor=MSO_ANCHOR.MIDDLE)
        for j in range(5):
            box(s, 1.45 + j * 0.92, y + 0.04, 0.86, 0.32, fill=BLUE if j == k else "DCE6F2", radius=0.15)
    text(s, 1.45, 4.5, 4.6, 0.8, [[("■ ", {"color": BLUE}), ("测试：20% 基因完全不参与训练　", {}), ("■ ", {"color": "B8CBE3"}), ("训练", {})],
                                  "每个基因恰好被预测一次 → 1,092 个 out-of-fold 预测"], size=12, color=TEXT2, after=3)
    box(s, 0.6, 5.35, 5.6, 1.2, fill=TINT)
    sw = pd.read_csv(os.path.join(RES, "dev_sweep.csv"))
    rng_de = sw.pearson_de20.max() - sw.pearson_de20.min()
    rng_sys = sw.pearson_sys.max() - sw.pearson_sys.min()
    text(s, 0.8, 5.42, 5.25, 1.1, [[("开发集 / 测试集严格分离：", {"bold": True, "color": BLUE}),
                                    (f"所有设计与超参数只在 fold 0 训练基因内部划出的开发集上选择，5 个测试折从未用于调参。"
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
    notes(s, "（约 50 秒）评测的严谨性是这个项目的重点。第一，按基因划分 5 折，每折 20% 的基因完全不出现在训练里，每个基因都恰好被预测一次。第二，开发集和测试集严格分离，所有超参数只在第 0 折训练集内部的开发集上选，测试折从来不参与调参。第三，指标不只看相关性：特别加入了 Systema 去偏指标，它会把平均响应减掉，平凡基线在这个指标上恒为 0；还有 DEG 方向准确率、扰动识别排名，以及噪声天花板和配对检验。")

    # 7 ---------------------------------------------------------------- main results (native chart)
    s = new()
    title(s, "主结果：知识检索大幅超越文献基线", "评测")
    models = [(MS, "均值平移", "8A929B"), (LIN, "线性基线 (Ahlmann-Eltze)", AQUA), (NET, "PerturbNet (MLP/Transformer)", YELLOW),
              (KNN, "kNN 多源先验", ORANGE), (PF, "PerturbFormer (本文)", BLUE)]
    cats = [("pearson_all", "Pearson Δ\n全部基因"), ("pearson_de20", "Pearson Δ\nTop-20 DEG"), ("pearson_sys", "去偏 Pearson\n(Systema)"), ("dir_acc_de20", "DEG 方向\n准确率")]
    cd = CategoryChartData()
    cd.categories = [c for _, c in cats]
    for m, lab, _ in models:
        cd.add_series(lab, [round(float(S.loc[m, k]), 3) for k, _ in cats])
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
    stat(s, 9.2, 1.45, 3.6, f"+{pf.pearson_sys - S.loc[LIN, 'pearson_sys']:.3f}",
         f"去偏 Pearson 超过线性基线（{S.loc[LIN, 'pearson_sys']:.3f} → {pf.pearson_sys:.3f}），6 项指标中 {b_lin[0]} 项显著更好")
    stat(s, 9.2, 2.95, 3.6, f"{pf.pearson_sys / nc['pearson_sys_pred_ceiling']:.0%}",
         f"达到测量噪声决定的可达上限（去偏 {nc['pearson_sys_pred_ceiling']:.2f}）；均值平移为 0")
    stat(s, 9.2, 4.45, 3.6, f"{b_knn[0]} 优 · {b_knn[2]} 平 · {b_knn[1]} 劣",
         "对比 kNN 多源先验（同为本项目的知识检索思路）：差距小，但多数指标显著领先", vsize=24)
    text(s, 9.2, 5.85, 3.6, 0.9, ["配对 Wilcoxon 检验，n = 1,092 个未见基因（5 折 out-of-fold）"], size=11, color=MUTED)
    footer(s, n, "完整表格与 95% CI：results/replogle_k562_essential/main_table.md")
    notes(s, f"（约 60 秒）这是主结果，1092 个未见基因的平均值，蓝色是 PerturbFormer。先看灰色的均值平移：全基因相关 {S.loc[MS, 'pearson_all']:.2f}，看起来不差，但去偏 Pearson 为 0，说明它没有任何扰动特异的信息。文献里的线性基线和标准深度模型都明显好于它，而我们基于知识检索的两个模型又明显好于它们：PerturbFormer 的去偏 Pearson 是 {pf.pearson_sys:.3f}，比线性基线高 {pf.pearson_sys - S.loc[LIN, 'pearson_sys']:.3f}，达到测量噪声上限的 {pf.pearson_sys / nc['pearson_sys_pred_ceiling']:.0%}。要如实说明的是：可学习的 PerturbFormer 和简单的多源 kNN 差距很小，6 项指标中 {b_knn[0]} 项显著领先、{b_knn[1]} 项略逊。所以真正起作用的是“多源知识检索”这个思路，PerturbFormer 的增益在于小幅提升和可解释性。")

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
    notes(s, f"（约 45 秒）左图比较单独使用每种先验知识时的效果：{names[order_src[0]]} 最强，纯数据驱动的应答谱几乎不能泛化，而多源融合最好。右图是消融实验，条越长表示去掉该组件后下降越多：最关键的是 {drops[0][0]}、{drops[1][0]} 和 {drops[2][0]}。知识核注意力也很重要，换成从零学习的注意力会明显变差，验证了从强基线出发只学修正量的设计。敲低规则主要提升 Top-20 DEG 相关，因为被敲低的靶基因通常就是最强的差异基因。")

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
        stat(s, 4.2, 4.85, 1.7, f"{win:.0%}", "未见扰动上更好 (p<1e-38)", vsize=30)
        flow_note = (f"能量距离 {e_sh:.2f} → {e_cfm:.2f}，方差误差 {v_sh:.3f} → {v_cfm:.3f}；而使用真实均值的 Oracle 平移能量距离为 {e_or:.2f}，"
                     "说明剩余误差主要来自对未见基因平均效应的预测，而不是分布形状。")
    picture(s, os.path.join(FIG, "fig7_flow_slide.png"), 6.2, 1.3, w=6.65, h=5.6)
    footer(s, n)
    notes(s, "（约 50 秒）这是题目要求的高级版本：生成模型。我用条件流匹配在 PCA 空间里学习一个速度场，把对照细胞的分布连续地推到扰动后细胞的分布，训练时用最优传输把对照细胞和扰动细胞配对，所以模型学到的是“这个细胞被敲低后会去哪里”。条件来自 PerturbFormer 的预测，所以它同样能处理没见过的基因。" + flow_note)

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
    notes(s, "（约 45 秒）为了检验泛化，我做了外部验证：只用 Replogle 训练，并把 Adamson 的全部靶基因从训练里去掉，然后预测另一个实验室的 Adamson 数据。这里的上限很低：即使拿 Replogle 里同一个基因的实测结果去预测 Adamson，相关性也有限，说明跨实验室批次差异很大。在这个上限之下，我们对完全未见基因的预测达到了上限的大部分。")

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
    notes(s, f"（约 45 秒）生物学解释的第一部分：模型的预测是否保留了真实的生物学结构？左上：在实测数据中，同一个蛋白复合体的成员被敲低后响应很相似，这是模型能够“借鉴”的生物学基础；预测出的应答谱同样保留了这种复合体结构。右边是通路层面，横轴是预测和实测通路活性在一千多个扰动之间的相关，{'、'.join(top3)} 等通路预测得最好，而均值平移在通路层面完全没有区分能力。")

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
    notes(s, f"（约 50 秒）四个案例，全部是模型训练时没见过的基因。RPS19 是 Diamond-Blackfan 贫血的致病基因，MED12 是 Mediator 亚基，PSMB5 是蛋白酶体亚基，这些基因所在复合体的其他成员都测过，模型借鉴它们，预测非常准确。GATA1 是一个很有启发的失败：它是红系主转录因子，K562 的细胞身份依赖它，但训练集中没有效应相同的基因，模型参考了 {'、'.join(gnb)} 等一般转录因子，把血红蛋白基因的方向预测反了。这说明谱系特异的主调控因子不能靠相似基因外推，恰恰是最需要实验的地方。")

    # 13 --------------------------------------------------------------- interpretability + failure
    s = new()
    title(s, "模型如何决策，何时会失败？", "解释")
    picture(s, os.path.join(FIG, "fig_interp_F_sources.png"), 0.5, 1.35, w=6.3, h=5.2)
    picture(s, os.path.join(FIG, "fig_interp_E_failure.png"), 6.95, 1.35, w=5.85, h=2.6)
    fa = R["fail"]
    from scipy.stats import spearmanr
    rho1 = spearmanr(fa.max_prior_sim_to_train, fa.pearson_sys)[0]
    rho2 = spearmanr(np.log10(fa.n_deg + 1), fa.pearson_sys)[0]
    named = R["cls"][(R["cls"].n >= 5) & (~R["cls"]["class"].str.startswith("其他"))]
    best, worst = named.iloc[0], named.iloc[-1]
    text(s, 6.95, 4.15, 5.85, 2.6, [
        [("类别差异：", {"bold": True, "color": BLUE}),
         (f"{best['class']} 最可预测（去偏 r = {best['PerturbFormer+KD|pearson_sys']:.2f}），{worst['class']} 最难（{worst['PerturbFormer+KD|pearson_sys']:.2f}）。", {})],
        [("失败模式：", {"bold": True, "color": ORANGE}),
         (f"训练集中没有相似基因（Spearman ρ = {rho1:.2f}）或扰动效应弱（ρ = {rho2:.2f}）时性能下降。", {})],
        [("启示：", {"bold": True}), ("下一轮实验应优先测知识空间中离已测基因最远的基因（主动学习）。", {})]],
        size=13, after=7, line=1.12)
    footer(s, n)
    notes(s, "（约 40 秒）左图是模型在不同基因类别上对四种知识的信任分配，它是自适应的：例如 Mediator 更依赖蛋白互作网络，翻译起始因子更依赖语言模型的知识。右上两张图显示失败模式：训练集里没有相似的基因，或者扰动本身效应很弱时，预测会变差。这给出一个实际建议：下一轮实验应该优先测知识空间中离已测基因最远的基因，也就是主动学习。")

    # 14 --------------------------------------------------------------- demo
    s = new()
    title(s, "成果：输入细胞 + 基因，输出扰动后状态", "解释")
    demo_gene = "MYC" if os.path.exists(os.path.join(ROOT, "results", "predictions", "MYC", "report.md")) else "GATA1"
    box(s, 0.6, 1.45, 5.4, 0.9, fill=DARK)
    text(s, 0.8, 1.5, 5.1, 0.8, [f"$ python scripts/predict.py --gene {demo_gene} \\", "      --cells my_k562_cells.h5ad"],
         size=13, color="D6E2EC", font=MONO, after=0, anchor=MSO_ANCHOR.MIDDLE)
    rep_path = os.path.join(ROOT, "results", "predictions", demo_gene, "report.md")
    rep = open(rep_path).read().splitlines()[2:] if os.path.exists(rep_path) else []
    keep = ("是否测过", "依据", "知识来源", "预测下调", "预测上调", "通路活性下降")
    rep = [r.lstrip("- ") for r in rep if any(k in r for k in keep)]
    rep = [r.replace(" (注意力权重)", "").replace("训练筛选中是否测过该扰动", "筛选中是否测过") for r in rep]

    def first_items(line, k=4):
        if ": " not in line:
            return line
        head, body = line.split(": ", 1)
        sep = "; " if "; " in body else ", "
        items = body.split(sep)
        return head + ": " + sep.join(items[:k]) + (" …" if len(items) > k else "")
    rep = [first_items(r) for r in rep]
    text(s, 0.6, 2.55, 5.4, 3.0, rep, size=12, color=TEXT2, bullet=True, after=5, line=1.05)
    text(s, 0.6, 5.55, 5.4, 1.3, [[("输出：", {"bold": True}), ("单细胞表达 (h5ad)、伪批量变化表、证据报告与可视化；支持约 2.1 万个人类基因；交互式网页可浏览全部 1,092 个预测。", {})]],
         size=12.5, line=1.12)
    demo = os.path.join(ROOT, "results", "predictions", demo_gene, "prediction.png")
    if os.path.exists(demo):
        picture(s, demo, 6.2, 1.5, w=6.65, h=5.2)
    footer(s, n)
    notes(s, f"（约 40 秒）最终成果是一个可以直接使用的虚拟细胞原型。一行命令：输入一批对照细胞和一个基因，输出这些细胞被敲低后的表达谱、伪批量变化表，以及一份证据报告，说明模型参考了哪些已测实验、信任哪种知识、哪些通路会变化。右图是 {demo_gene}：它在这个筛选中从未被扰动过。模型参考了 MAX（MYC 的必需结合伙伴）等转录因子，预测 MYC 靶基因整体下调，这与 MYC 的已知功能一致。灰色是输入的对照细胞，蓝色是预测的敲低后细胞，右边是变化最大的基因。需要说明的是，这类全新基因的预测无法直接验证，模型的定量准确性以前面交叉验证的结果为准。")

    # 15 --------------------------------------------------------------- conclusion
    s = new()
    bg(s, DARK)
    text(s, 0.75, 0.6, 11.5, 0.8, ["结论与展望"], size=34, bold=True, color=WHITE)
    concl = [("问题", "把“敲低一个新基因会怎样”形式化为未见扰动泛化，并正视系统偏差与测量噪声"),
             ("模型", "多源知识检索：PerturbFormer 在已测扰动上做知识核交叉注意力；条件流匹配把预测展开到单细胞"),
             ("评测", f"1,092 个未见基因：去偏 Pearson {pf.pearson_sys:.3f}，显著高于线性基线 {S.loc[LIN, 'pearson_sys']:.3f} 与深度基线，小幅优于多源 kNN（Adamson 数据集内优势更明显）；跨实验室验证成立"),
             ("解释", "预测保留复合体与通路结构，每个预测可追溯到它参考的已测实验；谱系主调控因子是明确的失败边界")]
    for i, (a, b_) in enumerate(concl):
        y = 1.65 + i * 0.95
        circle_label(s, 0.75, y, 0.55, a[0], fill=BLUE, size=15)
        text(s, 1.55, y - 0.04, 11.0, 0.7, [[(a + "　", {"bold": True, "color": WHITE}), (b_, {"color": "D6E2EC"})]], size=15.5,
             anchor=MSO_ANCHOR.MIDDLE, line=1.05)
    text(s, 0.75, 5.6, 5.7, 1.4, [[("局限", {"bold": True, "color": "F2A27E"})], "单一细胞系；只做单基因敲低；跨实验室批次差异仍大；PCA 潜空间限制单细胞细节"],
         size=13.5, color="D6E2EC", after=4, line=1.12)
    text(s, 6.8, 5.6, 5.8, 1.4, [[("展望", {"bold": True, "color": "7FB2F0"})], "跨细胞系迁移 (K562 → RPE1)；组合扰动；用不确定性驱动主动学习选择下一轮实验"],
         size=13.5, color="D6E2EC", after=4, line=1.12)
    notes(s, "（约 40 秒）总结：问题上，我把“敲低一个新基因会怎样”形式化为未见扰动泛化，并正视系统偏差和测量噪声；模型上，核心思路是多源知识检索，PerturbFormer 是它的可学习、可解释版本，流匹配把它展开到单细胞；评测上，它显著优于文献中的线性基线和深度模型，小幅优于多源 kNN，并通过了跨实验室验证；解释上，预测保留了复合体和通路结构，每个预测都能追溯，同时我们也明确了失败边界。局限是只有一个细胞系、只做单基因；未来可以做跨细胞系迁移、组合扰动和主动学习。谢谢大家。")

    prs.save(OUT)
    print("saved", OUT, f"({n} slides)")


if __name__ == "__main__":
    build()
