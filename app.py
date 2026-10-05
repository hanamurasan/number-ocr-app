import io
import re
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image
from openpyxl import load_workbook
from openpyxl.utils.cell import (
    coordinate_from_string,
    column_index_from_string,
    get_column_letter,
)

B = Path(__file__).parent
st.set_page_config(page_title="位置補正版・一桁ずつ認識→Excel", page_icon="🔢", layout="wide")
st.title("🔢 文字を中央寄せして一桁ずつ認識")
st.caption("紫色の十字を色と形で検出し、その左上の数字3桁を読み取ります。小数点は認識対象から外します。")

L = pd.read_csv(B / "labels.csv", dtype=str)
CT = [cv2.imread(str(p), 0) for p in (B / "cross_templates").glob("*.png")]
CT = [x for x in CT if x is not None]


def timekey(name):
    m = re.search(r"_(\d+(?:\.\d+)?)s(?:\.[^.]+)?$", name, re.I)
    return (0, float(m.group(1))) if m else (1, name.lower())


def _template_cross(rgb):
    """色検出に失敗した場合だけ使う従来のテンプレート照合。"""
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    edge = cv2.Canny(gray, 45, 140)
    h, w = gray.shape
    best = (-2.0, None, None)
    for template in CT:
        z0 = cv2.Canny(template, 45, 140)
        for scale in np.linspace(0.45, 2.4, 32):
            tw = max(10, int(z0.shape[1] * scale))
            th = max(10, int(z0.shape[0] * scale))
            if tw >= w or th >= h:
                continue
            z = cv2.resize(z0, (tw, th))
            result = cv2.matchTemplate(edge, z, cv2.TM_CCOEFF_NORMED)
            _, score, _, loc = cv2.minMaxLoc(result)
            if score > best[0]:
                best = (score, (loc[0] + tw // 2, loc[1] + th // 2), max(tw, th))
    return best


def cross(rgb):
    """紫色の十字を色と十字形状で検出する。背景の模様は使わない。"""
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
    h, w = hsv.shape[:2]

    # OpenCV HSVで紫～マゼンタ。彩度の低い白・水色の十字は除外する。
    mask = cv2.inRange(hsv, np.array([135, 75, 70]), np.array([179, 255, 255]))
    mask[:int(h * 0.15), :] = 0
    mask[int(h * 0.90):, :] = 0
    mask[:, :int(w * 0.08)] = 0
    mask[:, int(w * 0.92):] = 0
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((2, 2), np.uint8))

    count, labels, stats, centers = cv2.connectedComponentsWithStats(mask)
    best = None
    best_score = -1e9
    min_side = min(h, w)

    for i in range(1, count):
        x, y, cw, ch, area = stats[i]
        if area < 18:
            continue
        if cw < min_side * 0.025 or ch < min_side * 0.025:
            continue
        if cw > min_side * 0.16 or ch > min_side * 0.16:
            continue
        ratio = cw / max(ch, 1)
        if not 0.55 <= ratio <= 1.80:
            continue

        cx, cy = centers[i]
        half = int(max(cw, ch) * 0.70)
        x1, x2 = max(0, int(cx) - half), min(w, int(cx) + half + 1)
        y1, y2 = max(0, int(cy) - half), min(h, int(cy) + half + 1)
        patch = mask[y1:y2, x1:x2] > 0
        if patch.size == 0:
            continue

        ph, pw = patch.shape
        band_x = max(1, pw // 8)
        band_y = max(1, ph // 8)
        vertical = patch[:, max(0, pw // 2 - band_x):min(pw, pw // 2 + band_x + 1)].mean()
        horizontal = patch[max(0, ph // 2 - band_y):min(ph, ph // 2 + band_y + 1), :].mean()
        corners = np.concatenate([
            patch[:max(1, ph // 4), :max(1, pw // 4)].ravel(),
            patch[:max(1, ph // 4), -max(1, pw // 4):].ravel(),
            patch[-max(1, ph // 4):, :max(1, pw // 4)].ravel(),
            patch[-max(1, ph // 4):, -max(1, pw // 4):].ravel(),
        ]).mean()
        plus_score = vertical + horizontal - 1.4 * corners
        area_score = min(area / max(cw * ch, 1), 0.55)
        score = plus_score + area_score

        if score > best_score:
            best_score = score
            # 色マスクの十字は線幅のみなので、従来ROIと同じ尺度になるよう約1.3倍
            size = int(max(cw, ch) * 1.30)
            best = (float(score), (int(round(cx)), int(round(cy))), max(size, 20))

    # 誤検出のまま切り出さず、十分十字らしい候補がない時だけ従来法へ戻る
    if best is not None and best_score >= 0.55:
        return best
    return _template_cross(rgb)

def roi(rgb, center, size):
    cx, cy = center
    unit = size / 30.0
    h, w = rgb.shape[:2]
    box = (
        max(0, int(cx - 48 * unit)),
        max(0, int(cy - 33 * unit)),
        min(w, int(cx - 1 * unit)),
        min(h, int(cy - 3 * unit)),
    )
    return rgb[box[1]:box[3], box[0]:box[2]], box


def rotate_keep(gray, angle):
    h, w = gray.shape
    matrix = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
    return cv2.warpAffine(
        gray, matrix, (w, h), flags=cv2.INTER_CUBIC,
        borderMode=cv2.BORDER_REPLICATE
    )


def angle_score(gray):
    """数字の横線が同じ高さに集まるほど大きくなるスコア。輪郭は使わない。"""
    blur = cv2.GaussianBlur(gray, (3, 3), 0)
    edge = cv2.Sobel(blur, cv2.CV_32F, 0, 1, ksize=3)
    edge = np.abs(edge)
    h, w = edge.shape
    middle = edge[int(h * 0.12):int(h * 0.90), int(w * 0.05):int(w * 0.95)]
    projection = middle.sum(axis=1)
    return float(np.var(projection))


def deskew(gray):
    """-10～10度を試し、横方向の並びが最もそろう角度へ補正する。"""
    small = cv2.resize(gray, (188, 120), interpolation=cv2.INTER_AREA)
    best_angle = 0.0
    best_score = angle_score(small)
    for angle in np.arange(-10.0, 10.01, 1.0):
        candidate = rotate_keep(small, angle)
        score = angle_score(candidate)
        if score > best_score:
            best_score = score
            best_angle = float(angle)
    return rotate_keep(gray, best_angle), best_angle


def longest_active_span(profile, threshold, max_gap=4):
    """投影値が高い範囲を、小さな切れ目を埋めながら一つの連続範囲にする。"""
    active = profile > threshold
    if not active.any():
        return None

    # 小数点や桁間で分断されないよう短い空白を埋める
    active = active.astype(np.uint8)
    kernel = np.ones(max_gap * 2 + 1, np.uint8)
    closed = cv2.morphologyEx(active.reshape(1, -1), cv2.MORPH_CLOSE, kernel).ravel() > 0
    indexes = np.where(closed)[0]
    if len(indexes) == 0:
        return None
    return int(indexes[0]), int(indexes[-1] + 1)


def align_number(gray):
    """正しく検出した十字基準ROIを、そのまま共通サイズへ正規化する。"""
    aligned = cv2.resize(gray, (126, 72), interpolation=cv2.INTER_CUBIC)
    return aligned, aligned.copy(), (0, 0, 126, 72), 0.0


# 小数点の場所を空けた固定3スロット。小数点は読み取らず、最後に規則で挿入する。
SLOTS = [(0, 39), (35, 74), (82, 124)]
DECIMAL_GAP = (74, 82)


def normalize_character(aligned, position):
    """各枠内の文字を投影で中央寄せする。輪郭検出は使わない。"""
    x1, x2 = SLOTS[position]
    part = aligned[2:70, x1:x2].copy()
    part = cv2.GaussianBlur(part, (3, 3), 0)

    # 明るい文字・暗い文字の両方を試し、前景が少ない方を採用
    _, dark = cv2.threshold(part, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    _, light = cv2.threshold(part, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    dark_ratio = np.mean(dark > 0)
    light_ratio = np.mean(light > 0)
    binary = dark if abs(dark_ratio - 0.24) <= abs(light_ratio - 0.24) else light

    # 枠の端に入った隣の数字や線を弱める
    binary[:, :2] = 0
    binary[:, -2:] = 0
    binary[:2, :] = 0
    binary[-2:, :] = 0

    row = np.count_nonzero(binary, axis=1)
    col = np.count_nonzero(binary, axis=0)
    ys = np.where(row >= max(1, int(binary.shape[1] * 0.06)))[0]
    xs = np.where(col >= max(1, int(binary.shape[0] * 0.06)))[0]

    if len(xs) == 0 or len(ys) == 0:
        crop = binary
    else:
        crop = binary[ys[0]:ys[-1] + 1, xs[0]:xs[-1] + 1]

    h, w = crop.shape
    scale = min(30 / max(w, 1), 46 / max(h, 1))
    nw, nh = max(1, int(w * scale)), max(1, int(h * scale))
    resized = cv2.resize(crop, (nw, nh), interpolation=cv2.INTER_AREA)
    canvas = np.zeros((52, 36), np.uint8)
    ox, oy = (36 - nw) // 2, (52 - nh) // 2
    canvas[oy:oy + nh, ox:ox + nw] = resized
    return canvas


def charfeat(aligned, position):
    """二値形状と縦横勾配を組み合わせた特徴量。"""
    glyph = normalize_character(aligned, position).astype(np.float32) / 255.0
    gx = cv2.Sobel(glyph, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(glyph, cv2.CV_32F, 0, 1, ksize=3)
    feature = np.concatenate([glyph.ravel(), (gx * 0.35).ravel(), (gy * 0.35).ravel()])
    norm = np.linalg.norm(feature)
    return feature / (norm + 1e-6)


# 学習画像にも新規画像にも同じ自動整列を適用する
T = []
training_failures = []
for _, row in L.iterrows():
    path = B / "training" / row.filename
    if not path.exists():
        training_failures.append(str(row.filename))
        continue

    rgb = np.array(Image.open(path).convert("RGB"))
    score, center, size = cross(rgb)
    label_digits = re.sub(r"\D", "", str(row.value))
    if not center or len(label_digits) != 3:
        training_failures.append(str(row.filename))
        continue

    region, _ = roi(rgb, center, size)
    if region.size == 0:
        training_failures.append(str(row.filename))
        continue

    gray = cv2.cvtColor(region, cv2.COLOR_RGB2GRAY)
    aligned, _, _, _ = align_number(gray)
    for position, digit in enumerate(label_digits):
        T.append((position, digit, charfeat(aligned, position)))


def recognize(rgb):
    score, center, size = cross(rgb)
    if not center:
        return "", 0.0, None, None, [], None, None, 0.0

    region, box = roi(rgb, center, size)
    if region.size == 0:
        return "", 0.0, box, center, [], None, None, 0.0

    gray = cv2.cvtColor(region, cv2.COLOR_RGB2GRAY)
    aligned, corrected, align_box, angle = align_number(gray)
    output = ""
    details = []
    scores = []

    for position in range(3):
        query = charfeat(aligned, position)
        candidates = [
            (float(np.dot(query, template)), digit)
            for _template_position, digit, template in T
        ]
        if not candidates:
            return "", 0.0, box, center, details, aligned, align_box, angle

        # 各数字について上位3テンプレートの平均を使う。
        # 1枚だけ偶然似た画像に引っ張られにくくする。
        grouped = {}
        for similarity, digit in candidates:
            grouped.setdefault(digit, []).append(similarity)
        ranking = []
        for digit, values in grouped.items():
            top = sorted(values, reverse=True)[:3]
            ranking.append((float(np.mean(top)), digit))
        ranking.sort(reverse=True)
        output += ranking[0][1]
        scores.append(ranking[0][0])
        details.append(ranking[:4])

    value = output[:-1] + "." + output[-1]
    return value, float(np.mean(scores)), box, center, details, aligned, align_box, angle


def cellok(value):
    return bool(re.fullmatch(r"[A-Za-z]{1,3}[1-9][0-9]*", value.strip()))


if training_failures:
    with st.expander(f"学習に使えなかった画像：{len(training_failures)}枚"):
        st.write("、".join(training_failures))

st.write(f"使用できる一桁テンプレート：{len(T)}個")
excel = st.file_uploader("1. 入力先Excel", type=["xlsx"])
files = sorted(
    st.file_uploader(
        "2. 写真を選択",
        type=["png", "jpg", "jpeg", "webp"],
        accept_multiple_files=True,
    ) or [],
    key=lambda f: timekey(f.name),
)

if files:
    st.info("時間順：" + " → ".join(x.name for x in files))

rows = []
for i, file in enumerate(files):
    rgb = np.array(Image.open(file).convert("RGB"))
    value, confidence, box, center, details, aligned, align_box, angle = recognize(rgb)
    marked = rgb.copy()

    if box:
        cv2.rectangle(marked, (box[0], box[1]), (box[2], box[3]), (0, 255, 0), 2)
    if center:
        cv2.drawMarker(marked, center, (255, 0, 255), cv2.MARKER_CROSS, 20, 2)

    left, right = st.columns([1, 2])
    left.image(marked, caption=file.name, width="stretch")
    value = right.text_input("認識結果", value, key=str(i) + file.name)
    right.write(f"角度補正：{angle:+.1f}°　3桁の平均類似度：{confidence:.2f}")

    if aligned is not None:
        preview = cv2.cvtColor(aligned, cv2.COLOR_GRAY2RGB)
        for x1, x2 in SLOTS:
            cv2.rectangle(preview, (x1, 3), (x2, 69), (255, 0, 0), 1)
        cv2.rectangle(preview, (DECIMAL_GAP[0], 48), (DECIMAL_GAP[1], 69), (0, 255, 255), 1)
        right.image(preview, caption="赤枠＝数字3桁、黄色枠＝無視する小数点部分", width="stretch")
        digit_previews = [normalize_character(aligned, k) for k in range(3)]
        right.image(digit_previews, caption=["1桁目", "2桁目", "3桁目"], width=90)

    if confidence < 0.45:
        right.warning("信頼度が低いため確認してください。")

    with right.expander("各桁の候補"):
        for k, candidates in enumerate(details):
            right.write(
                f"{k + 1}桁目：" +
                ", ".join(f"{digit}({similarity:.2f})" for similarity, digit in candidates)
            )

    rows.append({
        "filename": file.name,
        "value": value,
        "confidence": round(confidence, 3),
        "angle": round(angle, 2),
    })

if rows:
    st.dataframe(pd.DataFrame(rows), width="stretch")
    if excel:
        workbook = load_workbook(io.BytesIO(excel.getvalue()))
        sheet_name = st.selectbox("入力シート", workbook.sheetnames)
        start = st.text_input("開始セル", "C4")

        if cellok(start):
            letters, first_row = coordinate_from_string(start.upper())
            column = column_index_from_string(letters)
            worksheet = workbook[sheet_name]
            invalid = []

            for offset, row in enumerate(rows):
                try:
                    numeric = float(row["value"])
                    cell = worksheet.cell(first_row + offset, column, numeric)
                    cell.number_format = "0.0"
                except (TypeError, ValueError):
                    invalid.append(row["filename"])

            if invalid:
                st.error("数値に変換できない画像：" + "、".join(invalid))
            else:
                end = f"{get_column_letter(column)}{first_row + len(rows) - 1}"
                output = io.BytesIO()
                workbook.save(output)
                st.success(f"{start.upper()}:{end}へ入力しました")
                st.download_button(
                    "入力済みExcelをダウンロード",
                    output.getvalue(),
                    f"{Path(excel.name).stem}_入力済み.xlsx",
                )
