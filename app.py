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
st.set_page_config(page_title="輪郭検出で一桁ずつ認識→Excel", page_icon="🔢", layout="wide")
st.title("🔢 角度補正＋輪郭検出で数字を一桁ずつ認識")
st.caption("数字3文字を輪郭から検出し、最後の数字の前に小数点を自動挿入します。")

L = pd.read_csv(B / "labels.csv", dtype=str)
CT = [cv2.imread(str(p), 0) for p in (B / "cross_templates").glob("*.png")]
CT = [x for x in CT if x is not None]


def timekey(name):
    m = re.search(r"_(\d+(?:\.\d+)?)s(?:\.[^.]+)?$", name, re.I)
    return (0, float(m.group(1))) if m else (1, name.lower())


def cross(rgb):
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    edge = cv2.Canny(gray, 45, 140)
    h, w = gray.shape
    best = (-2.0, None, None)

    for template in CT:
        template_edge = cv2.Canny(template, 45, 140)
        for scale in np.linspace(0.45, 2.4, 32):
            tw = max(10, int(template_edge.shape[1] * scale))
            th = max(10, int(template_edge.shape[0] * scale))
            if tw >= w or th >= h:
                continue

            resized = cv2.resize(template_edge, (tw, th))
            result = cv2.matchTemplate(edge, resized, cv2.TM_CCOEFF_NORMED)
            mask = np.full(result.shape, -2, np.float32)
            y1, y2 = int(result.shape[0] * 0.12), int(result.shape[0] * 0.90)
            x1, x2 = int(result.shape[1] * 0.10), int(result.shape[1] * 0.90)
            mask[y1:y2, x1:x2] = result[y1:y2, x1:x2]
            _, score, _, loc = cv2.minMaxLoc(mask)

            if score > best[0]:
                best = (score, (loc[0] + tw // 2, loc[1] + th // 2), max(tw, th))
    return best


def roi(rgb, center, size):
    cx, cy = center
    unit = size / 30.0
    h, w = rgb.shape[:2]
    box = (
        max(0, int(cx - 45 * unit)),
        max(0, int(cy - 30 * unit)),
        min(w, int(cx - 3 * unit)),
        min(h, int(cy - 6 * unit)),
    )
    return rgb[box[1]:box[3], box[0]:box[2]], box


def normroi(x):
    if x is None or x.size == 0:
        return None
    gray = cv2.cvtColor(x, cv2.COLOR_RGB2GRAY)
    return cv2.resize(gray, (252, 144), interpolation=cv2.INTER_CUBIC)


def binary_image(gray):
    blur = cv2.GaussianBlur(gray, (3, 3), 0)
    bw = cv2.adaptiveThreshold(
        blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV, 31, 7
    )
    # 細い切れ目だけをつなぎ、隣の数字同士は結合しにくい縦長カーネル
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 3))
    return cv2.morphologyEx(bw, cv2.MORPH_CLOSE, kernel, iterations=1)


def raw_digit_boxes(bw):
    """数字らしい輪郭だけを残す。ROIの枠・文字・十字の残りは除外する。"""
    h, w = bw.shape

    # 現在のROIでは数字は中央付近に並ぶため、上下端と左右端を探索対象から外す
    work = bw.copy()
    work[:int(h * 0.12), :] = 0
    work[int(h * 0.90):, :] = 0
    work[:, :int(w * 0.03)] = 0
    work[:, int(w * 0.97):] = 0

    contours, _ = cv2.findContours(work, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    candidates = []
    for contour in contours:
        x, y, cw, ch = cv2.boundingRect(contour)
        area = cv2.contourArea(contour)
        box_area = max(cw * ch, 1)
        fill = area / box_area
        cy = y + ch / 2

        # 小数点・細線・外枠・巨大な背景輪郭を除外
        if not (h * 0.32 <= ch <= h * 0.82):
            continue
        if not (w * 0.025 <= cw <= w * 0.22):
            continue
        if not (h * 0.25 <= cy <= h * 0.78):
            continue
        if not (0.06 <= fill <= 0.80):
            continue
        if x <= 1 or y <= 1 or x + cw >= w - 1 or y + ch >= h - 1:
            continue
        candidates.append((x, y, cw, ch))

    # 高さと縦位置が近い3個を、全組合せから選ぶ
    if len(candidates) < 3:
        return sorted(candidates, key=lambda b: b[0])

    from itertools import combinations
    best = None
    best_score = -1e18
    for group in combinations(candidates, 3):
        group = sorted(group, key=lambda b: b[0])
        heights = np.array([b[3] for b in group], dtype=float)
        centers = np.array([b[1] + b[3] / 2 for b in group], dtype=float)
        widths = np.array([b[2] for b in group], dtype=float)
        left_gap = group[1][0] - (group[0][0] + group[0][2])
        right_gap = group[2][0] - (group[1][0] + group[1][2])

        # 左から順に離れていて、同じ行・同程度の高さの3個を優先
        if left_gap < -w * 0.03 or right_gap < -w * 0.03:
            continue
        span = (group[2][0] + group[2][2]) - group[0][0]
        score = (
            3.0 * heights.mean()
            - 4.0 * heights.std()
            - 3.0 * centers.std()
            - 0.8 * widths.std()
            + 0.25 * span
            - 0.25 * abs(left_gap - right_gap)
        )
        if score > best_score:
            best_score = score
            best = group
    return best or []

def rotate_gray(gray, angle):
    h, w = gray.shape
    matrix = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
    return cv2.warpAffine(
        gray, matrix, (w, h), flags=cv2.INTER_CUBIC,
        borderMode=cv2.BORDER_REPLICATE
    )


def deskew(gray):
    """数字候補の中心を直線近似し、数字列が水平になるように回転する。"""
    bw = binary_image(gray)
    boxes = raw_digit_boxes(bw)
    if len(boxes) < 2:
        return gray, 0.0

    # 大きい候補を最大5個に絞り、x順で傾きを求める
    boxes = sorted(boxes, key=lambda b: b[2] * b[3], reverse=True)[:5]
    boxes = sorted(boxes, key=lambda b: b[0])
    xs = np.array([x + w / 2 for x, y, w, h in boxes], dtype=np.float32)
    ys = np.array([y + h / 2 for x, y, w, h in boxes], dtype=np.float32)
    if float(xs.max() - xs.min()) < 10:
        return gray, 0.0

    slope = float(np.polyfit(xs, ys, 1)[0])
    angle = float(np.degrees(np.arctan(slope)))
    angle = float(np.clip(angle, -18.0, 18.0))
    return rotate_gray(gray, angle), angle


def choose_three_boxes(boxes):
    """数字候補が多い場合、近い高さ・大きさで左から連続する3個を選ぶ。"""
    if len(boxes) < 3:
        return []
    if len(boxes) == 3:
        return boxes

    best_score = -1e9
    best = None
    for i in range(len(boxes) - 2):
        group = boxes[i:i + 3]
        heights = np.array([b[3] for b in group], dtype=float)
        centers_y = np.array([b[1] + b[3] / 2 for b in group], dtype=float)
        gaps = np.array([
            group[1][0] - (group[0][0] + group[0][2]),
            group[2][0] - (group[1][0] + group[1][2]),
        ], dtype=float)
        score = (
            heights.mean()
            - 2.0 * heights.std()
            - 1.5 * centers_y.std()
            - 0.5 * abs(gaps[0] - gaps[1])
        )
        if score > best_score:
            best_score = score
            best = group
    return best or []


def extract_digits(gray):
    corrected, angle = deskew(gray)
    bw = binary_image(corrected)
    boxes = choose_three_boxes(raw_digit_boxes(bw))
    digits = []

    for x, y, w, h in boxes:
        pad_x = max(2, int(w * 0.15))
        pad_y = max(2, int(h * 0.08))
        x1, y1 = max(0, x - pad_x), max(0, y - pad_y)
        x2, y2 = min(bw.shape[1], x + w + pad_x), min(bw.shape[0], y + h + pad_y)
        digits.append(bw[y1:y2, x1:x2])
    return digits, boxes, corrected, bw, angle


def charfeat(binary_digit):
    """輪郭で切り出した1文字を余白付き48×64へ正規化する。"""
    if binary_digit is None or binary_digit.size == 0:
        return None
    ys, xs = np.where(binary_digit > 0)
    if len(xs) == 0:
        return None
    crop = binary_digit[ys.min():ys.max() + 1, xs.min():xs.max() + 1]
    h, w = crop.shape
    scale = min(38 / max(w, 1), 54 / max(h, 1))
    nw, nh = max(1, int(w * scale)), max(1, int(h * scale))
    resized = cv2.resize(crop, (nw, nh), interpolation=cv2.INTER_AREA)
    canvas = np.zeros((64, 48), np.uint8)
    x0, y0 = (48 - nw) // 2, (64 - nh) // 2
    canvas[y0:y0 + nh, x0:x0 + nw] = resized
    feat = cv2.GaussianBlur(canvas, (3, 3), 0).astype(np.float32) / 255.0
    return (feat - feat.mean()) / (feat.std() + 1e-6)


# 学習画像から各桁テンプレートを作成
T = []
training_failures = []
for _, row in L.iterrows():
    path = B / "training" / row.filename
    if not path.exists():
        training_failures.append(str(row.filename))
        continue
    rgb = np.array(Image.open(path).convert("RGB"))
    score, center, size = cross(rgb)
    if not center:
        training_failures.append(str(row.filename))
        continue
    region, _ = roi(rgb, center, size)
    gray = normroi(region)
    if gray is None:
        training_failures.append(str(row.filename))
        continue
    digit_images, _, _, _, _ = extract_digits(gray)
    label_digits = re.sub(r"\D", "", str(row.value))
    if len(digit_images) != 3 or len(label_digits) != 3:
        training_failures.append(str(row.filename))
        continue
    for position, (digit, image) in enumerate(zip(label_digits, digit_images)):
        feature = charfeat(image)
        if feature is not None:
            T.append((position, digit, feature))


def recognize(rgb):
    score, center, size = cross(rgb)
    if not center:
        return "", 0.0, None, None, [], None, [], 0.0

    region, box = roi(rgb, center, size)
    gray = normroi(region)
    if gray is None:
        return "", 0.0, box, center, [], None, [], 0.0

    digit_images, digit_boxes, corrected, bw, angle = extract_digits(gray)
    if len(digit_images) != 3:
        return "", 0.0, box, center, [], bw, digit_boxes, angle

    output = ""
    details = []
    scores = []
    for position, digit_image in enumerate(digit_images):
        query = charfeat(digit_image)
        candidates = [
            (float((query * template).mean()), digit)
            for pos, digit, template in T
            if pos == position and query is not None
        ]
        if not candidates:
            return "", 0.0, box, center, details, bw, digit_boxes, angle

        # 同じ数字の複数テンプレートのうち最高値を代表値にする
        best_by_digit = {}
        for similarity, digit in candidates:
            best_by_digit[digit] = max(similarity, best_by_digit.get(digit, -1e9))
        ranking = sorted(
            [(similarity, digit) for digit, similarity in best_by_digit.items()],
            reverse=True,
        )
        output += ranking[0][1]
        scores.append(ranking[0][0])
        details.append(ranking[:4])

    value = output[:-1] + "." + output[-1]
    return value, float(np.mean(scores)), box, center, details, bw, digit_boxes, angle


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
    value, confidence, box, center, details, bw, digit_boxes, angle = recognize(rgb)
    marked = rgb.copy()

    if box:
        cv2.rectangle(marked, (box[0], box[1]), (box[2], box[3]), (0, 255, 0), 2)
    if center:
        cv2.drawMarker(marked, center, (255, 0, 255), cv2.MARKER_CROSS, 20, 2)

    left, right = st.columns([1, 2])
    left.image(marked, caption=file.name, width="stretch")
    value = right.text_input("認識結果", value, key=str(i) + file.name)
    right.write(f"角度補正：{angle:+.1f}°　3桁の平均類似度：{confidence:.2f}")

    if bw is not None:
        preview = cv2.cvtColor(bw, cv2.COLOR_GRAY2RGB)
        for x, y, w, h in digit_boxes:
            cv2.rectangle(preview, (x, y), (x + w, y + h), (255, 0, 0), 2)
        right.image(preview, caption=f"赤枠が数字3文字を囲めているか確認：{len(digit_boxes)}個", width="stretch")

    if len(digit_boxes) != 3:
        right.error("数字を3個検出できませんでした。認識結果を手入力してください。")
    elif confidence < 0.45:
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
        "detected_digits": len(digit_boxes),
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
            invalid_values = []

            for offset, row in enumerate(rows):
                try:
                    numeric_value = float(row["value"])
                    cell = worksheet.cell(first_row + offset, column, numeric_value)
                    cell.number_format = "0.0"
                except (TypeError, ValueError):
                    invalid_values.append(row["filename"])

            if invalid_values:
                st.error("数値に変換できないためExcelへ入れられない画像：" + "、".join(invalid_values))
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
