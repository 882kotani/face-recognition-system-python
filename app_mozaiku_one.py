import os
import time
import uuid
import base64
import shutil
import threading
import subprocess
import urllib.request
from concurrent.futures import ProcessPoolExecutor, as_completed
import cv2
import numpy as np
from flask import Flask, render_template_string, request, Response, jsonify, send_file

try:
    from PIL import Image, ImageDraw, ImageFont
    HAS_PIL = True
except ImportError:
    HAS_PIL = False

app = Flask(__name__)
UPLOAD_FOLDER = 'uploads'
OUTPUT_FOLDER = 'outputs'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

# 1. 顔検出モデル (YuNet)
DET_MODEL_PATH = "face_detection_yunet_2023mar.onnx"
DET_MODEL_URL = "https://github.com/opencv/opencv_zoo/raw/main/models/face_detection_yunet/face_detection_yunet_2023mar.onnx"

# 2. 顔認識・特徴抽出モデル (SFace)
REC_MODEL_PATH = "face_recognition_sface_2021dec.onnx"
REC_MODEL_URL = "https://github.com/opencv/opencv_zoo/raw/main/models/face_recognition_sface/face_recognition_sface_2021dec.onnx"

def download_if_not_exists(path, url, name):
    if not os.path.exists(path):
        print(f"{name}をダウンロードしています...")
        urllib.request.urlretrieve(url, path)
        print(f"{name}のダウンロード完了。")

download_if_not_exists(DET_MODEL_PATH, DET_MODEL_URL, "YuNetモデル")
download_if_not_exists(REC_MODEL_PATH, REC_MODEL_URL, "SFaceモデル")

# 日本語フォント探索（映像上の人物名描画用）
JP_FONT_PATH = None
if HAS_PIL:
    font_candidates = [
        r"C:\Windows\Fonts\meiryo.ttc",
        r"C:\Windows\Fonts\YuGothM.ttc",
        r"C:\Windows\Fonts\msgothic.ttc",
        "/System/Library/Fonts/Hiragino Sans GB.ttc",
        "/System/Library/Fonts/AppleGothic.ttf",
        "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
        "/usr/share/fonts/truetype/fonts-japanese-gothic.ttf",
    ]
    for fp in font_candidates:
        if os.path.exists(fp):
            JP_FONT_PATH = fp
            break

_font_cache = {}
def get_jp_font(size=18):
    if not HAS_PIL or not JP_FONT_PATH:
        return None
    if size not in _font_cache:
        try:
            _font_cache[size] = ImageFont.truetype(JP_FONT_PATH, size)
        except Exception:
            _font_cache[size] = None
    return _font_cache[size]

# メインプロセス用認識器
recognizer = cv2.FaceRecognizerSF.create(model=REC_MODEL_PATH, config="")
cache_lock = threading.Lock()
state_lock = threading.Lock()

def create_default_person(index=1):
    return {
        "id": str(uuid.uuid4()),
        "name": f"人物 {index}",
        "entries": []  # [{"id": str, "crop_b64": str, "source": "photo"|"boost", "vectors": [np.ndarray, ...]}]
    }

# アプリケーション状態
state = {
    "stream_id": None,
    "video_path": None,
    "original_name": "output",
    "total_frames": 0,
    "fps": 30.0,
    "current_frame": 0,
    "seek_frame": None,
    "paused": False,
    "settings_dirty": False,
    # 複数人物管理リスト
    "persons": [create_default_person(1)],
    # 処理モード: "mosaic" (モザイク処理) or "notify" (ポップアップ顔検出通知)
    "action_mode": "mosaic",
    # モザイク対象: "target_only" or "except_target"
    "mosaic_target_mode": "target_only",
    # モザイクの度合い (4〜50: 大きいほど粗いモザイク)
    "mosaic_scale": 16,
    "threshold": 0.36,
    "interp_gap": 5,              # チラつき防止補間 (0〜50f)
    "show_debug": True,
    # 現在フレームの検出結果（通知ポーリング用）: [{"id": str, "name": str, "score": float}, ...]
    "latest_detections": [],
    # フレームごとの顔座標＋特徴量キャッシュ: { frame_idx: [(x, y, w, h, feat), ...] }
    "feature_cache": {},
    # 書き出し・ダウンロード状態
    "export_status": "idle",
    "export_progress": 0,
    "export_message": "",
    "output_path": None
}


def find_person(person_id):
    for p in state["persons"]:
        if p["id"] == person_id:
            return p
    return None


def get_person_vectors(person):
    vectors = []
    for entry in person.get("entries", []):
        vectors.extend(entry["vectors"])
    return vectors


def get_persons_snapshot():
    """推論・描画用に現在の登録人物と特徴ベクトル一覧のスナップショットを取得"""
    snapshot = []
    with state_lock:
        for p in state["persons"]:
            vecs = get_person_vectors(p)
            if vecs:
                snapshot.append({
                    "id": p["id"],
                    "name": p["name"],
                    "vectors": list(vecs)
                })
    return snapshot


def compute_max_similarity(face_feat, target_vectors):
    if not target_vectors:
        return 0.0
    max_score = -1.0
    for t_vec in target_vectors:
        score = recognizer.match(t_vec, face_feat, cv2.FaceRecognizerSF_FR_COSINE)
        if score > max_score:
            max_score = score
    return max_score


def match_face_to_persons(face_feat, persons_snapshot, threshold):
    """
    1つの顔特徴量を登録済みの全人物と照合し、最も一致度の高い人物を返す
    戻り値: (matched_person_id, matched_person_name, best_score, is_matched)
    """
    if not persons_snapshot:
        return None, None, 0.0, False

    best_id = None
    best_name = None
    best_score = -1.0

    for p in persons_snapshot:
        score = compute_max_similarity(face_feat, p["vectors"])
        if score > best_score:
            best_score = score
            best_id = p["id"]
            best_name = p["name"]

    if best_score >= threshold:
        return best_id, best_name, best_score, True
    return None, None, max(0.0, best_score), False


def compute_iou(box_a, box_b):
    ax1, ay1, aw, ah = box_a
    bx1, by1, bw, bh = box_b
    ax2, ay2 = ax1 + aw, ay1 + ah
    bx2, by2 = bx1 + bw, by1 + bh

    inter_x1 = max(ax1, bx1)
    inter_y1 = max(ay1, by1)
    inter_x2 = min(ax2, bx2)
    inter_y2 = min(ay2, by2)

    inter_w = max(0, inter_x2 - inter_x1)
    inter_h = max(0, inter_y2 - inter_y1)
    inter_area = inter_w * inter_h
    if inter_area == 0:
        return 0.0

    union_area = (aw * ah) + (bw * bh) - inter_area
    return inter_area / max(1, union_area)


def evaluate_frame_faces(frame_faces, persons_snapshot, action_mode, mosaic_target_mode, threshold):
    """
    1フレームの抽出済み顔リストに対し、複数人物照合とモザイク/通知対象判定を評価する
    戻り値: [(x, y, w, h, is_target, score, should_mosaic, person_id, person_name), ...]
    """
    has_any_person = len(persons_snapshot) > 0
    evaluated = []

    for x, y, w, h, feat in frame_faces:
        p_id, p_name, score, is_matched = match_face_to_persons(feat, persons_snapshot, threshold)

        if action_mode == "notify":
            # ポップアップ顔検出通知モードではモザイクはかけない
            should_mosaic = False
        else:
            # モザイク処理モード
            if not has_any_person:
                should_mosaic = True
            elif mosaic_target_mode == "target_only":
                should_mosaic = is_matched
            else:  # "except_target"
                should_mosaic = not is_matched

        evaluated.append((x, y, w, h, is_matched, score, should_mosaic, p_id, p_name))
    return evaluated


def build_interpolated_items(target_idx, evaluated_map, max_gap, action_mode):
    """
    前後フレームの追跡対象枠を参照し、欠損フレームを線形補間してチラつきを防止する
    戻り値: [(x, y, w, h, is_target, score, should_mosaic, is_interp, person_id, person_name), ...]
    """
    raw_items = evaluated_map.get(target_idx, [])
    result_items = [
        (x, y, w, h, is_target, score, should_mosaic, False, p_id, p_name)
        for (x, y, w, h, is_target, score, should_mosaic, p_id, p_name) in raw_items
    ]

    if max_gap <= 0:
        return result_items

    # 追跡対象となる条件（モザイクモードならモザイク対象枠、通知モードなら登録人物一致枠）
    def is_trackable(item_tuple):
        _, _, _, _, is_t, _, sm, _, _ = item_tuple
        return is_t if action_mode == "notify" else sm

    covered_boxes = [
        (x, y, w, h)
        for (x, y, w, h, is_t, _, sm, _, _, _) in result_items
        if (is_t if action_mode == "notify" else sm)
    ]

    past_map = {}
    for p in range(target_idx - max_gap, target_idx):
        if p in evaluated_map:
            b_list = [
                (x, y, w, h, p_id, p_name)
                for (x, y, w, h, is_t, _, sm, p_id, p_name) in evaluated_map[p]
                if is_trackable((x, y, w, h, is_t, 0, sm, p_id, p_name))
            ]
            if b_list:
                past_map[p] = b_list

    if not past_map:
        return result_items

    future_map = {}
    for n in range(target_idx + 1, target_idx + max_gap + 1):
        if n in evaluated_map:
            b_list = [
                (x, y, w, h, p_id, p_name)
                for (x, y, w, h, is_t, _, sm, p_id, p_name) in evaluated_map[n]
                if is_trackable((x, y, w, h, is_t, 0, sm, p_id, p_name))
            ]
            if b_list:
                future_map[n] = b_list

    if not future_map:
        return result_items

    valid_pairs = []
    for p, boxes_p in past_map.items():
        for n, boxes_n in future_map.items():
            span = n - p
            if span <= max_gap + 1:
                valid_pairs.append((span, p, n, boxes_p, boxes_n))

    valid_pairs.sort(key=lambda item: item[0])

    for span, p, n, boxes_p, boxes_n in valid_pairs:
        pair_candidates = []
        for idx_p, bp in enumerate(boxes_p):
            xp, yp, wp, hp, pid_p, pname_p = bp
            cx_p, cy_p = xp + wp * 0.5, yp + hp * 0.5
            area_p = max(1, wp * hp)

            for idx_n, bn in enumerate(boxes_n):
                xn, yn, wn, hn, pid_n, pname_n = bn
                # 両端に人物IDが紐づいている場合、別人物同士は補間しない
                if pid_p is not None and pid_n is not None and pid_p != pid_n:
                    continue

                cx_n, cy_n = xn + wn * 0.5, yn + hn * 0.5
                area_n = max(1, wn * hn)

                area_ratio = area_n / area_p
                if area_ratio < 0.45 or area_ratio > 2.2:
                    continue

                avg_size = (wp + hp + wn + hn) * 0.25
                step_dist = np.hypot(cx_n - cx_p, cy_n - cy_p) / span
                if step_dist > avg_size * 0.65:
                    continue

                pair_candidates.append((step_dist, idx_p, idx_n, bp, bn))

        pair_candidates.sort(key=lambda item: item[0])
        used_p, used_n = set(), set()
        alpha = (target_idx - p) / float(span)

        for _, idx_p, idx_n, bp, bn in pair_candidates:
            if idx_p in used_p or idx_n in used_n:
                continue
            used_p.add(idx_p)
            used_n.add(idx_n)

            xp, yp, wp, hp, pid_p, pname_p = bp
            xn, yn, wn, hn, pid_n, pname_n = bn
            ix = int(round(xp + (xn - xp) * alpha))
            iy = int(round(yp + (yn - yp) * alpha))
            iw = int(round(wp + (wn - wp) * alpha))
            ih = int(round(hp + (hn - hp) * alpha))
            interp_box = (ix, iy, iw, ih)

            if all(compute_iou(interp_box, existing) < 0.35 for existing in covered_boxes):
                covered_boxes.append(interp_box)
                interp_pid = pid_p or pid_n
                interp_pname = pname_p or pname_n
                should_m = (action_mode == "mosaic")
                is_t = (interp_pid is not None)
                result_items.append((ix, iy, iw, ih, is_t, 0.0, should_m, True, interp_pid, interp_pname))

    return result_items


def analyze_video_chunk(video_path, frame_indices, det_model_path, rec_model_path):
    """マルチプロセスワーカー関数（フル解像度で顔検出と特徴量抽出）"""
    if not frame_indices:
        return {}

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return {}

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    detector = cv2.FaceDetectorYN.create(
        model=det_model_path,
        config="",
        input_size=(width, height),
        score_threshold=0.6,
        nms_threshold=0.3,
        top_k=5000
    )
    local_recognizer = cv2.FaceRecognizerSF.create(model=rec_model_path, config="")

    results = {}
    sorted_indices = sorted(frame_indices)
    current_pos = -1

    for target_idx in sorted_indices:
        if current_pos != target_idx:
            cap.set(cv2.CAP_PROP_POS_FRAMES, target_idx)
            current_pos = target_idx

        ret, frame = cap.read()
        current_pos += 1
        if not ret:
            break

        _, faces = detector.detect(frame)
        frame_faces = []
        if faces is not None:
            for face in faces:
                x, y, w, h = map(int, face[:4])
                aligned = local_recognizer.alignCrop(frame, face)
                feat = local_recognizer.feature(aligned)
                frame_faces.append((x, y, w, h, feat))

        results[target_idx] = frame_faces

    cap.release()
    return results


HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="ja">
<head>
    <meta charset="UTF-8">
    <title>AI 複数人物 顔認識モザイク＆検出通知</title>
    <style>
        body {
            font-family: 'Segoe UI', 'Meiryo', Arial, sans-serif;
            display: flex;
            flex-direction: column;
            align-items: center;
            padding: 20px;
            background-color: #f4f6f9;
            color: #333;
        }
        .container {
            max-width: 960px;
            width: 100%;
        }
        .top-grid {
            display: grid;
            grid-template-columns: 1.35fr 0.65fr;
            gap: 16px;
            margin-bottom: 16px;
        }
        @media (max-width: 820px) {
            .top-grid { grid-template-columns: 1fr; }
        }
        .card {
            background: #fff;
            border-radius: 8px;
            padding: 16px;
            box-shadow: 0 2px 6px rgba(0,0,0,0.06);
        }
        .card h3 {
            margin-top: 0;
            font-size: 16px;
            border-bottom: 1px solid #eee;
            padding-bottom: 8px;
            display: flex;
            justify-content: space-between;
            align-items: center;
            gap: 8px;
        }
        .add-person-btn {
            background: #007bff;
            color: #fff;
            border: none;
            border-radius: 5px;
            padding: 5px 12px;
            font-size: 12px;
            font-weight: bold;
            cursor: pointer;
            transition: 0.15s;
        }
        .add-person-btn:hover {
            background: #0056b3;
        }
        #persons-container {
            display: flex;
            flex-direction: column;
            gap: 12px;
            max-height: 420px;
            overflow-y: auto;
            padding-right: 4px;
        }
        .person-card {
            border: 1px solid #dce3ec;
            border-left: 4px solid #007bff;
            border-radius: 6px;
            padding: 10px 12px;
            background: #fcfdff;
        }
        .person-header {
            display: flex;
            align-items: center;
            justify-content: space-between;
            gap: 8px;
            margin-bottom: 8px;
        }
        .person-name-input {
            font-size: 14px;
            font-weight: bold;
            padding: 4px 8px;
            border: 1px solid #ced4da;
            border-radius: 4px;
            width: 180px;
            color: #222;
        }
        .person-name-input:focus {
            outline: none;
            border-color: #007bff;
            box-shadow: 0 0 0 2px rgba(0,123,255,0.15);
        }
        .person-drop-zone {
            border: 1.5px dashed #7aa7e9;
            border-radius: 5px;
            padding: 8px;
            text-align: center;
            font-size: 12px;
            color: #555;
            background: #fff;
            cursor: pointer;
            transition: 0.15s;
        }
        .person-drop-zone.dragover {
            background: #e3f2fd;
            border-color: #0056b3;
        }
        .target-list {
            display: flex;
            flex-wrap: wrap;
            gap: 6px;
            margin-top: 8px;
        }
        .target-chip {
            position: relative;
            width: 48px;
            height: 48px;
            border-radius: 5px;
            overflow: hidden;
            border: 2px solid #28a745;
        }
        .target-chip.boost {
            border-color: #ff9800;
        }
        .target-chip img {
            width: 100%;
            height: 100%;
            object-fit: cover;
        }
        .target-chip .badge {
            position: absolute;
            bottom: 0;
            left: 0;
            right: 0;
            font-size: 8px;
            text-align: center;
            background: rgba(0,0,0,0.65);
            color: #fff;
            padding: 1px 0;
        }
        .target-chip .del-btn {
            position: absolute;
            top: 1px;
            right: 1px;
            background: rgba(220, 53, 69, 0.9);
            color: #fff;
            border: none;
            border-radius: 50%;
            width: 15px;
            height: 15px;
            font-size: 10px;
            line-height: 13px;
            cursor: pointer;
            padding: 0;
        }
        .person-actions {
            display: flex;
            flex-wrap: wrap;
            gap: 6px;
            margin-top: 8px;
            align-items: center;
        }
        .person-msg {
            font-size: 11px;
            color: #666;
            margin-top: 4px;
        }
        .sub-btn {
            background: #f8f9fa;
            border: 1px solid #ced4da;
            border-radius: 4px;
            padding: 4px 8px;
            font-size: 11px;
            cursor: pointer;
            color: #333;
            transition: 0.15s;
        }
        .sub-btn:hover:not(:disabled) {
            background: #e2e6ea;
        }
        .sub-btn.accent {
            background: #fff3cd;
            border-color: #ffc107;
            color: #856404;
            font-weight: bold;
        }
        .sub-btn.danger {
            color: #dc3545;
            border-color: #f5c6cb;
            background: #fff5f5;
        }
        .sub-btn.danger:hover {
            background: #f8d7da;
        }
        .sub-btn:disabled {
            opacity: 0.5;
            cursor: not-allowed;
        }
        .video-drop-zone {
            height: 120px;
            border: 2px dashed #007bff;
            border-radius: 6px;
            display: flex;
            flex-direction: column;
            justify-content: center;
            align-items: center;
            background: #fafcff;
            cursor: pointer;
            text-align: center;
            padding: 10px;
            transition: 0.2s;
            font-size: 13px;
            color: #555;
        }
        .video-drop-zone.dragover {
            background: #e3f2fd;
            border-color: #0056b3;
        }
        /* コントロールパネル */
        .settings-card {
            margin-bottom: 14px;
            display: flex;
            flex-direction: column;
            gap: 12px;
        }
        .settings-row {
            display: flex;
            flex-wrap: wrap;
            gap: 14px 22px;
            align-items: center;
            justify-content: space-between;
        }
        .mode-switch {
            display: flex;
            gap: 10px;
            background: #eef2f7;
            padding: 4px;
            border-radius: 6px;
        }
        .mode-switch label {
            padding: 6px 14px;
            border-radius: 5px;
            cursor: pointer;
            font-size: 13px;
            font-weight: bold;
            transition: 0.15s;
            display: flex;
            align-items: center;
            gap: 6px;
        }
        .mode-switch input[type="radio"] {
            accent-color: #007bff;
        }
        .control-group {
            display: flex;
            align-items: center;
            gap: 8px;
            font-size: 13px;
        }
        #mosaic-options-bar {
            display: flex;
            flex-wrap: wrap;
            gap: 16px 24px;
            align-items: center;
            padding-top: 10px;
            border-top: 1px solid #eee;
        }
        #notify-options-bar {
            display: none;
            padding-top: 10px;
            border-top: 1px solid #eee;
            font-size: 13px;
            color: #0056b3;
            align-items: center;
            justify-content: space-between;
            flex-wrap: wrap;
            gap: 10px;
        }
        #active-detect-badges {
            display: flex;
            flex-wrap: wrap;
            gap: 6px;
        }
        .detect-pill {
            background: #28a745;
            color: #fff;
            padding: 3px 10px;
            border-radius: 12px;
            font-size: 12px;
            font-weight: bold;
            animation: pulseIn 0.2s ease;
        }
        @keyframes pulseIn {
            from { transform: scale(0.9); opacity: 0.6; }
            to { transform: scale(1); opacity: 1; }
        }
        #status {
            text-align: center;
            margin-bottom: 10px;
            font-weight: bold;
            color: #0056b3;
        }
        .player-wrapper {
            width: 100%;
            background: #000;
            border-radius: 8px;
            overflow: hidden;
            box-shadow: 0 4px 12px rgba(0,0,0,0.15);
            margin-bottom: 16px;
            position: relative;
        }
        #preview-box {
            width: 100%;
            display: flex;
            justify-content: center;
            align-items: center;
            min-height: 240px;
            background: #111;
        }
        #stream-img {
            width: 100%;
            height: auto;
            display: none;
        }
        #player-bar {
            display: none;
            background: #1e2229;
            color: #fff;
            padding: 10px 16px 14px;
            flex-direction: column;
            gap: 8px;
        }
        #seek-bar {
            width: 100%;
            cursor: pointer;
            accent-color: #007bff;
            height: 6px;
        }
        .player-buttons {
            display: flex;
            align-items: center;
            justify-content: space-between;
            flex-wrap: wrap;
            gap: 8px;
        }
        .btn-group {
            display: flex;
            align-items: center;
            gap: 8px;
        }
        .player-btn {
            background: #2f3542;
            color: #fff;
            border: 1px solid #444c5c;
            border-radius: 4px;
            padding: 5px 12px;
            font-size: 13px;
            cursor: pointer;
            transition: 0.15s;
        }
        .player-btn:hover {
            background: #007bff;
            border-color: #007bff;
        }
        #time-display {
            font-family: monospace;
            font-size: 14px;
            color: #d1d8e0;
        }
        #export-card {
            display: none;
            align-items: center;
            justify-content: space-between;
            flex-wrap: wrap;
            gap: 16px;
        }
        .primary-btn {
            background: #28a745;
            color: #fff;
            border: none;
            border-radius: 6px;
            padding: 10px 20px;
            font-size: 15px;
            font-weight: bold;
            cursor: pointer;
            transition: 0.2s;
        }
        .primary-btn:hover:not(:disabled) {
            background: #218838;
        }
        .primary-btn:disabled {
            background: #94d3a2;
            cursor: not-allowed;
        }
        .download-link-btn {
            display: none;
            background: #007bff;
            color: #fff;
            text-decoration: none;
            border-radius: 6px;
            padding: 10px 18px;
            font-size: 14px;
            font-weight: bold;
        }
        .download-link-btn:hover {
            background: #0056b3;
        }
        .progress-area {
            flex: 1;
            min-width: 240px;
        }
        .progress-bg {
            width: 100%;
            height: 10px;
            background: #e9ecef;
            border-radius: 5px;
            overflow: hidden;
            margin-top: 6px;
            display: none;
        }
        .progress-fill {
            height: 100%;
            width: 0%;
            background: #28a745;
            transition: width 0.2s;
        }
        #export-status-text {
            font-size: 13px;
            color: #555;
        }
        /* ポップアップ通知（トースト）コンテナ */
        #toast-container {
            position: fixed;
            top: 20px;
            right: 20px;
            z-index: 9999;
            display: flex;
            flex-direction: column;
            gap: 10px;
            pointer-events: none;
            max-width: 340px;
        }
        .toast-popup {
            background: rgba(24, 30, 42, 0.95);
            color: #fff;
            border-left: 5px solid #00d2ff;
            border-radius: 8px;
            padding: 12px 16px;
            box-shadow: 0 6px 20px rgba(0, 0, 0, 0.28);
            display: flex;
            align-items: center;
            gap: 12px;
            animation: slideInToast 0.25s ease-out;
            pointer-events: auto;
        }
        .toast-icon {
            font-size: 22px;
        }
        .toast-body {
            display: flex;
            flex-direction: column;
            gap: 2px;
        }
        .toast-title {
            font-size: 14px;
            font-weight: bold;
            color: #4dd0e1;
        }
        .toast-sub {
            font-size: 12px;
            color: #cfd8dc;
        }
        @keyframes slideInToast {
            from { transform: translateX(40px); opacity: 0; }
            to { transform: translateX(0); opacity: 1; }
        }
    </style>
</head>
<body>

<!-- ポップアップ顔検出通知エリア -->
<div id="toast-container"></div>

<div class="container">
    <h2 style="text-align:center; margin-bottom:16px;">AI 複数人物 顔認識モザイク ＆ 検出通知</h2>

    <div class="top-grid">
        <!-- STEP 1: 複数人物の顔登録 -->
        <div class="card">
            <h3>
                <span>1. 検出する人物の登録（複数人対応）</span>
                <button class="add-person-btn" onclick="addPerson()">＋ 人物を追加</button>
            </h3>
            <div id="persons-container"></div>
        </div>

        <!-- STEP 2: 対象動画 -->
        <div class="card">
            <h3>2. 処理する動画ファイル</h3>
            <div id="video-drop" class="video-drop-zone">
                <span>MP4 ファイルをドロップ<br>またはクリックして選択</span>
                <input type="file" id="video-input" accept="video/mp4" style="display: none;">
            </div>
            <div style="margin-top: 12px; font-size: 13px; color: #666;" id="video-status">
                動画未選択
            </div>
        </div>
    </div>

    <!-- STEP 3: モード＆詳細設定 -->
    <div class="card settings-card">
        <div class="settings-row">
            <div class="mode-switch">
                <label>
                    <input type="radio" name="action_mode" value="mosaic" checked onchange="updateSettings()">
                    🎭 モザイク処理
                </label>
                <label>
                    <input type="radio" name="action_mode" value="notify" onchange="updateSettings()">
                    🔔 ポップアップ顔検出通知（人物名表示）
                </label>
            </div>

            <div class="control-group">
                <label for="threshold"><strong>一致閾値:</strong> <span id="thresh-val">0.36</span></label>
                <input type="range" id="threshold" min="0.20" max="0.65" step="0.01" value="0.36" oninput="updateSettings()" style="width:95px;">
            </div>

            <div class="control-group">
                <label for="interp-gap"><strong>チラつき補間:</strong> <span id="interp-val">5f</span></label>
                <input type="range" id="interp-gap" min="0" max="50" step="1" value="5" oninput="updateSettings()" style="width:95px;">
            </div>

            <div class="control-group">
                <label><input type="checkbox" id="show-debug" checked onchange="updateSettings()"> 枠・人物名表示</label>
            </div>
        </div>

        <!-- モザイク処理モード時のみ表示される設定 -->
        <div id="mosaic-options-bar">
            <div class="control-group">
                <strong>モザイク対象:</strong>
                <label><input type="radio" name="mosaic_target_mode" value="target_only" checked onchange="updateSettings()"> 登録人物「だけ」</label>
                <label><input type="radio" name="mosaic_target_mode" value="except_target" onchange="updateSettings()"> 登録人物「以外」</label>
            </div>

            <div class="control-group">
                <label for="mosaic-scale"><strong>モザイクの度合い（粗さ）:</strong> <span id="mosaic-scale-val">16</span></label>
                <input type="range" id="mosaic-scale" min="4" max="50" step="1" value="16" oninput="updateSettings()" style="width:130px;">
            </div>
        </div>

        <!-- ポップアップ顔検出通知モード時のステータスバー -->
        <div id="notify-options-bar">
            <span>🔔 登録人物が映ると画面右上に人物名のポップアップ通知を表示します</span>
            <div id="active-detect-badges"></div>
        </div>
    </div>

    <div id="status">人物の顔写真と動画を選択してください</div>

    <div class="player-wrapper">
        <div id="preview-box">
            <img id="stream-img" alt="映像ストリーム">
        </div>
        <div id="player-bar">
            <input type="range" id="seek-bar" min="0" max="100" value="0">
            <div class="player-buttons">
                <div class="btn-group">
                    <button class="player-btn" id="play-pause-btn" onclick="togglePause()">❚❚ 一時停止</button>
                    <button class="player-btn" onclick="skipSeconds(-5)">◀ 5秒戻す</button>
                    <button class="player-btn" onclick="skipSeconds(5)">5秒進む ▶</button>
                </div>
                <div id="time-display">00:00 / 00:00</div>
            </div>
        </div>
    </div>

    <div class="card" id="export-card">
        <div class="progress-area">
            <div id="export-status-text">現在の設定で動画全体を高速書き出しします</div>
            <div class="progress-bg" id="export-progress-bg">
                <div class="progress-fill" id="export-progress-fill"></div>
            </div>
        </div>
        <div style="display:flex; gap:10px; align-items:center;">
            <a id="manual-download-btn" class="download-link-btn" href="/download_video">⬇ MP4を保存</a>
            <button id="export-btn" class="primary-btn" onclick="startExport()">処理済み動画を書き出し</button>
        </div>
    </div>
</div>

<script>
    let personsData = [];
    let totalFrames = 0;
    let videoFps = 30.0;
    let hasVideo = false;
    let isDraggingSeek = false;
    let progressTimer = null;
    let exportTimer = null;

    // ポップアップ通知の重複・連続表示制御用マップ: { personId: lastNotifiedTimestampMs }
    const lastPopupTimes = {};
    const POPUP_COOLDOWN_MS = 4000;

    const seekBar = document.getElementById('seek-bar');
    const playPauseBtn = document.getElementById('play-pause-btn');
    const timeDisplay = document.getElementById('time-display');

    // 初期人物リストの取得・描画
    fetch('/get_persons')
        .then(res => res.json())
        .then(data => {
            personsData = data.persons || [];
            renderPersons();
        });

    function showDetectionPopup(personName, score, frameIdx) {
        const container = document.getElementById('toast-container');
        const timeStr = formatTime(frameIdx / videoFps);
        const toast = document.createElement('div');
        toast.className = 'toast-popup';
        toast.innerHTML = `
            <div class="toast-icon">🔔</div>
            <div class="toast-body">
                <div class="toast-title">${escapeHtml(personName)} を検出しました</div>
                <div class="toast-sub">再生位置: ${timeStr} ／ 一致度: ${(score * 100).toFixed(0)}%</div>
            </div>
        `;
        container.appendChild(toast);

        // 3.5秒後に自動でフェードアウト削除
        setTimeout(() => {
            toast.style.transition = 'opacity 0.3s, transform 0.3s';
            toast.style.opacity = '0';
            toast.style.transform = 'translateX(30px)';
            setTimeout(() => toast.remove(), 300);
        }, 3500);
    }

    function escapeHtml(str) {
        return String(str)
            .replace(/&/g, '&amp;')
            .replace(/</g, '&lt;')
            .replace(/>/g, '&gt;')
            .replace(/"/g, '&quot;');
    }

    function addPerson() {
        fetch('/add_person', { method: 'POST' })
            .then(res => res.json())
            .then(data => {
                if (data.status === 'ok') {
                    personsData = data.persons;
                    renderPersons();
                }
            });
    }

    function removePerson(personId) {
        fetch('/remove_person', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ person_id: personId })
        })
        .then(res => res.json())
        .then(data => {
            if (data.status === 'ok') {
                personsData = data.persons;
                renderPersons();
            }
        });
    }

    function updatePersonName(personId, newName) {
        fetch('/update_person_name', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ person_id: personId, name: newName })
        });
    }

    function renderPersons(statusMessages = {}) {
        const container = document.getElementById('persons-container');
        container.innerHTML = '';

        personsData.forEach((person) => {
            const hasEntries = person.entries && person.entries.length > 0;
            const card = document.createElement('div');
            card.className = 'person-card';

            const chipsHtml = (person.entries || []).map(entry => `
                <div class="target-chip ${entry.source === 'boost' ? 'boost' : ''}">
                    <img src="data:image/jpeg;base64,${entry.crop_b64}" alt="顔">
                    <span class="badge">${entry.source === 'boost' ? '動画抽出' : '登録画像'}</span>
                    <button class="del-btn" title="削除" onclick="removeTargetEntry('${person.id}', '${entry.id}')">×</button>
                </div>
            `).join('');

            const msgText = statusMessages[person.id] ||
                (hasEntries ? `登録済み顔パターン: ${person.entries.length}枚（左右反転含む ${person.entries.length * 2}通り）` : '顔画像をドロップまたは選択してください');

            card.innerHTML = `
                <div class="person-header">
                    <div style="display:flex; align-items:center; gap:6px;">
                        <span style="font-size:12px; color:#555;">人物名:</span>
                        <input type="text" class="person-name-input" value="${escapeHtml(person.name)}"
                               placeholder="名前を入力" onchange="updatePersonName('${person.id}', this.value)">
                    </div>
                    <button class="sub-btn danger" onclick="removePerson('${person.id}')">人物を削除</button>
                </div>

                <div class="person-drop-zone" id="drop-${person.id}">
                    <span>顔写真 (JPG / PNG) をここにドロップ（またはクリックして選択・複数枚可）</span>
                    <input type="file" id="input-${person.id}" accept="image/*" multiple style="display:none;">
                </div>

                <div class="target-list">${chipsHtml}</div>
                <div class="person-msg" id="msg-${person.id}">${msgText}</div>

                <div class="person-actions">
                    <button class="sub-btn accent" id="boost-${person.id}"
                            onclick="boostPersonFromVideo('${person.id}')"
                            ${!(hasVideo && hasEntries) ? 'disabled' : ''}>
                        ⚡ 動画内の本人顔で自動補強
                    </button>
                    <button class="sub-btn" id="cap-${person.id}"
                            onclick="captureCurrentFrameForPerson('${person.id}')"
                            ${!hasVideo ? 'disabled' : ''}>
                        ＋ 今映っている顔を追加
                    </button>
                </div>
            `;
            container.appendChild(card);

            // ドロップ＆ファイル選択イベントの紐付け
            const zone = document.getElementById(`drop-${person.id}`);
            const input = document.getElementById(`input-${person.id}`);
            zone.addEventListener('click', () => input.click());
            zone.addEventListener('dragover', (e) => { e.preventDefault(); zone.classList.add('dragover'); });
            zone.addEventListener('dragleave', () => zone.classList.remove('dragover'));
            zone.addEventListener('drop', (e) => {
                e.preventDefault();
                zone.classList.remove('dragover');
                if (e.dataTransfer.files.length > 0) {
                    uploadPersonFaces(person.id, e.dataTransfer.files);
                }
            });
            input.addEventListener('change', () => {
                if (input.files.length > 0) {
                    uploadPersonFaces(person.id, input.files);
                }
            });
        });
    }

    function uploadPersonFaces(personId, files) {
        const msgEl = document.getElementById(`msg-${personId}`);
        if (msgEl) {
            msgEl.textContent = '顔を解析・特徴量抽出中...';
            msgEl.style.color = '#0056b3';
        }

        const formData = new FormData();
        formData.append('person_id', personId);
        for (let i = 0; i < files.length; i++) {
            formData.append('face_image', files[i]);
        }

        fetch('/upload_target', { method: 'POST', body: formData })
            .then(res => res.json())
            .then(data => {
                if (data.persons) {
                    personsData = data.persons;
                    renderPersons({ [personId]: data.message });
                }
            });
    }

    function removeTargetEntry(personId, entryId) {
        fetch('/remove_target', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ person_id: personId, entry_id: entryId })
        })
        .then(res => res.json())
        .then(data => {
            if (data.persons) {
                personsData = data.persons;
                renderPersons();
            }
        });
    }

    function boostPersonFromVideo(personId) {
        const btn = document.getElementById(`boost-${personId}`);
        const msgEl = document.getElementById(`msg-${personId}`);
        if (btn) btn.disabled = true;
        if (msgEl) {
            msgEl.textContent = '動画内から本人顔をサンプリング抽出中...';
            msgEl.style.color = '#856404';
        }

        fetch('/boost_from_video', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ person_id: personId })
        })
        .then(res => res.json())
        .then(data => {
            if (data.persons) {
                personsData = data.persons;
                renderPersons({ [personId]: data.message });
            } else if (msgEl) {
                if (btn) btn.disabled = false;
                msgEl.textContent = data.message || '一致候補が見つかりませんでした';
                msgEl.style.color = '#dc3545';
            }
        });
    }

    function captureCurrentFrameForPerson(personId) {
        const msgEl = document.getElementById(`msg-${personId}`);
        fetch('/add_current_frame_face', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ person_id: personId })
        })
        .then(res => res.json())
        .then(data => {
            if (data.persons) {
                personsData = data.persons;
                renderPersons({ [personId]: data.message });
            } else if (msgEl) {
                msgEl.textContent = data.message || '現在のフレームから顔を検出できませんでした';
                msgEl.style.color = '#dc3545';
            }
        });
    }

    function formatTime(seconds) {
        if (!isFinite(seconds) || seconds < 0) seconds = 0;
        const m = Math.floor(seconds / 60);
        const s = Math.floor(seconds % 60);
        return String(m).padStart(2, '0') + ':' + String(s).padStart(2, '0');
    }

    function updateTimeText(currentFrame) {
        const curSec = currentFrame / videoFps;
        const totSec = totalFrames / videoFps;
        timeDisplay.textContent = `${formatTime(curSec)} / ${formatTime(totSec)}`;
    }

    seekBar.addEventListener('mousedown', () => { isDraggingSeek = true; });
    seekBar.addEventListener('touchstart', () => { isDraggingSeek = true; });

    seekBar.addEventListener('input', () => {
        const targetFrame = parseInt(seekBar.value, 10);
        updateTimeText(targetFrame);
    });

    seekBar.addEventListener('change', () => {
        const targetFrame = parseInt(seekBar.value, 10);
        sendControl({ action: 'seek', frame: targetFrame }).then(() => {
            isDraggingSeek = false;
        });
    });

    function togglePause() {
        sendControl({ action: 'toggle_pause' });
    }

    function skipSeconds(sec) {
        const current = parseInt(seekBar.value, 10);
        const delta = Math.round(sec * videoFps);
        const target = Math.max(0, Math.min(totalFrames - 1, current + delta));
        seekBar.value = target;
        updateTimeText(target);
        sendControl({ action: 'seek', frame: target });
    }

    function sendControl(payload) {
        return fetch('/control', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(payload)
        })
        .then(res => res.json())
        .then(data => {
            if (typeof data.paused === 'boolean') {
                playPauseBtn.textContent = data.paused ? '▶ 再生' : '❚❚ 一時停止';
            }
        });
    }

    function startProgressPolling() {
        if (progressTimer) clearInterval(progressTimer);
        progressTimer = setInterval(() => {
            fetch('/progress')
                .then(res => res.json())
                .then(data => {
                    if (!isDraggingSeek && totalFrames > 0) {
                        seekBar.value = data.current_frame;
                        updateTimeText(data.current_frame);
                    }
                    playPauseBtn.textContent = data.paused ? '▶ 再生' : '❚❚ 一時停止';

                    // ポップアップ顔検出通知処理
                    const actionMode = document.querySelector('input[name="action_mode"]:checked').value;
                    const badgesWrap = document.getElementById('active-detect-badges');
                    const detections = data.detected_persons || [];

                    if (actionMode === 'notify') {
                        badgesWrap.innerHTML = detections.map(d =>
                            `<span class="detect-pill">検出中: ${escapeHtml(d.name)}</span>`
                        ).join('');

                        if (!data.paused && detections.length > 0) {
                            const now = Date.now();
                            detections.forEach(det => {
                                const lastTime = lastPopupTimes[det.id] || 0;
                                if (now - lastTime > POPUP_COOLDOWN_MS) {
                                    lastPopupTimes[det.id] = now;
                                    showDetectionPopup(det.name, det.score, data.current_frame);
                                }
                            });
                        }
                    }
                })
                .catch(() => {});
        }, 250);
    }

    // 動画ドロップ＆選択セットアップ
    const videoDrop = document.getElementById('video-drop');
    const videoInput = document.getElementById('video-input');
    videoDrop.addEventListener('click', () => videoInput.click());
    videoDrop.addEventListener('dragover', (e) => { e.preventDefault(); videoDrop.classList.add('dragover'); });
    videoDrop.addEventListener('dragleave', () => videoDrop.classList.remove('dragover'));
    videoDrop.addEventListener('drop', (e) => {
        e.preventDefault();
        videoDrop.classList.remove('dragover');
        if (e.dataTransfer.files.length > 0) uploadVideoFile(e.dataTransfer.files[0]);
    });
    videoInput.addEventListener('change', () => {
        if (videoInput.files.length > 0) uploadVideoFile(videoInput.files[0]);
    });

    function uploadVideoFile(file) {
        if (file.type !== 'video/mp4') {
            alert('MP4形式の動画ファイルを選択してください。');
            return;
        }
        const statusText = document.getElementById('status');
        const videoStatus = document.getElementById('video-status');
        const streamImg = document.getElementById('stream-img');
        const playerBar = document.getElementById('player-bar');
        const exportCard = document.getElementById('export-card');

        statusText.textContent = '動画を読み込み中...';
        const formData = new FormData();
        formData.append('video', file);

        fetch('/upload_video', { method: 'POST', body: formData })
            .then(res => res.json())
            .then(data => {
                if (data.status === 'ok') {
                    hasVideo = true;
                    totalFrames = data.total_frames;
                    videoFps = data.fps;
                    renderPersons(); // 動画読み込み完了後に人物カードの補強ボタンを有効化

                    seekBar.max = Math.max(0, totalFrames - 1);
                    seekBar.value = 0;
                    updateTimeText(0);

                    videoStatus.textContent = `選択中: ${file.name} (${formatTime(totalFrames / videoFps)})`;
                    statusText.textContent = '再生・リアルタイム認識処理中...';

                    streamImg.src = '/video_feed?' + new Date().getTime();
                    streamImg.style.display = 'block';
                    playerBar.style.display = 'flex';
                    exportCard.style.display = 'flex';

                    document.getElementById('manual-download-btn').style.display = 'none';
                    document.getElementById('export-progress-bg').style.display = 'none';
                    document.getElementById('export-status-text').textContent = '現在の設定で動画全体を高速書き出しします';

                    startProgressPolling();
                }
            });
    }

    function updateSettings() {
        const actionMode = document.querySelector('input[name="action_mode"]:checked').value;
        const mosaicTargetMode = document.querySelector('input[name="mosaic_target_mode"]:checked').value;
        const mosaicScale = parseInt(document.getElementById('mosaic-scale').value, 10);
        const threshold = parseFloat(document.getElementById('threshold').value);
        const interpGap = parseInt(document.getElementById('interp-gap').value, 10);
        const showDebug = document.getElementById('show-debug').checked;

        document.getElementById('thresh-val').textContent = threshold.toFixed(2);
        document.getElementById('interp-val').textContent = interpGap === 0 ? 'OFF' : `${interpGap}f`;
        document.getElementById('mosaic-scale-val').textContent = mosaicScale;

        // モードに応じてオプションバーを切り替え
        document.getElementById('mosaic-options-bar').style.display = (actionMode === 'mosaic') ? 'flex' : 'none';
        document.getElementById('notify-options-bar').style.display = (actionMode === 'notify') ? 'flex' : 'none';

        fetch('/update_settings', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                action_mode: actionMode,
                mosaic_target_mode: mosaicTargetMode,
                mosaic_scale: mosaicScale,
                threshold: threshold,
                interp_gap: interpGap,
                show_debug: showDebug
            })
        });
    }

    function startExport() {
        const exportBtn = document.getElementById('export-btn');
        const progressBg = document.getElementById('export-progress-bg');
        const progressFill = document.getElementById('export-progress-fill');
        const statusText = document.getElementById('export-status-text');
        const downloadBtn = document.getElementById('manual-download-btn');

        exportBtn.disabled = true;
        downloadBtn.style.display = 'none';
        progressBg.style.display = 'block';
        progressFill.style.width = '0%';
        statusText.textContent = '書き出し準備中（プレビューを一時停止しました）...';

        fetch('/export_video', { method: 'POST' })
            .then(res => res.json())
            .then(data => {
                if (data.status !== 'ok') {
                    exportBtn.disabled = false;
                    statusText.textContent = data.message || 'エラーが発生しました';
                    return;
                }

                if (exportTimer) clearInterval(exportTimer);
                exportTimer = setInterval(() => {
                    fetch('/export_status')
                        .then(res => res.json())
                        .then(st => {
                            progressFill.style.width = st.progress + '%';
                            statusText.textContent = st.message;

                            if (st.status === 'done') {
                                clearInterval(exportTimer);
                                exportBtn.disabled = false;
                                downloadBtn.style.display = 'inline-block';
                                window.location.href = '/download_video?t=' + new Date().getTime();
                            } else if (st.status === 'error') {
                                clearInterval(exportTimer);
                                exportBtn.disabled = false;
                            }
                        });
                }, 250);
            })
            .catch(() => {
                exportBtn.disabled = false;
                statusText.textContent = '通信エラーが発生しました';
            });
    }
</script>
</body>
</html>
"""

def serialize_persons():
    with state_lock:
        return [
            {
                "id": p["id"],
                "name": p["name"],
                "entries": [
                    {
                        "id": e["id"],
                        "crop_b64": e["crop_b64"],
                        "source": e["source"]
                    }
                    for e in p["entries"]
                ]
            }
            for p in state["persons"]
        ]

@app.route('/')
def index():
    return render_template_string(HTML_TEMPLATE)

@app.route('/get_persons')
def get_persons():
    return jsonify({'status': 'ok', 'persons': serialize_persons()})

@app.route('/add_person', methods=['POST'])
def add_person():
    with state_lock:
        next_idx = len(state["persons"]) + 1
        state["persons"].append(create_default_person(next_idx))
    return jsonify({'status': 'ok', 'persons': serialize_persons()})

@app.route('/remove_person', methods=['POST'])
def remove_person():
    data = request.get_json()
    person_id = data.get('person_id')
    with state_lock:
        state["persons"] = [p for p in state["persons"] if p["id"] != person_id]
        if len(state["persons"]) == 0:
            state["persons"].append(create_default_person(1))
        state["settings_dirty"] = True
    return jsonify({'status': 'ok', 'persons': serialize_persons()})

@app.route('/update_person_name', methods=['POST'])
def update_person_name():
    data = request.get_json()
    person_id = data.get('person_id')
    new_name = str(data.get('name', '')).strip() or "名称未設定"
    with state_lock:
        person = find_person(person_id)
        if person:
            person["name"] = new_name
            state["settings_dirty"] = True
    return jsonify({'status': 'ok'})

@app.route('/upload_target', methods=['POST'])
def upload_target():
    person_id = request.form.get('person_id')
    files = request.files.getlist('face_image')
    if not files or not person_id:
        return jsonify({'status': 'error', 'message': 'パラメータが不足しています'}), 400

    with state_lock:
        person = find_person(person_id)
    if not person:
        return jsonify({'status': 'error', 'message': '対象の人物が見つかりません'}), 404

    added_count = 0
    for file in files:
        file_bytes = np.frombuffer(file.read(), np.uint8)
        img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
        if img is None:
            continue

        h, w = img.shape[:2]
        max_side = max(w, h)
        if max_side > 1280:
            scale = 1280 / max_side
            img = cv2.resize(img, (int(w * scale), int(h * scale)))
            h, w = img.shape[:2]

        detector = cv2.FaceDetectorYN.create(
            model=DET_MODEL_PATH,
            config="",
            input_size=(w, h),
            score_threshold=0.5,
            nms_threshold=0.3,
            top_k=5000
        )
        _, faces = detector.detect(img)
        if faces is None or len(faces) == 0:
            continue

        target_face = max(faces, key=lambda f: f[2] * f[3])
        aligned_face = recognizer.alignCrop(img, target_face)

        feat_orig = recognizer.feature(aligned_face)
        aligned_flipped = cv2.flip(aligned_face, 1)
        feat_flip = recognizer.feature(aligned_flipped)

        _, buf = cv2.imencode('.jpg', aligned_face)
        face_b64 = base64.b64encode(buf).decode('utf-8')

        with state_lock:
            person["entries"].append({
                "id": str(uuid.uuid4()),
                "crop_b64": face_b64,
                "source": "photo",
                "vectors": [feat_orig, feat_flip]
            })
        added_count += 1

    if added_count == 0:
        return jsonify({
            'status': 'error',
            'message': '画像から顔を検出できませんでした。別の写真をお試しください。',
            'persons': serialize_persons()
        })

    state["settings_dirty"] = True
    total_cnt = len(person["entries"])
    return jsonify({
        'status': 'ok',
        'message': f'{added_count}枚追加しました（計{total_cnt}枚 / 左右反転含む{total_cnt * 2}パターン）',
        'persons': serialize_persons()
    })

@app.route('/boost_from_video', methods=['POST'])
def boost_from_video():
    data = request.get_json() or {}
    person_id = data.get('person_id')
    video_path = state["video_path"]

    with state_lock:
        person = find_person(person_id)
        base_vectors = get_person_vectors(person) if person else []

    if not video_path or not os.path.exists(video_path):
        return jsonify({'status': 'error', 'message': '先に動画を読み込んでください'})
    if not person or not base_vectors:
        return jsonify({'status': 'error', 'message': '先にこの人物の顔写真を1枚以上登録してください'})

    cap = cv2.VideoCapture(video_path)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    detector = cv2.FaceDetectorYN.create(
        model=DET_MODEL_PATH,
        config="",
        input_size=(width, height),
        score_threshold=0.6,
        nms_threshold=0.3,
        top_k=5000
    )

    sample_count = min(45, max(1, total))
    step = max(1, total // sample_count)
    sample_indices = list(range(0, total, step))[:sample_count]

    candidates = []

    for idx in sample_indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if not ret:
            continue

        _, faces = detector.detect(frame)
        if faces is None:
            continue

        best_face = None
        best_score = -1.0
        best_feat = None

        for face in faces:
            aligned = recognizer.alignCrop(frame, face)
            feat = recognizer.feature(aligned)
            score = compute_max_similarity(feat, base_vectors)
            if score > best_score:
                best_score = score
                best_face = aligned
                best_feat = feat

        if best_face is not None and best_score >= 0.25:
            candidates.append((best_score, idx, best_face, best_feat))

    cap.release()

    if not candidates:
        return jsonify({
            'status': 'error',
            'message': '動画内から一致候補が見つかりませんでした。別の角度の顔写真を追加してお試しください。'
        })

    candidates.sort(key=lambda x: x[0], reverse=True)
    selected = []
    min_frame_gap = max(1, total // 15)

    for cand in candidates:
        if all(abs(cand[1] - s[1]) >= min_frame_gap for s in selected):
            selected.append(cand)
        if len(selected) >= 5:
            break

    with state_lock:
        for score, idx, aligned_crop, feat in selected:
            feat_flip = recognizer.feature(cv2.flip(aligned_crop, 1))
            _, buf = cv2.imencode('.jpg', aligned_crop)
            face_b64 = base64.b64encode(buf).decode('utf-8')
            person["entries"].append({
                "id": str(uuid.uuid4()),
                "crop_b64": face_b64,
                "source": "boost",
                "vectors": [feat, feat_flip]
            })

    state["settings_dirty"] = True
    top_score = selected[0][0]
    return jsonify({
        'status': 'ok',
        'message': f'動画内から {len(selected)}パターンを自動追加しました（最高一致度: {top_score:.2f}）',
        'persons': serialize_persons()
    })

@app.route('/add_current_frame_face', methods=['POST'])
def add_current_frame_face():
    data = request.get_json() or {}
    person_id = data.get('person_id')
    video_path = state["video_path"]

    with state_lock:
        person = find_person(person_id)
        base_vectors = get_person_vectors(person) if person else []

    if not person:
        return jsonify({'status': 'error', 'message': '対象の人物が見つかりません'})
    if not video_path or not os.path.exists(video_path):
        return jsonify({'status': 'error', 'message': '動画が選択されていません'})

    cap = cv2.VideoCapture(video_path)
    cap.set(cv2.CAP_PROP_POS_FRAMES, state["current_frame"])
    ret, frame = cap.read()
    cap.release()
    if not ret:
        return jsonify({'status': 'error', 'message': 'フレームを読み込めませんでした'})

    h, w = frame.shape[:2]
    detector = cv2.FaceDetectorYN.create(
        model=DET_MODEL_PATH,
        config="",
        input_size=(w, h),
        score_threshold=0.6,
        nms_threshold=0.3,
        top_k=5000
    )
    _, faces = detector.detect(frame)
    if faces is None or len(faces) == 0:
        return jsonify({'status': 'error', 'message': 'このフレームに顔が検出されませんでした'})

    if base_vectors:
        chosen_face = max(
            faces,
            key=lambda f: compute_max_similarity(recognizer.feature(recognizer.alignCrop(frame, f)), base_vectors)
        )
    else:
        chosen_face = max(faces, key=lambda f: f[2] * f[3])

    aligned_face = recognizer.alignCrop(frame, chosen_face)
    feat_orig = recognizer.feature(aligned_face)
    feat_flip = recognizer.feature(cv2.flip(aligned_face, 1))

    _, buf = cv2.imencode('.jpg', aligned_face)
    face_b64 = base64.b64encode(buf).decode('utf-8')

    with state_lock:
        person["entries"].append({
            "id": str(uuid.uuid4()),
            "crop_b64": face_b64,
            "source": "boost",
            "vectors": [feat_orig, feat_flip]
        })
    state["settings_dirty"] = True

    return jsonify({
        'status': 'ok',
        'message': f'フレーム #{state["current_frame"]} の顔を「{person["name"]}」に追加しました',
        'persons': serialize_persons()
    })

@app.route('/remove_target', methods=['POST'])
def remove_target():
    data = request.get_json()
    person_id = data.get('person_id')
    entry_id = data.get('entry_id')
    with state_lock:
        person = find_person(person_id)
        if person:
            person["entries"] = [e for e in person["entries"] if e["id"] != entry_id]
        state["settings_dirty"] = True
    return jsonify({'status': 'ok', 'persons': serialize_persons()})

@app.route('/upload_video', methods=['POST'])
def upload_video():
    if 'video' not in request.files:
        return jsonify({'status': 'error'}), 400

    file = request.files['video']
    base_name = os.path.splitext(file.filename)[0] if file.filename else "video"
    saved_path = os.path.join(UPLOAD_FOLDER, 'temp_video.mp4')
    file.save(saved_path)

    cap = cv2.VideoCapture(saved_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0:
        fps = 30.0
    cap.release()

    with cache_lock:
        state["feature_cache"] = {}

    state["stream_id"] = str(uuid.uuid4())
    state["video_path"] = saved_path
    state["original_name"] = base_name
    state["total_frames"] = total_frames
    state["fps"] = fps
    state["current_frame"] = 0
    state["seek_frame"] = None
    state["paused"] = False
    state["latest_detections"] = []
    state["export_status"] = "idle"
    state["export_progress"] = 0
    state["output_path"] = None

    return jsonify({
        'status': 'ok',
        'total_frames': total_frames,
        'fps': fps
    })

@app.route('/control', methods=['POST'])
def control():
    data = request.get_json()
    action = data.get('action')

    if action == 'seek':
        target_frame = int(data.get('frame', 0))
        state['seek_frame'] = max(0, min(state['total_frames'] - 1, target_frame))
    elif action == 'toggle_pause':
        if state['paused'] and state['current_frame'] >= state['total_frames'] - 1:
            state['seek_frame'] = 0
        state['paused'] = not state['paused']

    return jsonify({
        'status': 'ok',
        'paused': state['paused'],
        'current_frame': state['current_frame']
    })

@app.route('/progress')
def progress():
    return jsonify({
        'current_frame': state['current_frame'],
        'total_frames': state['total_frames'],
        'paused': state['paused'],
        'detected_persons': state['latest_detections']
    })

@app.route('/update_settings', methods=['POST'])
def update_settings():
    data = request.get_json()
    if 'action_mode' in data and data['action_mode'] in ('mosaic', 'notify'):
        state['action_mode'] = data['action_mode']
    if 'mosaic_target_mode' in data and data['mosaic_target_mode'] in ('target_only', 'except_target'):
        state['mosaic_target_mode'] = data['mosaic_target_mode']
    if 'mosaic_scale' in data:
        state['mosaic_scale'] = max(4, min(50, int(data['mosaic_scale'])))
    if 'threshold' in data:
        state['threshold'] = float(data['threshold'])
    if 'interp_gap' in data:
        state['interp_gap'] = max(0, min(50, int(data['interp_gap'])))
    if 'show_debug' in data:
        state['show_debug'] = bool(data['show_debug'])
    state['settings_dirty'] = True
    return jsonify({'status': 'ok'})

def apply_mosaic(frame, x, y, w, h, scale=16):
    x1 = max(0, x)
    y1 = max(0, y)
    x2 = min(frame.shape[1], x + w)
    y2 = min(frame.shape[0], y + h)
    roi_w, roi_h = x2 - x1, y2 - y1

    if roi_w > 0 and roi_h > 0:
        face_roi = frame[y1:y2, x1:x2]
        small = cv2.resize(face_roi, (max(1, roi_w // scale), max(1, roi_h // scale)), interpolation=cv2.INTER_LINEAR)
        mosaic = cv2.resize(small, (roi_w, roi_h), interpolation=cv2.INTER_NEAREST)
        frame[y1:y2, x1:x2] = mosaic

def detect_and_extract_features(frame, detector):
    _, faces = detector.detect(frame)
    frame_faces = []
    if faces is not None:
        for face in faces:
            x, y, w, h = map(int, face[:4])
            aligned = recognizer.alignCrop(frame, face)
            feat = recognizer.feature(aligned)
            frame_faces.append((x, y, w, h, feat))
    return frame_faces

def draw_boxes_and_labels(frame, box_labels):
    """
    バウンディングボックスと日本語人物名ラベルを描画
    box_labels: [(x, y, w, h, bgr_color, text), ...]
    """
    if not box_labels:
        return frame

    # まずOpenCVで枠線を描画
    for x, y, w, h, color, _ in box_labels:
        cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)

    # 日本語フォントが使える場合はPillowで綺麗に日本語テキスト描画
    font = get_jp_font(18)
    if HAS_PIL and font is not None:
        pil_img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        draw = ImageDraw.Draw(pil_img)
        for x, y, w, h, color, text in box_labels:
            rgb_color = (color[2], color[1], color[0])
            bbox = draw.textbbox((0, 0), text, font=font)
            tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
            tx = max(0, x)
            ty = max(0, y - th - 8)
            draw.rectangle([tx, ty, tx + tw + 10, ty + th + 6], fill=rgb_color)
            draw.text((tx + 5, ty + 2), text, font=font, fill=(255, 255, 255))
        return cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
    else:
        for x, y, w, h, color, text in box_labels:
            safe_text = text.encode('ascii', 'ignore').decode('ascii') or "DETECTED"
            cv2.putText(frame, safe_text, (x, max(20, y - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)
        return frame

def render_items_on_frame(frame, render_items, has_any_person, action_mode, mosaic_scale, show_debug):
    """確定したモザイク＆人物名ラベルリストをフレームに描画"""
    box_labels = []

    for x, y, w, h, is_target, score, should_mosaic, is_interp, p_id, p_name in render_items:
        if action_mode == "mosaic" and should_mosaic:
            apply_mosaic(frame, x, y, w, h, scale=mosaic_scale)

        if show_debug:
            if action_mode == "notify":
                # 通知モード時は登録人物のみ枠と名前を表示
                if is_target and p_name:
                    color = (255, 200, 0) if is_interp else (0, 180, 255)
                    label = f"{p_name}" if is_interp else f"{p_name} ({score:.2f})"
                    box_labels.append((x, y, w, h, color, label))
            else:
                # モザイクモード時
                if has_any_person or is_interp:
                    if is_interp:
                        color = (255, 220, 0)
                        label = f"{p_name} (補間)" if p_name else "補間"
                    elif is_target:
                        color = (0, 0, 255)
                        label = f"{p_name} ({score:.2f})"
                    else:
                        color = (0, 200, 80)
                        label = f"未登録 ({score:.2f})"
                    box_labels.append((x, y, w, h, color, label))

    if box_labels:
        frame = draw_boxes_and_labels(frame, box_labels)
    return frame

def export_video_worker(video_path, persons_snapshot, action_mode, mosaic_target_mode, mosaic_scale, threshold, interp_gap):
    cap = None
    ffmpeg_proc = None
    writer = None
    final_out_path = os.path.join(OUTPUT_FOLDER, 'final_mosaic.mp4')

    try:
        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        if fps <= 0:
            fps = 30.0
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()
        cap = None

        with cache_lock:
            cached_keys = set(state["feature_cache"].keys())

        missing_frames = [i for i in range(total) if i not in cached_keys]
        already_cached = total - len(missing_frames)

        if missing_frames:
            cpu_count = os.cpu_count() or 4
            num_workers = min(cpu_count, 8, max(1, len(missing_frames) // 30))
            num_chunks = min(len(missing_frames), num_workers * 4)
            chunk_size = max(1, (len(missing_frames) + num_chunks - 1) // num_chunks)
            chunks = [missing_frames[i:i + chunk_size] for i in range(0, len(missing_frames), chunk_size)]

            completed_frames = already_cached
            state["export_message"] = f"並列AI解析中 ({num_workers}プロセス)... {int(completed_frames / max(1, total) * 70)}%"

            with ProcessPoolExecutor(max_workers=num_workers) as executor:
                futures = [
                    executor.submit(analyze_video_chunk, video_path, ch, DET_MODEL_PATH, REC_MODEL_PATH)
                    for ch in chunks
                ]
                for future in as_completed(futures):
                    chunk_result = future.result()
                    with cache_lock:
                        state["feature_cache"].update(chunk_result)
                    completed_frames += len(chunk_result)
                    pct = min(70, int((completed_frames / max(1, total)) * 70))
                    state["export_progress"] = pct
                    state["export_message"] = f"並列AI解析中 ({num_workers}プロセス)... {pct}% ({completed_frames}/{total})"
        else:
            state["export_progress"] = 70
            state["export_message"] = "キャッシュ済みAI解析データを使用（推論スキップ）..."

        state["export_progress"] = 74
        state["export_message"] = "複数人物照合 ＆ 前後フレーム座標補間を計算中..."
        with cache_lock:
            local_cache = dict(state["feature_cache"])

        evaluated_map = {
            idx: evaluate_frame_faces(
                local_cache.get(idx, []), persons_snapshot, action_mode, mosaic_target_mode, threshold
            )
            for idx in range(total)
        }
        has_any_person = len(persons_snapshot) > 0
        # 通知モードでの書き出し時は映像上に人物名ラベルを描画、モザイクモード時はクリーン映像出力
        export_show_debug = (action_mode == "notify")

        cap = cv2.VideoCapture(video_path)
        ffmpeg_bin = shutil.which("ffmpeg")

        if ffmpeg_bin:
            cmd = [
                ffmpeg_bin, "-y",
                "-f", "rawvideo",
                "-vcodec", "rawvideo",
                "-s", f"{width}x{height}",
                "-pix_fmt", "bgr24",
                "-r", str(fps),
                "-i", "-",
                "-i", video_path,
                "-map", "0:v:0",
                "-map", "1:a?",
                "-c:v", "libx264",
                "-crf", "18",
                "-preset", "veryfast",
                "-pix_fmt", "yuv420p",
                "-c:a", "aac",
                "-shortest",
                final_out_path
            ]
            ffmpeg_proc = subprocess.Popen(
                cmd,
                stdin=subprocess.PIPE,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL
            )
        else:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            writer = cv2.VideoWriter(final_out_path, fourcc, fps, (width, height))

        frame_idx = 0
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            render_items = build_interpolated_items(frame_idx, evaluated_map, interp_gap, action_mode)
            processed_frame = render_items_on_frame(
                frame, render_items, has_any_person, action_mode, mosaic_scale, show_debug=export_show_debug
            )

            if ffmpeg_proc is not None:
                ffmpeg_proc.stdin.write(processed_frame.tobytes())
            elif writer is not None:
                writer.write(processed_frame)

            frame_idx += 1
            if total > 0 and frame_idx % 10 == 0:
                pct = 75 + min(24, int((frame_idx / total) * 24))
                state["export_progress"] = pct
                state["export_message"] = f"高速パイプエンコード中... {pct}% ({frame_idx}/{total})"

        if ffmpeg_proc is not None:
            ffmpeg_proc.stdin.close()
            ffmpeg_proc.wait()
            ffmpeg_proc = None
        if writer is not None:
            writer.release()
            writer = None

        state["output_path"] = final_out_path
        state["export_progress"] = 100
        state["export_status"] = "done"
        state["export_message"] = "書き出し完了！ダウンロードを開始しました。"

    except Exception as e:
        state["export_status"] = "error"
        state["export_message"] = f"書き出しエラー: {str(e)}"
    finally:
        if ffmpeg_proc is not None:
            try:
                ffmpeg_proc.stdin.close()
                ffmpeg_proc.kill()
            except Exception:
                pass
        if writer is not None:
            writer.release()
        if cap is not None:
            cap.release()

@app.route('/export_video', methods=['POST'])
def export_video():
    if not state["video_path"] or not os.path.exists(state["video_path"]):
        return jsonify({'status': 'error', 'message': '動画が選択されていません'}), 400
    if state["export_status"] == "processing":
        return jsonify({'status': 'error', 'message': '現在書き出し処理中です'}), 400

    state["paused"] = True
    state["export_status"] = "processing"
    state["export_progress"] = 0
    state["export_message"] = "並列処理の準備中..."
    state["output_path"] = None

    persons_snapshot = get_persons_snapshot()

    t = threading.Thread(
        target=export_video_worker,
        args=(
            state["video_path"],
            persons_snapshot,
            state["action_mode"],
            state["mosaic_target_mode"],
            state["mosaic_scale"],
            state["threshold"],
            state["interp_gap"]
        ),
        daemon=True
    )
    t.start()

    return jsonify({'status': 'ok'})

@app.route('/export_status')
def export_status():
    return jsonify({
        'status': state["export_status"],
        'progress': state["export_progress"],
        'message': state["export_message"]
    })

@app.route('/download_video')
def download_video():
    out_path = state.get("output_path")
    if not out_path or not os.path.exists(out_path):
        return "ファイルが見つかりません", 404

    suffix = "notify" if state["action_mode"] == "notify" else "mosaic"
    download_filename = f"{state['original_name']}_{suffix}.mp4"
    return send_file(
        out_path,
        as_attachment=True,
        download_name=download_filename,
        mimetype='video/mp4'
    )

def generate_frames():
    video_path = state["video_path"]
    my_stream_id = state["stream_id"]
    if not video_path or not os.path.exists(video_path):
        return

    cap = cv2.VideoCapture(video_path)
    try:
        fps = state["fps"]
        total_frames = state["total_frames"]
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        frame_delay = 1.0 / fps if fps > 0 else 0.03

        detector = cv2.FaceDetectorYN.create(
            model=DET_MODEL_PATH,
            config="",
            input_size=(width, height),
            score_threshold=0.6,
            nms_threshold=0.3,
            top_k=5000
        )

        frame_ring_buffer = {}
        play_idx = 0
        cap_read_idx = 0

        def ensure_frame_buffered(idx):
            nonlocal cap_read_idx
            if idx < 0 or idx >= total_frames:
                return False
            if idx not in frame_ring_buffer:
                if cap_read_idx != idx:
                    cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
                    cap_read_idx = idx
                ret, f_img = cap.read()
                cap_read_idx += 1
                if not ret:
                    return False
                frame_ring_buffer[idx] = f_img

            with cache_lock:
                has_feat = (idx in state["feature_cache"])
            if not has_feat:
                feats = detect_and_extract_features(frame_ring_buffer[idx], detector)
                with cache_lock:
                    state["feature_cache"][idx] = feats
            return True

        while cap.isOpened() and state["stream_id"] == my_stream_id:
            start_time = time.time()
            force_render = False

            if state["seek_frame"] is not None:
                play_idx = state["seek_frame"]
                state["seek_frame"] = None
                state["current_frame"] = play_idx
                frame_ring_buffer.clear()
                force_render = True

            if state["paused"] and not force_render:
                if state["settings_dirty"]:
                    state["settings_dirty"] = False
                    force_render = True
                else:
                    time.sleep(0.05)
                    continue

            state["settings_dirty"] = False

            if play_idx >= total_frames:
                state["paused"] = True
                state["current_frame"] = max(0, total_frames - 1)
                time.sleep(0.05)
                continue

            if not ensure_frame_buffered(play_idx):
                state["paused"] = True
                state["current_frame"] = max(0, total_frames - 1)
                time.sleep(0.05)
                continue

            interp_gap = state["interp_gap"]
            for ahead in range(1, interp_gap + 1):
                if play_idx + ahead < total_frames:
                    ensure_frame_buffered(play_idx + ahead)

            min_keep = play_idx - interp_gap - 1
            for old_k in list(frame_ring_buffer.keys()):
                if old_k < min_keep:
                    del frame_ring_buffer[old_k]

            state["current_frame"] = play_idx
            frame = frame_ring_buffer[play_idx].copy()

            persons_snapshot = get_persons_snapshot()
            action_mode = state["action_mode"]
            mosaic_target_mode = state["mosaic_target_mode"]
            mosaic_scale = state["mosaic_scale"]
            threshold = state["threshold"]
            show_debug = state["show_debug"]

            win_start = max(0, play_idx - interp_gap)
            win_end = min(total_frames - 1, play_idx + interp_gap)
            evaluated_map = {}
            with cache_lock:
                for w_idx in range(win_start, win_end + 1):
                    if w_idx in state["feature_cache"]:
                        evaluated_map[w_idx] = evaluate_frame_faces(
                            state["feature_cache"][w_idx],
                            persons_snapshot,
                            action_mode,
                            mosaic_target_mode,
                            threshold
                        )

            render_items = build_interpolated_items(play_idx, evaluated_map, interp_gap, action_mode)

            # ポップアップ通知用に、現在フレームで検出された人物リストを更新
            detected_map = {}
            for _, _, _, _, is_t, score, _, _, p_id, p_name in render_items:
                if is_t and p_id and p_name:
                    if p_id not in detected_map or score > detected_map[p_id]["score"]:
                        detected_map[p_id] = {"id": p_id, "name": p_name, "score": float(score)}
            state["latest_detections"] = list(detected_map.values())

            frame = render_items_on_frame(
                frame,
                render_items,
                len(persons_snapshot) > 0,
                action_mode,
                mosaic_scale,
                show_debug
            )

            _, buffer = cv2.imencode('.jpg', frame)
            frame_bytes = buffer.tobytes()

            if not state["paused"]:
                play_idx += 1
                elapsed = time.time() - start_time
                if frame_delay > elapsed:
                    time.sleep(frame_delay - elapsed)

            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
    finally:
        cap.release()

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=False)