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

# メインプロセス用認識器
recognizer = cv2.FaceRecognizerSF.create(model=REC_MODEL_PATH, config="")
cache_lock = threading.Lock()

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
    # 複数ターゲット管理: [{"id": str, "crop_b64": str, "source": "photo"|"boost", "vectors": [np.ndarray, ...]}, ...]
    "target_entries": [],
    "mode": "target_only",        # "target_only" or "except_target"
    "threshold": 0.36,
    "interp_gap": 5,              # チラつき防止: 前後補間する最大欠損フレーム数 (0=OFF, 最大50)
    "show_debug": True,
    # フレームごとの顔座標＋特徴量キャッシュ: { frame_idx: [(x, y, w, h, feat), ...] }
    "feature_cache": {},
    # 書き出し・ダウンロード状態
    "export_status": "idle",
    "export_progress": 0,
    "export_message": "",
    "output_path": None
}


def get_all_target_vectors():
    """登録されている全ターゲット（宣材写真＋補強キャプチャ）の特徴ベクトル一覧を取得"""
    vectors = []
    for entry in state["target_entries"]:
        vectors.extend(entry["vectors"])
    return vectors


def compute_max_similarity(face_feat, target_vectors):
    """候補の顔特徴量と、登録済み全テンプレートベクトルとの最大コサイン類似度を算出"""
    if not target_vectors:
        return 0.0
    max_score = -1.0
    for t_vec in target_vectors:
        score = recognizer.match(t_vec, face_feat, cv2.FaceRecognizerSF_FR_COSINE)
        if score > max_score:
            max_score = score
    return max_score


def compute_iou(box_a, box_b):
    """2つの矩形 (x, y, w, h) の IoU（重なり率）を算出"""
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


def evaluate_frame_faces(frame_faces, target_vectors, mode, threshold):
    """
    1フレームの抽出済み顔リストに対し、類似度判定とモザイク要否を評価する
    戻り値: [(x, y, w, h, is_target, score, should_mosaic), ...]
    """
    has_targets = len(target_vectors) > 0
    evaluated = []
    for x, y, w, h, feat in frame_faces:
        is_target = False
        score = 0.0
        if has_targets:
            score = compute_max_similarity(feat, target_vectors)
            is_target = (score >= threshold)

        if not has_targets:
            should_mosaic = True
        elif mode == "target_only":
            should_mosaic = is_target
        else:
            should_mosaic = not is_target

        evaluated.append((x, y, w, h, is_target, score, should_mosaic))
    return evaluated


def build_interpolated_items(target_idx, evaluated_map, max_gap):
    """
    前後フレームのモザイク対象枠を参照し、欠損フレームを線形補間してチラつきを防止する
    evaluated_map: { frame_idx: [(x, y, w, h, is_target, score, should_mosaic), ...] }
    戻り値: [(x, y, w, h, is_target, score, should_mosaic, is_interp), ...]
    """
    raw_items = evaluated_map.get(target_idx, [])
    result_items = [
        (x, y, w, h, is_target, score, should_mosaic, False)
        for (x, y, w, h, is_target, score, should_mosaic) in raw_items
    ]

    if max_gap <= 0:
        return result_items

    covered_mosaic_boxes = [
        (x, y, w, h) for (x, y, w, h, _, _, should_mosaic, _) in result_items if should_mosaic
    ]

    # 過去側・未来側でモザイク対象枠が存在するフレームのみ事前抽出（最大50fでも高速動作）
    past_boxes_map = {}
    for p in range(target_idx - max_gap, target_idx):
        if p in evaluated_map:
            b_list = [(x, y, w, h) for (x, y, w, h, _, _, sm) in evaluated_map[p] if sm]
            if b_list:
                past_boxes_map[p] = b_list

    if not past_boxes_map:
        return result_items

    future_boxes_map = {}
    for n in range(target_idx + 1, target_idx + max_gap + 1):
        if n in evaluated_map:
            b_list = [(x, y, w, h) for (x, y, w, h, _, _, sm) in evaluated_map[n] if sm]
            if b_list:
                future_boxes_map[n] = b_list

    if not future_boxes_map:
        return result_items

    # 時間幅 (n - p) が狭いペアから順に評価
    valid_pairs = []
    for p, boxes_p in past_boxes_map.items():
        for n, boxes_n in future_boxes_map.items():
            span = n - p
            if span <= max_gap + 1:
                valid_pairs.append((span, p, n, boxes_p, boxes_n))

    valid_pairs.sort(key=lambda item: item[0])

    for span, p, n, boxes_p, boxes_n in valid_pairs:
        pair_candidates = []
        for idx_p, bp in enumerate(boxes_p):
            xp, yp, wp, hp = bp
            cx_p, cy_p = xp + wp * 0.5, yp + hp * 0.5
            area_p = max(1, wp * hp)

            for idx_n, bn in enumerate(boxes_n):
                xn, yn, wn, hn = bn
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

            xp, yp, wp, hp = bp
            xn, yn, wn, hn = bn
            ix = int(round(xp + (xn - xp) * alpha))
            iy = int(round(yp + (yn - yp) * alpha))
            iw = int(round(wp + (wn - wp) * alpha))
            ih = int(round(hp + (hn - hp) * alpha))
            interp_box = (ix, iy, iw, ih)

            if all(compute_iou(interp_box, existing) < 0.35 for existing in covered_mosaic_boxes):
                covered_mosaic_boxes.append(interp_box)
                result_items.append((ix, iy, iw, ih, True, 0.0, True, True))

    return result_items


def analyze_video_chunk(video_path, frame_indices, det_model_path, rec_model_path):
    """
    マルチプロセスワーカー関数（トップレベル定義）
    指定されたフレーム番号リストに対し、フル解像度で顔検出と特徴量抽出を行う
    """
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
    <title>AI 特定人物 顔認識モザイク</title>
    <style>
        body {
            font-family: 'Segoe UI', Arial, sans-serif;
            display: flex;
            flex-direction: column;
            align-items: center;
            padding: 24px;
            background-color: #f4f6f9;
            color: #333;
        }
        .container {
            max-width: 920px;
            width: 100%;
        }
        .step-grid {
            display: grid;
            grid-template-columns: 1.2fr 0.8fr;
            gap: 16px;
            margin-bottom: 16px;
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
        }
        .drop-zone {
            height: 86px;
            border: 2px dashed #007bff;
            border-radius: 6px;
            display: flex;
            flex-direction: column;
            justify-content: center;
            align-items: center;
            background: #fafcff;
            cursor: pointer;
            text-align: center;
            padding: 8px;
            transition: 0.2s;
            font-size: 13px;
            color: #555;
        }
        .drop-zone.dragover {
            background: #e3f2fd;
            border-color: #0056b3;
        }
        #target-list {
            display: flex;
            flex-wrap: wrap;
            gap: 8px;
            margin-top: 10px;
            min-height: 24px;
        }
        .target-chip {
            position: relative;
            width: 54px;
            height: 54px;
            border-radius: 6px;
            overflow: hidden;
            border: 2px solid #28a745;
            box-shadow: 0 1px 3px rgba(0,0,0,0.15);
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
            font-size: 9px;
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
            width: 16px;
            height: 16px;
            font-size: 10px;
            line-height: 14px;
            cursor: pointer;
            padding: 0;
        }
        .boost-actions {
            display: flex;
            flex-wrap: wrap;
            gap: 8px;
            margin-top: 10px;
        }
        .sub-btn {
            background: #f8f9fa;
            border: 1px solid #ced4da;
            border-radius: 4px;
            padding: 5px 10px;
            font-size: 12px;
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
        .sub-btn.accent:hover:not(:disabled) {
            background: #ffe69c;
        }
        .sub-btn:disabled {
            opacity: 0.5;
            cursor: not-allowed;
        }
        .controls {
            display: flex;
            flex-wrap: wrap;
            gap: 16px 22px;
            align-items: center;
            justify-content: space-between;
            margin-bottom: 14px;
        }
        .control-group {
            display: flex;
            align-items: center;
            gap: 8px;
            font-size: 13px;
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
    </style>
</head>
<body>
<div class="container">
    <h2 style="text-align:center;">AI 特定人物 顔認識モザイク</h2>

    <div class="step-grid">
        <!-- STEP 1: ターゲット顔写真（複数枚＆自動補強対応） -->
        <div class="card">
            <h3>
                <span>1. 検出したい人の顔写真（複数枚追加可）</span>
                <button class="sub-btn" onclick="clearTargets()" style="font-size:11px;">全クリア</button>
            </h3>
            <div id="face-drop" class="drop-zone">
                <span>宣材写真・顔写真 (JPG / PNG) をドロップ<br>※複数枚の追加や左右反転特徴量の自動生成に対応</span>
                <input type="file" id="face-input" accept="image/*" multiple style="display: none;">
            </div>

            <div id="target-list"></div>
            <div id="face-status" style="font-size:12px; color:#666; margin-top:6px;">
                未登録（宣材写真を入れる場合は、複数枚登録または下の「動画内顔で補強」が有効です）
            </div>

            <div class="boost-actions">
                <button id="boost-btn" class="sub-btn accent" onclick="boostFromVideo()" disabled>
                    ⚡ 動画内の本人顔で自動補強（宣材写真向け）
                </button>
                <button id="capture-frame-btn" class="sub-btn" onclick="captureCurrentFrameFace()" disabled>
                    ＋ 今映っている顔を追加
                </button>
            </div>
        </div>

        <!-- STEP 2: 対象動画 -->
        <div class="card">
            <h3>2. 処理する動画ファイル</h3>
            <div id="video-drop" class="drop-zone">
                <span>MP4 ファイルをドロップ<br>またはクリックして選択</span>
                <input type="file" id="video-input" accept="video/mp4" style="display: none;">
            </div>
            <div style="margin-top: 12px; font-size: 13px; color: #666;" id="video-status">
                動画未選択
            </div>
        </div>
    </div>

    <!-- 設定コントロール -->
    <div class="card controls">
        <div class="control-group">
            <strong>モザイク対象:</strong>
            <label><input type="radio" name="mode" value="target_only" checked onchange="updateSettings()"> 特定の人「だけ」</label>
            <label><input type="radio" name="mode" value="except_target" onchange="updateSettings()"> 特定の人「以外」</label>
        </div>

        <div class="control-group">
            <label for="threshold"><strong>一致閾値:</strong> <span id="thresh-val">0.36</span></label>
            <input type="range" id="threshold" min="0.20" max="0.65" step="0.01" value="0.36" oninput="updateSettings()" style="width:105px;">
        </div>

        <div class="control-group">
            <label for="interp-gap"><strong>チラつき防止補間:</strong> <span id="interp-val">5f</span></label>
            <input type="range" id="interp-gap" min="0" max="50" step="1" value="5" oninput="updateSettings()" style="width:105px;">
        </div>

        <div class="control-group">
            <label><input type="checkbox" id="show-debug" checked onchange="updateSettings()"> 判定枠表示</label>
        </div>
    </div>

    <div id="status">顔写真と動画を選択してください</div>

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
            <div id="export-status-text">現在の設定（モザイク対象・一致閾値・チラつき防止補間）で動画全体を高速書き出しします</div>
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
    let totalFrames = 0;
    let videoFps = 30.0;
    let hasVideo = false;
    let hasTargets = false;
    let isDraggingSeek = false;
    let progressTimer = null;
    let exportTimer = null;

    const seekBar = document.getElementById('seek-bar');
    const playPauseBtn = document.getElementById('play-pause-btn');
    const timeDisplay = document.getElementById('time-display');

    function updateActionButtons() {
        document.getElementById('boost-btn').disabled = !(hasVideo && hasTargets);
        document.getElementById('capture-frame-btn').disabled = !hasVideo;
    }

    function renderTargetList(entries) {
        const listEl = document.getElementById('target-list');
        listEl.innerHTML = '';
        hasTargets = entries && entries.length > 0;

        entries.forEach(item => {
            const div = document.createElement('div');
            div.className = 'target-chip' + (item.source === 'boost' ? ' boost' : '');
            const label = item.source === 'boost' ? '動画抽出' : '登録画像';
            div.innerHTML = `
                <img src="data:image/jpeg;base64,${item.crop_b64}" alt="顔">
                <span class="badge">${label}</span>
                <button class="del-btn" title="削除" onclick="removeTarget('${item.id}')">×</button>
            `;
            listEl.appendChild(div);
        });
        updateActionButtons();
    }

    function removeTarget(id) {
        fetch('/remove_target', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ id })
        })
        .then(res => res.json())
        .then(data => {
            renderTargetList(data.entries || []);
            document.getElementById('face-status').textContent =
                data.entries.length > 0 ? `登録済みテンプレート: ${data.entries.length}枚（左右反転含む ${data.entries.length * 2}パターン）` : '未登録';
        });
    }

    function clearTargets() {
        fetch('/clear_targets', { method: 'POST' })
            .then(res => res.json())
            .then(() => {
                renderTargetList([]);
                const faceStatus = document.getElementById('face-status');
                faceStatus.textContent = '登録をクリアしました';
                faceStatus.style.color = '#666';
            });
    }

    function boostFromVideo() {
        const btn = document.getElementById('boost-btn');
        const faceStatus = document.getElementById('face-status');
        btn.disabled = true;
        faceStatus.textContent = '動画内から宣材写真に一致する本人顔をサンプリング抽出中...';
        faceStatus.style.color = '#856404';

        fetch('/boost_from_video', { method: 'POST' })
            .then(res => res.json())
            .then(data => {
                btn.disabled = false;
                if (data.status === 'ok') {
                    renderTargetList(data.entries);
                    faceStatus.textContent = data.message;
                    faceStatus.style.color = '#28a745';
                } else {
                    faceStatus.textContent = data.message || '動画内から一致する顔を見つけられませんでした';
                    faceStatus.style.color = '#dc3545';
                }
            })
            .catch(() => {
                btn.disabled = false;
                faceStatus.textContent = '補強処理中にエラーが発生しました';
            });
    }

    function captureCurrentFrameFace() {
        const faceStatus = document.getElementById('face-status');
        fetch('/add_current_frame_face', { method: 'POST' })
            .then(res => res.json())
            .then(data => {
                if (data.status === 'ok') {
                    renderTargetList(data.entries);
                    faceStatus.textContent = data.message;
                    faceStatus.style.color = '#28a745';
                } else {
                    faceStatus.textContent = data.message || '現在のフレームから顔を検出できませんでした';
                    faceStatus.style.color = '#dc3545';
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
                })
                .catch(() => {});
        }, 250);
    }

    function setupDropZone(zoneId, inputId, handler) {
        const zone = document.getElementById(zoneId);
        const input = document.getElementById(inputId);
        zone.addEventListener('click', () => input.click());
        zone.addEventListener('dragover', (e) => { e.preventDefault(); zone.classList.add('dragover'); });
        zone.addEventListener('dragleave', () => zone.classList.remove('dragover'));
        zone.addEventListener('drop', (e) => {
            e.preventDefault();
            zone.classList.remove('dragover');
            if (e.dataTransfer.files.length > 0) handler(e.dataTransfer.files);
        });
        input.addEventListener('change', () => {
            if (input.files.length > 0) handler(input.files);
        });
    }

    setupDropZone('face-drop', 'face-input', (files) => {
        const faceStatus = document.getElementById('face-status');
        faceStatus.textContent = '顔を解析・特徴量拡張中...';
        faceStatus.style.color = '#0056b3';

        const formData = new FormData();
        for (let i = 0; i < files.length; i++) {
            formData.append('face_image', files[i]);
        }

        fetch('/upload_target', { method: 'POST', body: formData })
            .then(res => res.json())
            .then(data => {
                if (data.status === 'ok') {
                    renderTargetList(data.entries);
                    faceStatus.textContent = data.message;
                    faceStatus.style.color = '#28a745';
                } else {
                    faceStatus.textContent = data.message || '顔を検出できませんでした';
                    faceStatus.style.color = '#dc3545';
                }
            });
    });

    setupDropZone('video-drop', 'video-input', (files) => {
        const file = files[0];
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
                    updateActionButtons();

                    seekBar.max = Math.max(0, totalFrames - 1);
                    seekBar.value = 0;
                    updateTimeText(0);

                    videoStatus.textContent = `選択中: ${file.name} (${formatTime(totalFrames / videoFps)})`;
                    statusText.textContent = '再生・特定人物認識処理中...';

                    streamImg.src = '/video_feed?' + new Date().getTime();
                    streamImg.style.display = 'block';
                    playerBar.style.display = 'flex';
                    exportCard.style.display = 'flex';

                    document.getElementById('manual-download-btn').style.display = 'none';
                    document.getElementById('export-progress-bg').style.display = 'none';
                    document.getElementById('export-status-text').textContent = '現在の設定（モザイク対象・一致閾値・チラつき防止補間）で動画全体を高速書き出しします';

                    startProgressPolling();
                }
            });
    });

    function updateSettings() {
        const mode = document.querySelector('input[name="mode"]:checked').value;
        const threshold = parseFloat(document.getElementById('threshold').value);
        const interpGap = parseInt(document.getElementById('interp-gap').value, 10);
        const showDebug = document.getElementById('show-debug').checked;

        document.getElementById('thresh-val').textContent = threshold.toFixed(2);
        document.getElementById('interp-val').textContent = interpGap === 0 ? 'OFF' : `${interpGap}f`;

        fetch('/update_settings', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                mode,
                threshold,
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

def serialize_target_entries():
    return [
        {
            "id": e["id"],
            "crop_b64": e["crop_b64"],
            "source": e["source"]
        }
        for e in state["target_entries"]
    ]

@app.route('/')
def index():
    return render_template_string(HTML_TEMPLATE)

@app.route('/upload_target', methods=['POST'])
def upload_target():
    files = request.files.getlist('face_image')
    if not files:
        return jsonify({'status': 'error', 'message': 'ファイルがありません'}), 400

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

        state["target_entries"].append({
            "id": str(uuid.uuid4()),
            "crop_b64": face_b64,
            "source": "photo",
            "vectors": [feat_orig, feat_flip]
        })
        added_count += 1

    if added_count == 0:
        return jsonify({
            'status': 'error',
            'message': '画像から顔を検出できませんでした。顔が極端に傾いている・隠れている場合は別の写真をお試しください。',
            'entries': serialize_target_entries()
        })

    state["settings_dirty"] = True
    total_cnt = len(state["target_entries"])
    return jsonify({
        'status': 'ok',
        'message': f'{added_count}枚追加しました（計{total_cnt}枚 / 左右反転含む{total_cnt * 2}パターンで照合）',
        'entries': serialize_target_entries()
    })

@app.route('/boost_from_video', methods=['POST'])
def boost_from_video():
    video_path = state["video_path"]
    base_vectors = get_all_target_vectors()

    if not video_path or not os.path.exists(video_path):
        return jsonify({'status': 'error', 'message': '先に動画を読み込んでください'})
    if not base_vectors:
        return jsonify({'status': 'error', 'message': '先に宣材写真（ベース顔）を1枚以上登録してください'})

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
            'message': '動画内から一致候補が見つかりませんでした。別の角度の宣材写真を追加してお試しください。'
        })

    candidates.sort(key=lambda x: x[0], reverse=True)
    selected = []
    min_frame_gap = max(1, total // 15)

    for cand in candidates:
        if all(abs(cand[1] - s[1]) >= min_frame_gap for s in selected):
            selected.append(cand)
        if len(selected) >= 5:
            break

    for score, idx, aligned_crop, feat in selected:
        feat_flip = recognizer.feature(cv2.flip(aligned_crop, 1))
        _, buf = cv2.imencode('.jpg', aligned_crop)
        face_b64 = base64.b64encode(buf).decode('utf-8')
        state["target_entries"].append({
            "id": str(uuid.uuid4()),
            "crop_b64": face_b64,
            "source": "boost",
            "vectors": [feat, feat_flip]
        })

    state["settings_dirty"] = True
    top_score = selected[0][0]
    return jsonify({
        'status': 'ok',
        'message': f'動画内の本人顔 {len(selected)}パターンを自動追加しました（最高一致度: {top_score:.2f}）',
        'entries': serialize_target_entries()
    })

@app.route('/add_current_frame_face', methods=['POST'])
def add_current_frame_face():
    video_path = state["video_path"]
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

    base_vectors = get_all_target_vectors()
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

    state["target_entries"].append({
        "id": str(uuid.uuid4()),
        "crop_b64": face_b64,
        "source": "boost",
        "vectors": [feat_orig, feat_flip]
    })
    state["settings_dirty"] = True

    return jsonify({
        'status': 'ok',
        'message': f'フレーム #{state["current_frame"]} の顔をテンプレートに追加しました',
        'entries': serialize_target_entries()
    })

@app.route('/remove_target', methods=['POST'])
def remove_target():
    data = request.get_json()
    target_id = data.get('id')
    state["target_entries"] = [e for e in state["target_entries"] if e["id"] != target_id]
    state["settings_dirty"] = True
    return jsonify({'status': 'ok', 'entries': serialize_target_entries()})

@app.route('/clear_targets', methods=['POST'])
def clear_targets():
    state["target_entries"] = []
    state["settings_dirty"] = True
    return jsonify({'status': 'ok'})

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
        'paused': state['paused']
    })

@app.route('/update_settings', methods=['POST'])
def update_settings():
    data = request.get_json()
    if 'mode' in data:
        state['mode'] = data['mode']
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

def render_items_on_frame(frame, render_items, has_targets, show_debug):
    """確定したモザイク＆デバッグ枠リストをフレームに描画"""
    for x, y, w, h, is_target, score, should_mosaic, is_interp in render_items:
        if should_mosaic:
            apply_mosaic(frame, x, y, w, h)

        if show_debug and (has_targets or is_interp):
            if is_interp:
                color = (255, 220, 0)  # 水色：前後フレーム補間によるモザイク維持
                label = "INTERP"
            elif is_target:
                color = (0, 0, 255)    # 赤色：ターゲット一致
                label = f"TARGET ({score:.2f})"
            else:
                color = (0, 255, 0)    # 緑色：他人
                label = f"OTHER ({score:.2f})"

            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
            cv2.putText(frame, label, (x, max(20, y - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)

    return frame

def export_video_worker(video_path, target_vectors, mode, threshold, interp_gap):
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

        # 全フレーム分の類似度評価と前後フレーム座標補間を事前計算
        state["export_progress"] = 74
        state["export_message"] = "前後フレーム座標補間（チラつき防止）を計算中..."
        with cache_lock:
            local_cache = dict(state["feature_cache"])

        evaluated_map = {
            idx: evaluate_frame_faces(local_cache.get(idx, []), target_vectors, mode, threshold)
            for idx in range(total)
        }
        has_targets = len(target_vectors) > 0

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

            render_items = build_interpolated_items(frame_idx, evaluated_map, interp_gap)
            processed_frame = render_items_on_frame(frame, render_items, has_targets, show_debug=False)

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

    target_vectors_snapshot = list(get_all_target_vectors())

    t = threading.Thread(
        target=export_video_worker,
        args=(
            state["video_path"],
            target_vectors_snapshot,
            state["mode"],
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

    download_filename = f"{state['original_name']}_mosaic.mp4"
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

        # 先読み用の画像リングバッファ（逆シークせずに数フレーム先までAI解析するため）
        frame_ring_buffer = {}
        play_idx = 0
        cap_read_idx = 0

        def ensure_frame_buffered(idx):
            """指定フレームを画像リングバッファと特徴量キャッシュに読み込む"""
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
            # 前後フレーム補間のため、未来側の最大 interp_gap フレーム分を先読み解析
            for ahead in range(1, interp_gap + 1):
                if play_idx + ahead < total_frames:
                    ensure_frame_buffered(play_idx + ahead)

            # ウィンドウ範囲外の古い生画像だけを解放（メモリ節約）
            min_keep = play_idx - interp_gap - 1
            for old_k in list(frame_ring_buffer.keys()):
                if old_k < min_keep:
                    del frame_ring_buffer[old_k]

            state["current_frame"] = play_idx
            frame = frame_ring_buffer[play_idx].copy()

            target_vectors = get_all_target_vectors()
            mode = state["mode"]
            threshold = state["threshold"]
            show_debug = state["show_debug"]

            # 前後ウィンドウ内のフレームを評価して補間
            win_start = max(0, play_idx - interp_gap)
            win_end = min(total_frames - 1, play_idx + interp_gap)
            evaluated_map = {}
            with cache_lock:
                for w_idx in range(win_start, win_end + 1):
                    if w_idx in state["feature_cache"]:
                        evaluated_map[w_idx] = evaluate_frame_faces(
                            state["feature_cache"][w_idx], target_vectors, mode, threshold
                        )

            render_items = build_interpolated_items(play_idx, evaluated_map, interp_gap)
            frame = render_items_on_frame(frame, render_items, len(target_vectors) > 0, show_debug)

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