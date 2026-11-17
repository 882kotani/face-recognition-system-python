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
from flask import Flask, request, Response, jsonify, send_file
from flask_cors import CORS  # ← 追加

app = Flask(__name__)
# フロントエンド（別オリジン）からのアクセスを許可
CORS(app)

UPLOAD_FOLDER = 'uploads'
OUTPUT_FOLDER = 'outputs'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

# モデル設定
DET_MODEL_PATH = "face_detection_yunet_2023mar.onnx"
DET_MODEL_URL = "https://github.com/opencv/opencv_zoo/raw/main/models/face_detection_yunet/face_detection_yunet_2023mar.onnx"
REC_MODEL_PATH = "face_recognition_sface_2021dec.onnx"
REC_MODEL_URL = "https://github.com/opencv/opencv_zoo/raw/main/models/face_recognition_sface/face_recognition_sface_2021dec.onnx"

def download_if_not_exists(path, url, name):
    if not os.path.exists(path):
        print(f"{name}をダウンロードしています...")
        urllib.request.urlretrieve(url, path)
        print(f"{name}のダウンロード完了。")

download_if_not_exists(DET_MODEL_PATH, DET_MODEL_URL, "YuNetモデル")
download_if_not_exists(REC_MODEL_PATH, REC_MODEL_URL, "SFaceモデル")

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
    "target_entries": [],
    "mode": "target_only",
    "threshold": 0.36,
    "show_debug": True,
    "feature_cache": {},
    "export_status": "idle",
    "export_progress": 0,
    "export_message": "",
    "output_path": None
}

def get_all_target_vectors():
    vectors = []
    for entry in state["target_entries"]:
        vectors.extend(entry["vectors"])
    return vectors

def compute_max_similarity(face_feat, target_vectors):
    if not target_vectors:
        return 0.0
    max_score = -1.0
    for t_vec in target_vectors:
        score = recognizer.match(t_vec, face_feat, cv2.FaceRecognizerSF_FR_COSINE)
        if score > max_score:
            max_score = score
    return max_score

def analyze_video_chunk(video_path, frame_indices, det_model_path, rec_model_path):
    if not frame_indices: return {}
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened(): return {}
    
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    detector = cv2.FaceDetectorYN.create(
        model=det_model_path, config="", input_size=(width, height),
        score_threshold=0.6, nms_threshold=0.3, top_k=5000
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
        if not ret: break

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

def serialize_target_entries():
    return [{"id": e["id"], "crop_b64": e["crop_b64"], "source": e["source"]} for e in state["target_entries"]]

# === API Endpoints ===

@app.route('/upload_target', methods=['POST'])
def upload_target():
    files = request.files.getlist('face_image')
    if not files:
        return jsonify({'status': 'error', 'message': 'ファイルがありません'}), 400

    added_count = 0
    for file in files:
        file_bytes = np.frombuffer(file.read(), np.uint8)
        img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
        if img is None: continue

        h, w = img.shape[:2]
        max_side = max(w, h)
        if max_side > 1280:
            scale = 1280 / max_side
            img = cv2.resize(img, (int(w * scale), int(h * scale)))
            h, w = img.shape[:2]

        detector = cv2.FaceDetectorYN.create(
            model=DET_MODEL_PATH, config="", input_size=(w, h),
            score_threshold=0.5, nms_threshold=0.3, top_k=5000
        )
        _, faces = detector.detect(img)
        if faces is None or len(faces) == 0: continue

        target_face = max(faces, key=lambda f: f[2] * f[3])
        aligned_face = recognizer.alignCrop(img, target_face)
        feat_orig = recognizer.feature(aligned_face)
        feat_flip = recognizer.feature(cv2.flip(aligned_face, 1))

        _, buf = cv2.imencode('.jpg', aligned_face)
        state["target_entries"].append({
            "id": str(uuid.uuid4()), "crop_b64": base64.b64encode(buf).decode('utf-8'),
            "source": "photo", "vectors": [feat_orig, feat_flip]
        })
        added_count += 1

    if added_count == 0:
        return jsonify({'status': 'error', 'message': '画像から顔を検出できませんでした。', 'entries': serialize_target_entries()})
    
    state["settings_dirty"] = True
    return jsonify({
        'status': 'ok',
        'message': f'{added_count}枚追加しました',
        'entries': serialize_target_entries()
    })

@app.route('/boost_from_video', methods=['POST'])
def boost_from_video():
    video_path = state["video_path"]
    base_vectors = get_all_target_vectors()
    if not video_path or not os.path.exists(video_path): return jsonify({'status': 'error', 'message': '先に動画を読み込んでください'})
    if not base_vectors: return jsonify({'status': 'error', 'message': '先に宣材写真（ベース顔）を登録してください'})

    cap = cv2.VideoCapture(video_path)
    total, width, height = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)), int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    detector = cv2.FaceDetectorYN.create(model=DET_MODEL_PATH, config="", input_size=(width, height), score_threshold=0.6, nms_threshold=0.3, top_k=5000)

    sample_count = min(45, max(1, total))
    sample_indices = list(range(0, total, max(1, total // sample_count)))[:sample_count]
    candidates = []

    for idx in sample_indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if not ret: continue
        _, faces = detector.detect(frame)
        if faces is None: continue

        best_face, best_score, best_feat = None, -1.0, None
        for face in faces:
            aligned = recognizer.alignCrop(frame, face)
            feat = recognizer.feature(aligned)
            score = compute_max_similarity(feat, base_vectors)
            if score > best_score: best_score, best_face, best_feat = score, aligned, feat
        if best_face is not None and best_score >= 0.25:
            candidates.append((best_score, idx, best_face, best_feat))
    cap.release()

    if not candidates: return jsonify({'status': 'error', 'message': '動画内から一致候補が見つかりませんでした。'})

    candidates.sort(key=lambda x: x[0], reverse=True)
    selected = []
    min_frame_gap = max(1, total // 15)
    for cand in candidates:
        if all(abs(cand[1] - s[1]) >= min_frame_gap for s in selected): selected.append(cand)
        if len(selected) >= 5: break

    for score, idx, aligned_crop, feat in selected:
        feat_flip = recognizer.feature(cv2.flip(aligned_crop, 1))
        _, buf = cv2.imencode('.jpg', aligned_crop)
        state["target_entries"].append({
            "id": str(uuid.uuid4()), "crop_b64": base64.b64encode(buf).decode('utf-8'),
            "source": "boost", "vectors": [feat, feat_flip]
        })
    state["settings_dirty"] = True
    return jsonify({'status': 'ok', 'message': f'動画内から {len(selected)}パターンを自動追加しました', 'entries': serialize_target_entries()})

@app.route('/add_current_frame_face', methods=['POST'])
def add_current_frame_face():
    video_path = state["video_path"]
    if not video_path: return jsonify({'status': 'error', 'message': '動画が選択されていません'})
    
    cap = cv2.VideoCapture(video_path)
    cap.set(cv2.CAP_PROP_POS_FRAMES, state["current_frame"])
    ret, frame = cap.read()
    cap.release()
    if not ret: return jsonify({'status': 'error', 'message': 'フレームを読み込めませんでした'})

    h, w = frame.shape[:2]
    detector = cv2.FaceDetectorYN.create(model=DET_MODEL_PATH, config="", input_size=(w, h), score_threshold=0.6, nms_threshold=0.3, top_k=5000)
    _, faces = detector.detect(frame)
    if faces is None or len(faces) == 0: return jsonify({'status': 'error', 'message': '顔が検出されませんでした'})

    base_vectors = get_all_target_vectors()
    if base_vectors:
        chosen_face = max(faces, key=lambda f: compute_max_similarity(recognizer.feature(recognizer.alignCrop(frame, f)), base_vectors))
    else:
        chosen_face = max(faces, key=lambda f: f[2] * f[3])

    aligned = recognizer.alignCrop(frame, chosen_face)
    _, buf = cv2.imencode('.jpg', aligned)
    state["target_entries"].append({
        "id": str(uuid.uuid4()), "crop_b64": base64.b64encode(buf).decode('utf-8'),
        "source": "boost", "vectors": [recognizer.feature(aligned), recognizer.feature(cv2.flip(aligned, 1))]
    })
    state["settings_dirty"] = True
    return jsonify({'status': 'ok', 'message': '現在の顔を追加しました', 'entries': serialize_target_entries()})

@app.route('/remove_target', methods=['POST'])
def remove_target():
    state["target_entries"] = [e for e in state["target_entries"] if e["id"] != request.get_json().get('id')]
    state["settings_dirty"] = True
    return jsonify({'status': 'ok', 'entries': serialize_target_entries()})

@app.route('/clear_targets', methods=['POST'])
def clear_targets():
    state["target_entries"] = []
    state["settings_dirty"] = True
    return jsonify({'status': 'ok'})

@app.route('/upload_video', methods=['POST'])
def upload_video():
    if 'video' not in request.files: return jsonify({'status': 'error'}), 400
    file = request.files['video']
    saved_path = os.path.join(UPLOAD_FOLDER, 'temp_video.mp4')
    file.save(saved_path)

    cap = cv2.VideoCapture(saved_path)
    total_frames, fps = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)), cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0: fps = 30.0
    cap.release()

    with cache_lock: state["feature_cache"] = {}
    state.update({
        "stream_id": str(uuid.uuid4()), "video_path": saved_path, "original_name": os.path.splitext(file.filename)[0],
        "total_frames": total_frames, "fps": fps, "current_frame": 0, "seek_frame": None, "paused": False,
        "export_status": "idle", "export_progress": 0, "output_path": None
    })
    return jsonify({'status': 'ok', 'total_frames': total_frames, 'fps': fps})

@app.route('/control', methods=['POST'])
def control():
    data = request.get_json()
    if data.get('action') == 'seek': state['seek_frame'] = max(0, min(state['total_frames'] - 1, int(data.get('frame', 0))))
    elif data.get('action') == 'toggle_pause':
        if state['paused'] and state['current_frame'] >= state['total_frames'] - 1: state['seek_frame'] = 0
        state['paused'] = not state['paused']
    return jsonify({'status': 'ok', 'paused': state['paused'], 'current_frame': state['current_frame']})

@app.route('/progress')
def progress(): return jsonify({'current_frame': state['current_frame'], 'total_frames': state['total_frames'], 'paused': state['paused']})

@app.route('/update_settings', methods=['POST'])
def update_settings():
    data = request.get_json()
    if 'mode' in data: state['mode'] = data['mode']
    if 'threshold' in data: state['threshold'] = float(data['threshold'])
    if 'show_debug' in data: state['show_debug'] = bool(data['show_debug'])
    state['settings_dirty'] = True
    return jsonify({'status': 'ok'})

def apply_mosaic(frame, x, y, w, h, scale=16):
    x1, y1, x2, y2 = max(0, x), max(0, y), min(frame.shape[1], x + w), min(frame.shape[0], y + h)
    if x2 - x1 > 0 and y2 - y1 > 0:
        small = cv2.resize(frame[y1:y2, x1:x2], (max(1, (x2-x1)//scale), max(1, (y2-y1)//scale)), interpolation=cv2.INTER_LINEAR)
        frame[y1:y2, x1:x2] = cv2.resize(small, (x2-x1, y2-y1), interpolation=cv2.INTER_NEAREST)

def render_mosaic_from_features(frame, frame_faces, target_vectors, mode, threshold, show_debug):
    has_targets = len(target_vectors) > 0
    for x, y, w, h, feat in frame_faces:
        score = compute_max_similarity(feat, target_vectors) if has_targets else 0.0
        is_target = score >= threshold
        should_mosaic = True if not has_targets else (is_target if mode == "target_only" else not is_target)
        if should_mosaic: apply_mosaic(frame, x, y, w, h)
        if show_debug and has_targets:
            color = (0, 0, 255) if is_target else (0, 255, 0)
            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
            cv2.putText(frame, f"T({score:.2f})" if is_target else f"O({score:.2f})", (x, max(20, y - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)
    return frame

def export_video_worker(video_path, target_vectors, mode, threshold):
    final_out_path = os.path.join(OUTPUT_FOLDER, 'final_mosaic.mp4')
    ffmpeg_proc, writer = None, None
    try:
        cap = cv2.VideoCapture(video_path)
        fps, width, height, total = cap.get(cv2.CAP_PROP_FPS), int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)), int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if fps <= 0: fps = 30.0
        cap.release()

        with cache_lock: missing_frames = [i for i in range(total) if i not in state["feature_cache"]]
        if missing_frames:
            num_workers = min(os.cpu_count() or 4, 8, max(1, len(missing_frames) // 30))
            chunk_size = max(1, (len(missing_frames) + num_workers * 4 - 1) // (num_workers * 4))
            chunks = [missing_frames[i:i + chunk_size] for i in range(0, len(missing_frames), chunk_size)]
            completed = total - len(missing_frames)
            with ProcessPoolExecutor(max_workers=num_workers) as executor:
                for future in as_completed([executor.submit(analyze_video_chunk, video_path, ch, DET_MODEL_PATH, REC_MODEL_PATH) for ch in chunks]):
                    res = future.result()
                    with cache_lock: state["feature_cache"].update(res)
                    completed += len(res)
                    state["export_progress"] = min(75, int((completed / total) * 75))
                    state["export_message"] = f"並列AI解析中... {state['export_progress']}%"

        cap = cv2.VideoCapture(video_path)
        ffmpeg_bin = shutil.which("ffmpeg")
        if ffmpeg_bin:
            cmd = [ffmpeg_bin, "-y", "-f", "rawvideo", "-vcodec", "rawvideo", "-s", f"{width}x{height}", "-pix_fmt", "bgr24", "-r", str(fps), "-i", "-", "-i", video_path, "-map", "0:v:0", "-map", "1:a?", "-c:v", "libx264", "-crf", "18", "-preset", "veryfast", "-pix_fmt", "yuv420p", "-c:a", "aac", "-shortest", final_out_path]
            ffmpeg_proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        else:
            writer = cv2.VideoWriter(final_out_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))

        with cache_lock: local_cache = dict(state["feature_cache"])
        for frame_idx in range(total):
            ret, frame = cap.read()
            if not ret: break
            processed = render_mosaic_from_features(frame, local_cache.get(frame_idx, []), target_vectors, mode, threshold, False)
            if ffmpeg_proc: ffmpeg_proc.stdin.write(processed.tobytes())
            elif writer: writer.write(processed)
            if frame_idx % 10 == 0:
                state["export_progress"] = 75 + min(24, int((frame_idx / total) * 24))
                state["export_message"] = f"エンコード中... {state['export_progress']}%"
        
        if ffmpeg_proc: ffmpeg_proc.stdin.close(); ffmpeg_proc.wait()
        if writer: writer.release()
        state["output_path"] = final_out_path
        state["export_progress"], state["export_status"], state["export_message"] = 100, "done", "完了！"
    except Exception as e:
        state["export_status"], state["export_message"] = "error", f"エラー: {str(e)}"
    finally:
        if ffmpeg_proc:
            try: ffmpeg_proc.kill()
            except: pass
        if cap: cap.release()

@app.route('/export_video', methods=['POST'])
def export_video():
    if not state["video_path"]: return jsonify({'status': 'error', 'message': '動画未選択'}), 400
    if state["export_status"] == "processing": return jsonify({'status': 'error', 'message': '処理中'}), 400
    state.update({"paused": True, "export_status": "processing", "export_progress": 0, "output_path": None})
    threading.Thread(target=export_video_worker, args=(state["video_path"], get_all_target_vectors(), state["mode"], state["threshold"]), daemon=True).start()
    return jsonify({'status': 'ok'})

@app.route('/export_status')
def export_status(): return jsonify({'status': state["export_status"], 'progress': state["export_progress"], 'message': state["export_message"]})

@app.route('/download_video')
def download_video():
    if not state.get("output_path"): return "Not Found", 404
    return send_file(state["output_path"], as_attachment=True, download_name=f"{state['original_name']}_mosaic.mp4", mimetype='video/mp4')

def generate_frames():
    my_stream_id = state["stream_id"]
    if not state["video_path"]: return
    cap = cv2.VideoCapture(state["video_path"])
    try:
        fps, width, height = state["fps"], int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        frame_delay = 1.0 / fps if fps > 0 else 0.03
        detector = cv2.FaceDetectorYN.create(model=DET_MODEL_PATH, config="", input_size=(width, height), score_threshold=0.6, nms_threshold=0.3, top_k=5000)

        while cap.isOpened() and state["stream_id"] == my_stream_id:
            start_time = time.time()
            force_render = False

            if state["seek_frame"] is not None:
                cap.set(cv2.CAP_PROP_POS_FRAMES, state["seek_frame"])
                state["current_frame"] = state["seek_frame"]
                state["seek_frame"] = None
                force_render = True

            if state["paused"] and not force_render:
                if state["settings_dirty"]: state["settings_dirty"], force_render = False, True
                else: time.sleep(0.05); continue
            
            state["settings_dirty"] = False
            ret, frame = cap.read()
            if not ret:
                state["paused"], state["current_frame"] = True, max(0, state["total_frames"] - 1)
                time.sleep(0.05); continue

            pos = max(0, int(cap.get(cv2.CAP_PROP_POS_FRAMES)) - 1)
            state["current_frame"] = pos

            with cache_lock: frame_faces = state["feature_cache"].get(pos)
            if frame_faces is None:
                _, faces = detector.detect(frame)
                frame_faces = []
                if faces is not None:
                    for f in faces:
                        aligned = recognizer.alignCrop(frame, f)
                        frame_faces.append((*map(int, f[:4]), recognizer.feature(aligned)))
                with cache_lock: state["feature_cache"][pos] = frame_faces

            frame = render_mosaic_from_features(frame, frame_faces, get_all_target_vectors(), state["mode"], state["threshold"], state["show_debug"])
            _, buffer = cv2.imencode('.jpg', frame)
            
            if not state["paused"]:
                elapsed = time.time() - start_time
                if frame_delay > elapsed: time.sleep(frame_delay - elapsed)

            yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')
    finally: cap.release()

@app.route('/video_feed')
def video_feed(): return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=False)