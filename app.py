import os
import time
import urllib.request
import cv2
import numpy as np
from flask import Flask, render_template_string, request, Response, jsonify

app = Flask(__name__)
UPLOAD_FOLDER = 'uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

MODEL_PATH = "face_detection_yunet_2023mar.onnx"
MODEL_URL = "https://github.com/opencv/opencv_zoo/raw/main/models/face_detection_yunet/face_detection_yunet_2023mar.onnx"
STAMP_IMAGE_PATH = "face_stamp.png"

# YuNetモデルが存在しない場合は自動ダウンロード
if not os.path.exists(MODEL_PATH):
    print("YuNetモデルをダウンロードしています...")
    urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
    print("ダウンロード完了。")

current_video_path = None

HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="ja">
<head>
    <meta charset="UTF-8">
    <title>AI 顔スタンプ合成 (動画 / Webカメラ)</title>
    <style>
        body {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            display: flex;
            flex-direction: column;
            align-items: center;
            padding: 30px;
            background-color: #f0f2f5;
            color: #333;
        }
        h2 {
            margin-bottom: 20px;
        }
        .control-panel {
            display: flex;
            gap: 15px;
            margin-bottom: 20px;
        }
        .btn {
            padding: 10px 20px;
            font-size: 15px;
            font-weight: bold;
            border: none;
            border-radius: 6px;
            cursor: pointer;
            transition: 0.2s ease;
        }
        .btn-primary {
            background-color: #007bff;
            color: white;
        }
        .btn-primary:hover {
            background-color: #0056b3;
        }
        .btn-secondary {
            background-color: #6c757d;
            color: white;
        }
        .btn-secondary:hover {
            background-color: #545b62;
        }
        .btn:disabled {
            background-color: #ccc;
            cursor: not-allowed;
        }
        #drop-zone {
            width: 640px;
            height: 120px;
            border: 2px dashed #007bff;
            border-radius: 8px;
            display: flex;
            justify-content: center;
            align-items: center;
            background: #ffffff;
            cursor: pointer;
            margin-bottom: 15px;
            transition: 0.2s;
        }
        #drop-zone.dragover {
            background: #e3f2fd;
            border-color: #0056b3;
        }
        #drop-zone p {
            margin: 0;
            color: #555;
            font-size: 14px;
        }
        #status {
            margin-bottom: 15px;
            font-weight: bold;
            font-size: 15px;
        }
        #preview-box {
            max-width: 800px;
            width: 100%;
            min-height: 400px;
            background: #111;
            border-radius: 8px;
            overflow: hidden;
            display: flex;
            justify-content: center;
            align-items: center;
            box-shadow: 0 4px 12px rgba(0,0,0,0.15);
        }
        #stream-img {
            width: 100%;
            height: auto;
            display: none;
        }
    </style>
</head>
<body>
    <h2>AI 顔スタンプ合成 リアルタイムシステム</h2>
    
    <div class="control-panel">
        <button id="btn-webcam" class="btn btn-primary">📹 Webカメラを開始</button>
        <button id="btn-file-mode" class="btn btn-secondary">📁 動画ドロップに戻す</button>
    </div>

    <div id="drop-zone">
        <p>ここに MP4 ファイルをドロップ（またはクリックして選択）</p>
        <input type="file" id="file-input" accept="video/mp4" style="display: none;">
    </div>

    <div id="status">動画を選択するか、Webカメラを開始してください</div>

    <div id="preview-box">
        <img id="stream-img" alt="映像ストリーム">
    </div>

    <script>
        const dropZone = document.getElementById('drop-zone');
        const fileInput = document.getElementById('file-input');
        const streamImg = document.getElementById('stream-img');
        const statusText = document.getElementById('status');
        const btnWebcam = document.getElementById('btn-webcam');
        const btnFileMode = document.getElementById('btn-file-mode');

        let currentMode = 'none';

        // Webカメラモードへの切り替え
        btnWebcam.addEventListener('click', () => {
            currentMode = 'webcam';
            dropZone.style.display = 'none';
            statusText.textContent = 'Webカメラを起動中...';
            
            // 映像ストリームをWebカメラモードで再接続
            streamImg.src = '/video_feed?source=webcam&t=' + new Date().getTime();
            streamImg.style.display = 'block';
            statusText.textContent = 'Webカメラ映像を配信中（リアルタイム顔スタンプ合成）';
        });

        // 動画ファイルモードへの切り替え
        btnFileMode.addEventListener('click', () => {
            currentMode = 'file';
            dropZone.style.display = 'flex';
            statusText.textContent = 'MP4ファイルをドロップしてください';
            streamImg.src = '';
            streamImg.style.display = 'none';
        });

        // ドロップゾーンの処理
        dropZone.addEventListener('click', () => fileInput.click());

        dropZone.addEventListener('dragover', (e) => {
            e.preventDefault();
            dropZone.classList.add('dragover');
        });

        dropZone.addEventListener('dragleave', () => {
            dropZone.classList.remove('dragover');
        });

        dropZone.addEventListener('drop', (e) => {
            e.preventDefault();
            dropZone.classList.remove('dragover');
            if (e.dataTransfer.files.length > 0) {
                uploadAndPlay(e.dataTransfer.files[0]);
            }
        });

        fileInput.addEventListener('change', () => {
            if (fileInput.files.length > 0) {
                uploadAndPlay(fileInput.files[0]);
            }
        });

        function uploadAndPlay(file) {
            if (file.type !== 'video/mp4') {
                alert('MP4形式の動画ファイルを選択してください。');
                return;
            }

            statusText.textContent = '動画を読み込み中...';
            const formData = new FormData();
            formData.append('video', file);

            fetch('/upload', {
                method: 'POST',
                body: formData
            })
            .then(res => res.json())
            .then(data => {
                if (data.status === 'ok') {
                    statusText.textContent = '動画を再生中（顔スタンプ合成中）...';
                    streamImg.src = '/video_feed?source=file&t=' + new Date().getTime();
                    streamImg.style.display = 'block';
                }
            })
            .catch(err => {
                console.error(err);
                statusText.textContent = 'アップロードに失敗しました。';
            });
        }
    </script>
</body>
</html>
"""

def overlay_image_alpha(bg, fg, x, y, w, h):
    """透過PNGを背景画像に重ね合わせる関数"""
    if w <= 0 or h <= 0:
        return bg

    fg_resized = cv2.resize(fg, (w, h), interpolation=cv2.INTER_AREA)

    x1, y1 = max(0, x), max(0, y)
    x2, y2 = min(bg.shape[1], x + w), min(bg.shape[0], y + h)

    if x2 <= x1 or y2 <= y1:
        return bg

    fg_x1 = x1 - x
    fg_y1 = y1 - y
    fg_x2 = fg_x1 + (x2 - x1)
    fg_y2 = fg_y1 + (y2 - y1)

    roi = bg[y1:y2, x1:x2]
    fg_sub = fg_resized[fg_y1:fg_y2, fg_x1:fg_x2]

    if fg_sub.shape[2] == 4:
        alpha = (fg_sub[:, :, 3] / 255.0)[:, :, np.newaxis]
        bg[y1:y2, x1:x2] = (alpha * fg_sub[:, :, :3] + (1.0 - alpha) * roi).astype(np.uint8)
    else:
        bg[y1:y2, x1:x2] = fg_sub

    return bg

@app.route('/')
def index():
    return render_template_string(HTML_TEMPLATE)

@app.route('/upload', methods=['POST'])
def upload():
    global current_video_path
    if 'video' not in request.files:
        return jsonify({'status': 'error'}), 400
    
    file = request.files['video']
    saved_path = os.path.join(UPLOAD_FOLDER, 'temp_video.mp4')
    file.save(saved_path)
    current_video_path = saved_path
    return jsonify({'status': 'ok'})

def generate_frames(source_type):
    global current_video_path

    # スタンプ画像の読み込み
    stamp = None
    if os.path.exists(STAMP_IMAGE_PATH):
        stamp = cv2.imread(STAMP_IMAGE_PATH, cv2.IMREAD_UNCHANGED)

    # 入力ソースの決定
    if source_type == 'webcam':
        cap = cv2.VideoCapture(0)
        is_webcam = True
    else:
        if not current_video_path or not os.path.exists(current_video_path):
            return
        cap = cv2.VideoCapture(current_video_path)
        is_webcam = False

    if not cap.isOpened():
        print(f"入力ソースを開けませんでした: {source_type}")
        return

    # 初回フレームを読み込んで画面サイズを取得
    ret, frame = cap.read()
    if not ret:
        cap.release()
        return

    h, w = frame.shape[:2]
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_delay = (1.0 / fps) if (fps > 0 and not is_webcam) else 0

    # YuNet検出器初期化
    detector = cv2.FaceDetectorYN.create(
        model=MODEL_PATH,
        config="",
        input_size=(w, h),
        score_threshold=0.6,
        nms_threshold=0.3,
        top_k=5000
    )

    while cap.isOpened():
        start_time = time.time()

        # 顔検出
        _, faces = detector.detect(frame)

        if faces is not None and stamp is not None:
            for face in faces:
                fx, fy, fw, fh = map(int, face[:4])

                # 元の顔が隠れるよう 1.3 倍に拡大
                scale = 1.3
                stamp_w = int(fw * scale)
                stamp_h = int(fh * scale)

                # 中心を合わせて少し上寄りに配置
                cx = fx + fw // 2
                cy = fy + fh // 2
                stamp_x = cx - stamp_w // 2
                stamp_y = cy - stamp_h // 2 - int(fh * 0.05)

                frame = overlay_image_alpha(frame, stamp, stamp_x, stamp_y, stamp_w, stamp_h)

        # JPEGエンコード
        _, buffer = cv2.imencode('.jpg', frame)
        frame_bytes = buffer.tobytes()

        # 動画ファイルの場合は実再生速度に合わせる（Webカメラの場合は遅延なく送信）
        if not is_webcam and frame_delay > 0:
            elapsed = time.time() - start_time
            if frame_delay > elapsed:
                time.sleep(frame_delay - elapsed)

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

        # 次フレーム取得
        ret, frame = cap.read()
        if not ret:
            break

    cap.release()

@app.route('/video_feed')
def video_feed():
    source = request.args.get('source', 'file')
    return Response(generate_frames(source), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=False)