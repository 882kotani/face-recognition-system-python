import os
import time
import urllib.request
import cv2
from flask import Flask, render_template_string, request, Response, jsonify

app = Flask(__name__)
UPLOAD_FOLDER = 'uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

MODEL_PATH = "face_detection_yunet_2023mar.onnx"
MODEL_URL = "https://github.com/opencv/opencv_zoo/raw/main/models/face_detection_yunet/face_detection_yunet_2023mar.onnx"

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
    <title>AI 自動顔 プレビュー</title>
    <style>
        body {
            font-family: Arial, sans-serif;
            display: flex;
            flex-direction: column;
            align-items: center;
            padding: 30px;
            background-color: #f4f6f9;
        }
        #drop-zone {
            width: 640px;
            height: 140px;
            border: 2px dashed #007bff;
            border-radius: 8px;
            display: flex;
            justify-content: center;
            align-items: center;
            background: #ffffff;
            cursor: pointer;
            margin-bottom: 20px;
            transition: 0.2s;
        }
        #drop-zone.dragover {
            background: #e3f2fd;
            border-color: #0056b3;
        }
        #drop-zone p {
            margin: 0;
            color: #555;
            font-size: 15px;
        }
        #status {
            margin-bottom: 15px;
            font-weight: bold;
            color: #333;
        }
        #preview-box {
            max-width: 800px;
            width: 100%;
            background: #000;
            border-radius: 8px;
            overflow: hidden;
            display: flex;
            justify-content: center;
        }
        #stream-img {
            width: 100%;
            height: auto;
            display: none;
        }
    </style>
</head>
<body>
    <h2>AI 自動顔 プレビュー再生</h2>
    
    <div id="drop-zone">
        <p>ここに MP4 ファイルをドロップ（またはクリックして選択）</p>
        <input type="file" id="file-input" accept="video/mp4" style="display: none;">
    </div>

    <div id="status">動画を選択してください</div>

    <div id="preview-box">
        <img id="stream-img" alt="映像ストリーム">
    </div>

    <script>
        const dropZone = document.getElementById('drop-zone');
        const fileInput = document.getElementById('file-input');
        const streamImg = document.getElementById('stream-img');
        const statusText = document.getElementById('status');

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
                    statusText.textContent = '再生・自動モザイク処理中...';
                    streamImg.src = '/video_feed?' + new Date().getTime();
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

def generate_frames():
    global current_video_path
    if not current_video_path or not os.path.exists(current_video_path):
        return

    cap = cv2.VideoCapture(current_video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frame_delay = 1.0 / fps if fps > 0 else 0.03

    # YuNet 検出器初期化
    detector = cv2.FaceDetectorYN.create(
        model=MODEL_PATH,
        config="",
        input_size=(width, height),
        score_threshold=0.6,   # 確信度の閾値
        nms_threshold=0.3,     # 重複枠の除去閾値
        top_k=5000
    )

    while cap.isOpened():
        start_time = time.time()
        ret, frame = cap.read()
        if not ret:
            break

        # 顔検出
        _, faces = detector.detect(frame)

        if faces is not None:
            for face in faces:
                x, y, w, h = map(int, face[:4])

                # 画面外にはみ出した座標のクリッピング
                x1 = max(0, x)
                y1 = max(0, y)
                x2 = min(frame.shape[1], x + w)
                y2 = min(frame.shape[0], y + h)

                roi_w = x2 - x1
                roi_h = y2 - y1

                if roi_w > 0 and roi_h > 0:
                    face_roi = frame[y1:y2, x1:x2]

                    # 縮小時の解像度（数値を大きくするとモザイクの粒が粗くなります）
                    mosaic_scale = 16
                    small_w = max(1, roi_w // mosaic_scale)
                    small_h = max(1, roi_h // mosaic_scale)

                    # 縮小した後に元のサイズへ拡大してモザイク化
                    small = cv2.resize(face_roi, (small_w, small_h), interpolation=cv2.INTER_LINEAR)
                    mosaic = cv2.resize(small, (roi_w, roi_h), interpolation=cv2.INTER_NEAREST)

                    # 元のフレームに上書き
                    frame[y1:y2, x1:x2] = mosaic

        # JPEGエンコード
        _, buffer = cv2.imencode('.jpg', frame)
        frame_bytes = buffer.tobytes()

        # 実再生速度に合わせたディレイ制御
        elapsed = time.time() - start_time
        if frame_delay > elapsed:
            time.sleep(frame_delay - elapsed)

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

    cap.release()

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=False)