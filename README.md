# AI 複数人物 顔認識モザイク ＆ 検出通知システム (`app_mozaiku.py`)

OpenCV（YuNet 顔検出 ＋ SFace 顔認識）と Flask を用いた、特定人物のリアルタイム顔認識モザイク処理およびポップアップ検出通知 Web アプリケーションです。  
**単一 Python ファイル構成（バックエンド API ＋ インライン HTML/CSS/JS フロントエンド）** で動作し、認識精度と出力画質を落とさずにマルチプロセス並列推論と `ffmpeg` パイプエンコードによる高速書き出しを実現しています。

> **AI エージェント / 引き継ぎ開発者向け注記**  
> 本ドキュメントは、別の生成 AI や開発者がコードベースを即座に理解し、安全に機能追加・リファクタリングを行えるよう、アーキテクチャ、データ構造、API 仕様、マルチプロセス上の制約事項を網羅しています。

---

## 1. 主な機能一覧

1. **複数人物の登録・管理**
   - GUI 上の「＋ 人物を追加」ボタンから検出対象人物を無制限に追加・削除可能。
   - 各人物ごとに**人物名（日本語対応）**および**複数枚の顔写真**を登録可能。
2. **ドメインギャップ対策（宣材写真・角度変化への対応）**
   - **左右反転（TTA）拡張**: 登録画像ごとに「元向き」と「水平反転（`cv2.flip`）」の 2 種類の特徴ベクトルを自動生成。
   - **動画内顔での自動補強（Boost）**: 外部写真（宣材写真など）を起点に、動画全体から最大 45 フレームをサンプリングし、最も類似度の高い動画内の本人顔を最大 5 パターン自動抽出してテンプレートへ追加。
   - **現在フレームからの手動追加**: プレビューで任意のシーンを停止し、映っている顔をワンクリックで特定人物のテンプレートへ追加。
3. **2 つの動作モード切り替え**
   - **モザイク処理 (`mosaic`)**: 登録人物「だけ」または「以外」にモザイクを適用。モザイクの粗さ（ブロック縮小率 `4〜50`）をリアルタイム調整可能。
   - **ポップアップ顔検出通知 (`notify`)**: モザイクをかけず、登録人物が検出された瞬間に画面右上にトースト通知（人物名・再生時間・一致度）を表示し、映像上にも日本語の人物名ラベルを描画。
4. **チラつき防止（前後フレーム座標補間）**
   - 一瞬の横顔や遮蔽で検出が途切れた際、前後 `0〜50` フレーム（可変）の検出座標から欠損フレームのバウンディングボックスを線形補間してモザイク・追跡を維持。
5. **インタラクティブ・プレビュー＆高速 MP4 書き出し**
   - 動画アップロード後に自動再生。シークバー移動、再生／一時停止、±5 秒スキップに対応。一時停止中のパラメータ変更も即時再描画。
   - 書き出し時は**フル解像度・全フレーム推論**を維持したまま、キャッシュ再利用 ＋ `ProcessPoolExecutor` 並列解析 ＋ `ffmpeg` 標準入力パイプ直接エンコード（音声維持・H.264）を実行。

---

## 2. ディレクトリ構成と自動生成ファイル

```text
project_root/
├── app_mozaiku.py                      # アプリケーション本体（Flask + OpenCV + フロントエンドHTML）
├── README.md                           # 本ドキュメント
├── face_detection_yunet_2023mar.onnx   # [自動DL] YuNet 顔検出モデル
├── face_recognition_sface_2021dec.onnx # [自動DL] SFace 顔特徴抽出モデル
├── uploads/
│   └── temp_video.mp4                  # [自動生成] アップロードされた入力動画（上書き保存）
└── outputs/
    └── final_mosaic.mp4                # [自動生成] 書き出し完了後のダウンロード用MP4
```

---

## 3. 環境構築・起動方法

### 必須要件

- **Python**: 3.8 以上
- **Python パッケージ**:
  ```bash
  pip install flask opencv-python numpy Pillow
  ```

  - `opencv-python` は `cv2.FaceDetectorYN` と `cv2.FaceRecognizerSF` を含む **4.5.4 以上** が必要です。
  - `Pillow` は映像フレーム上に日本語の人物名を描画するために使用します（未インストール時は ASCII 文字列にフォールバック）。
- **外部ツール（推奨）**:
  - **`ffmpeg`**: システム PATH に存在する場合、書き出し時に元動画の音声トラック結合と H.264 (`libx264`, `crf 18`, `veryfast`) エンコードを 1 パスで実行します。未導入環境では OpenCV の `mp4v`（音声なし）に自動フォールバックします。

### 起動コマンド

```bash
python app_mozaiku.py
```

起動後、ブラウザで `http://127.0.0.1:5000` にアクセスします。初回起動時のみ OpenCV Zoo から ONNX モデル（計 約 37MB）が自動ダウンロードされます。

---

## 4. アーキテクチャと処理パイプライン

### 4.1 処理の全体フロー

1. **顔登録 (`/upload_target`, `/boost_from_video`, `/add_current_frame_face`)**:
   - YuNet で顔領域と 5 点ランドマークを検出 → `recognizer.alignCrop()` でアライメント正規化 → `recognizer.feature()` で **128 次元特徴ベクトル** を抽出。
2. **プレビュー配信 (`/video_feed` -> `generate_frames()`)**:
   - `frame_ring_buffer` を用い、現在再生フレーム `play_idx` から未来側 `interp_gap` フレーム先まで先読みして推論結果を `state["feature_cache"]` に蓄積。
   - 前後ウィンドウ `[play_idx - interp_gap, play_idx + interp_gap]` の評価結果から線形補間を行い、MJPEG ストリームとして配信。
3. **動画書き出し (`/export_video` -> `export_video_worker()`)**:
   - 裏側のプレビュー再生を自動一時停止 (`state["paused"] = True`) して CPU 競合を防止。
   - `state["feature_cache"]` に未登録のフレーム番号のみを抽出し、`ProcessPoolExecutor`（最大 8 ワーカー）で並列フル解像度推論を実行してキャッシュを補完。
   - 全フレームの補間座標を計算後、`ffmpeg` サブプロセスの `stdin` へ生フレーム (`bgr24`) を直接パイプ送信して MP4 を生成。

### 4.2 アルゴリズム仕様

- **顔照合 (`match_face_to_persons`)**:
  - コサイン類似度（`cv2.FaceRecognizerSF_FR_COSINE`）を使用。登録済み全人物・全テンプレート（反転含む）との最大類似度が `threshold`（デフォルト `0.36`）以上の場合に一致と判定。
- **チラつき防止補間 (`build_interpolated_items`)**:
  - 欠損フレーム `target_idx` を挟む過去フレーム `p` と未来フレーム `n`（`n - p <= max_gap + 1`）の追跡枠ペアを探索し、以下の安全条件をすべて満たす場合のみ線形補間枠を生成します。
    1. **同一人物制約**: 両端に人物 ID がある場合、同一 ID であること。
    2. **面積比制限**: `0.45 <= area_n / area_p <= 2.2`（急激なズームや別人誤結合の防止）。
    3. **移動速度制限**: 1 フレームあたりの中心移動距離 `<= 平均顔サイズ * 0.65`（シーン切り替え時の画面横断誤補間の防止）。
    4. **重複除外**: 当該フレームで既に確定している枠との IoU が `0.35` 未満であること。

---

## 5. グローバル状態 (`state`) データ構造

バックエンドのアプリケーション状態は単一の辞書 `state` で管理されています。スレッド間競合を防ぐため、`state_lock` および `cache_lock` を使用します。

```python
state = {
    "stream_id": str | None,         # ストリーム世代管理用UUID（新動画アップロード時に更新）
    "video_path": str | None,        # 現在の動画ファイルパス ("uploads/temp_video.mp4")
    "original_name": str,            # ダウンロードファイル名用の元ファイル名（拡張子なし）
    "total_frames": int,             # 動画の総フレーム数
    "fps": float,                    # 動画のフレームレート
    "current_frame": int,            # 現在の再生・表示フレーム番号 (0-indexed)
    "seek_frame": int | None,        # シーク要求時にセットされる目標フレーム番号
    "paused": bool,                  # 一時停止フラグ
    "settings_dirty": bool,          # 一時停止中の設定変更を検知して再描画するフラグ

    # 複数人物管理リスト
    "persons": [
        {
            "id": str,               # 人物UUID
            "name": str,             # 人物名（UI表示・映像ラベル・通知用）
            "entries": [
                {
                    "id": str,       # テンプレート画像UUID
                    "crop_b64": str, # サムネイル表示用Base64 JPEG
                    "source": str,   # "photo" (手動登録) | "boost" (動画抽出)
                    "vectors": [np.ndarray, np.ndarray] # [元向き128次元, 左右反転128次元]
                }
            ]
        }
    ],

    "action_mode": str,              # "mosaic" | "notify"
    "mosaic_target_mode": str,       # "target_only" | "except_target"
    "mosaic_scale": int,             # モザイク縮小率 (4〜50、大きいほど粗い)
    "threshold": float,              # コサイン類似度閾値 (0.20〜0.65)
    "interp_gap": int,               # チラつき防止補間の最大フレーム幅 (0〜50)
    "show_debug": bool,              # プレビュー時の枠・ラベル描画ON/OFF

    "latest_detections": [           # /progress ポーリングでフロントに返す現在フレーム検出人物
        {"id": str, "name": str, "score": float}
    ],

    # 推論結果キャッシュ: { frame_idx (int): [(x, y, w, h, feat_ndarray), ...] }
    "feature_cache": dict,

    # 書き出し進捗状態
    "export_status": str,            # "idle" | "processing" | "done" | "error"
    "export_progress": int,          # 0〜100
    "export_message": str,           # UI進捗表示テキスト
    "output_path": str | None        # 書き出し完了ファイルパス
}
```

---

## 6. REST API ＆ ストリーミング仕様

| エンドポイント            | メソッド | リクエスト形式                                                                                     | 概要・レスポンス                                                                            |
| :------------------------ | :------- | :------------------------------------------------------------------------------------------------- | :------------------------------------------------------------------------------------------ |
| `/`                       | `GET`    | -                                                                                                  | メイン画面（`HTML_TEMPLATE`）を返却                                                         |
| `/get_persons`            | `GET`    | -                                                                                                  | 登録済み人物リストを返却 (`{"status": "ok", "persons": [...]}`)                             |
| `/add_person`             | `POST`   | -                                                                                                  | 新規人物カードを追加し、更新後の `persons` を返却                                           |
| `/remove_person`          | `POST`   | JSON: `{"person_id": str}`                                                                         | 指定人物を削除（0人になる場合は自動でデフォルト1人を作成）                                  |
| `/update_person_name`     | `POST`   | JSON: `{"person_id": str, "name": str}`                                                            | 指定人物の表示名を更新                                                                      |
| `/upload_target`          | `POST`   | `multipart/form-data` (`person_id`, `face_image[]`)                                                | 指定人物に複数枚の顔画像を登録・特徴量抽出                                                  |
| `/boost_from_video`       | `POST`   | JSON: `{"person_id": str}`                                                                         | 動画内をサンプリングし、登録顔に類似する本人顔を最大5枚自動追加                             |
| `/add_current_frame_face` | `POST`   | JSON: `{"person_id": str}`                                                                         | 現在のプレビューフレームの顔を抽出し、指定人物へ追加                                        |
| `/remove_target`          | `POST`   | JSON: `{"person_id": str, "entry_id": str}`                                                        | 人物内の特定顔サムネイルを削除                                                              |
| `/upload_video`           | `POST`   | `multipart/form-data` (`video`)                                                                    | MP4動画を保存し、`feature_cache` を初期化。`total_frames`, `fps` を返却                     |
| `/video_feed`             | `GET`    | Query: `?t=<timestamp>`                                                                            | `multipart/x-mixed-replace` 形式の MJPEG 映像ストリーム                                     |
| `/control`                | `POST`   | JSON: `{"action": "seek"\|"toggle_pause", "frame"?: int}`                                          | 再生位置のシークまたは一時停止／再開の切り替え                                              |
| `/progress`               | `GET`    | -                                                                                                  | `current_frame`, `total_frames`, `paused`, `detected_persons` を返却（250ms間隔ポーリング） |
| `/update_settings`        | `POST`   | JSON: `action_mode`, `mosaic_target_mode`, `mosaic_scale`, `threshold`, `interp_gap`, `show_debug` | 各種パラメータのリアルタイム更新                                                            |
| `/export_video`           | `POST`   | -                                                                                                  | バックグラウンドスレッドで動画全体の並列解析・書き出しを開始                                |
| `/export_status`          | `GET`    | -                                                                                                  | 書き出しの `status`, `progress`, `message` を返却                                           |
| `/download_video`         | `GET`    | -                                                                                                  | 完成した MP4 ファイルを添付ダウンロード（`send_file`）として返却                            |

---

## 7. 改修・機能追加を行う際の重要制約（AI 向けガイドライン）

別の生成 AI に本コードの修正を依頼する際は、以下の実装制約を必ず維持させてください。

1. **Windows におけるマルチプロセス制約 (`ProcessPoolExecutor`)**
   - Windows 環境では `multiprocessing` が `spawn` 方式で起動されるため、ワーカー関数 `analyze_video_chunk()` は必ず**モジュールのトップレベル**に定義する必要があります（関数内ネストや `lambda` は `PicklingError` になります）。
   - `cv2.FaceDetectorYN` や `cv2.FaceRecognizerSF` のインスタンスは C++ バインディングであり pickle 化できないため、プロセス間で渡さずワーカー関数内で都度 `create()` してください。
2. **推論精度・画質の維持**
   - `analyze_video_chunk` および `generate_frames` では、顔認識精度を担保するため**フレームを縮小せず元解像度のまま**推論を行っています。高速化のために推論用フレームをリサイズする変更は、ユーザーから明示的な許可がない限り行わないでください。
3. **キャッシュの整合性**
   - `state["feature_cache"]` は「モデルの生出力（顔座標と 128 次元ベクトル）」のみを保持しており、閾値 (`threshold`) やモード (`action_mode`, `mosaic_target_mode`) に依存する判定結果はキャッシュしていません。そのため、**新しい動画がアップロードされた時のみキャッシュを破棄**し、設定変更時はキャッシュを破棄せず再評価のみ行う設計を維持してください。
4. **日本語描画の互換性**
   - OpenCV 標準の `cv2.putText` は日本語を描画できません（`???` に文字化けします）。映像上に人物名を描画する際は `draw_boxes_and_labels()` のように `Pillow` (`ImageDraw`) へ変換して描画する方式を維持してください。
5. **ポップアップ通知のスパム防止**
   - フロントエンドの `startProgressPolling()` では 250ms ごとに `/progress` をポーリングしています。毎フレーム通知が出続けるのを防ぐため、人物 ID ごとのクールダウン制御（`POPUP_COOLDOWN_MS = 4000`）をフロントエンド側で維持してください。

---

## 8. 既知の制限事項と今後の拡張候補

- **単一セッション設計**: サーバー側のグローバル辞書 `state` と固定ファイルパス（`uploads/temp_video.mp4`）を使用しているため、複数ブラウザタブや複数ユーザーからの同時アクセスには対応していません（マルチユーザー化する場合はセッション ID ごとのディレクトリと状態管理テーブルへの分離が必要）。
- **拡張アイデア**:
  - **人物ごとの個別アクション設定**: 「人物 A はモザイク、人物 B は通知のみ、未登録者はモザイク」といった人物単位のルール設定。
  - **シークバー上の検出タイムライン表示**: `feature_cache` を利用した登場シーンのヒートマップ表示とシーンジャンプ機能。
  - **モザイク以外のエフェクト**: ガウスぼかし（`cv2.GaussianBlur`）やカスタム PNG 画像（スタンプ）のオーバーレイ対応。
