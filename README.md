# Hongguo Video Translator

FastAPI server tải video Hongguo/direct MP4/M3U8, nhận dạng và forced-align lời thoại bằng WhisperX, tùy chọn speaker diarization bằng pyannote, dịch sang tiếng Việt, burn subtitle và tạo bản thuyết minh VieNeu-TTS chạy local. Metadata được lưu trong SQLite; một worker thread xử lý đúng một job tại một thời điểm để bảo vệ GPU.

## Cài đặt trên Vast.ai Ubuntu

```bash
apt update && apt install -y ffmpeg git curl
cd /workspace/server
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
nano .env
uvicorn app.main:app --host 0.0.0.0 --port 8000
```

Hoặc sau khi cài dependency: `./run.sh`. Chỉ cần điền API key của provider sẽ dùng. Model Whisper được tải ở lần dùng đầu tiên và có thể tốn vài GB với model lớn.

> `/workspace` chỉ bền vững nếu instance có host volume. Kiểm tra bằng `vast-capabilities | jq '.instance.workspace_is_volume'`; hãy đồng bộ dữ liệu ra ngoài trước khi recycle/destroy nếu kết quả là `false`.

## Truy cập từ ngoài

Expose port 8000 khi tạo Vast.ai instance rồi mở `http://SERVER_IP:8000`. Vì port direct có thể công khai, nên ưu tiên đặt ứng dụng sau Caddy token auth theo base image, hoặc dùng SSH tunnel riêng tư:

```bash
ssh -p SSH_PORT -L 8000:127.0.0.1:8000 root@SERVER_IP
```

Sau đó mở `http://localhost:8000`. Các port được cấp cố định lúc tạo instance; nếu 8000 không được cấp, dùng một normal port trống hoặc SSH tunnel. Với dịch vụ lâu dài trên Vast base image, tạo supervisor service thay vì chạy Uvicorn trong shell.

## API

```bash
curl -X POST http://127.0.0.1:8000/api/jobs \
  -H 'Content-Type: application/json' \
  -d '{
    "url":"https://example.com/video.mp4",
    "provider":"openai",
    "asr_model":"large-v3",
    "diarize":true,
    "max_speakers":6,
    "source_language_code":"zh",
    "source_language":"Chinese",
    "target_language":"Vietnamese",
    "burn_subtitles":true,
    "dub":false,
    "narrator_mode":true,
    "tts_voice":"Ngọc Linh",
    "original_audio_volume":0.0
  }'
```

- `GET /api/jobs` — job gần đây
- `GET /api/jobs/{id}` — trạng thái và output hiện có
- `GET /api/jobs/{id}/files/{filename}` — tải output được phép
- `DELETE /api/jobs/{id}` — xóa job đã done/failed và toàn bộ file

Job đang queued/running không thể bị xóa. Khi server restart, job đang chạy được đưa lại vào hàng đợi. Đường dẫn tải chỉ chấp nhận tên output định trước và không cho traversal.

## Output

Mỗi job nằm trong `data/jobs/{uuid}/`: `source.mp4`, `source.srt`, `vi.srt`, tùy chọn `vi-burned.mp4`, các WAV TTS riêng lẻ và `vi-dubbed.mp4`. SQLite ở `data/jobs.sqlite3`. FFmpeg đặt từng WAV vào timeline bằng `adelay`/`amix` và mux thẳng ra MP4; không tạo WAV timeline khổng lồ trong RAM.

Chế độ thuyết minh dùng một giọng VieNeu-TTS v3 Turbo local với style kể chuyện. Các cue cùng lượt nói được ghép thành câu tự nhiên và đặt lại đúng timeline. `original_audio_volume=0` thay audio gốc; giá trị `0..1` trộn audio gốc nhỏ bên dưới.

## Kiểm tra

```bash
source .venv/bin/activate
pytest -q
python -m compileall app
python -c 'from app.main import app; print(app.title)'
```
