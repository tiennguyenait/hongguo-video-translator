# Hongguo Video Translator

FastAPI server tải video Hongguo/direct MP4/M3U8, nhận dạng và forced-align lời thoại bằng WhisperX, tùy chọn speaker diarization bằng pyannote, dịch hai lượt có ngữ cảnh sang tiếng Việt, burn subtitle và tạo bản thuyết minh VieNeu-TTS chạy local. Metadata được lưu trong SQLite; một worker thread xử lý đúng một job tại một thời điểm để bảo vệ GPU.

Pipeline 2.0 có checkpoint nguyên tử, cache TTS theo fingerprint, semantic dialogue units, sửa fragment diarization, text normalization riêng cho cách đọc, speech timing plan, tự phát hiện/che phụ đề gốc, FFmpeg sidechain ducking/loudness và QA tự động. Job bị gián đoạn tiếp tục từ artifact hợp lệ gần nhất.

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
- `GET /api/jobs/{id}/subtitles` — lấy source/translation để review
- `PATCH /api/jobs/{id}/subtitles` — lưu câu sửa và render lại từ checkpoint
- `DELETE /api/jobs/{id}` — xóa job đã done/failed và toàn bộ file

Job đang queued/running không thể bị xóa. Khi server restart, job đang chạy được đưa lại vào hàng đợi. Đường dẫn tải chỉ chấp nhận tên output định trước và không cho traversal.

## Output

Mỗi job nằm trong `data/jobs/{uuid}/`: `source.mp4`, `source.srt`, `vi-draft.json`, `vi-final.json`, `vi.srt`, `dialogue-units.json`, `subtitle-regions.json`, `speech-plan.json`, `tts-timing.json`, `artifacts.json`, `qa-report.json`, tùy chọn `vi-burned.mp4`, các WAV TTS riêng lẻ và `vi-dubbed.mp4`. SQLite ở `data/jobs.sqlite3`. FFmpeg đặt từng WAV vào timeline bằng `adelay`/`amix` và mux thẳng ra MP4; không tạo WAV timeline khổng lồ trong RAM.

`hide_source_subtitles=true` (mặc định) lấy mẫu frame theo timestamp ASR, phát hiện vùng chữ bằng OpenCV, dùng temporal consensus để chọn một vùng ổn định và phủ tối đúng các khoảng có thoại trước khi burn chữ Việt. Kết quả được cache trong `subtitle-regions.json`; confidence thấp được ghi warning trong QA.

Khi có vùng che, server sinh `vi.ass` thay vì đặt thêm nền riêng cho SRT. Chữ Việt dùng Ubuntu Sans SemiBold (fallback DejaVu Sans), được căn giữa trong chính vùng che và tự chọn cỡ chữ/chia tối đa hai dòng theo metric font thật. Cỡ chữ chỉ thay đổi theo bước 2 px để tránh hiệu ứng phóng–thu khó chịu; quyết định layout lưu trong `subtitle-layout.json`.

Chế độ thuyết minh dùng một giọng VieNeu-TTS v3 Turbo local với style kể chuyện. Các cue cùng lượt nói được ghép thành câu tự nhiên và đặt lại đúng timeline. `subtitle_text` được giữ nguyên để hiển thị, còn `spoken_text` chuẩn hóa số/đơn vị cho TTS. `original_audio_volume=0` thay audio gốc; giá trị `0..1` giữ nhạc/hiệu ứng và tự động duck audio gốc khi giọng Việt xuất hiện.

UI cho phép sửa từng câu sau khi job hoàn tất. Khi lưu, server giữ nguyên download/ASR/translation checkpoint, chỉ tạo lại subtitle burn, những clip TTS bị thay đổi, mix và QA.

## Kiểm tra

```bash
source .venv/bin/activate
pytest -q
python -m compileall app
python -c 'from app.main import app; print(app.title)'
```

E2E release cần kiểm tra thêm:

```bash
ffmpeg -v error -i data/jobs/JOB_ID/vi-dubbed.mp4 -f null -
ffprobe -v error -show_entries format=duration:stream=codec_type,codec_name,sample_rate \
  -of json data/jobs/JOB_ID/vi-dubbed.mp4
jq . data/jobs/JOB_ID/qa-report.json
```
