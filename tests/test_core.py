from datetime import timedelta
from pathlib import Path
from types import SimpleNamespace

import pytest
import srt

from app.config import get_settings
from app.jobs import apply_subtitle_review, safe_job_file
from app.subtitle import parse_translation_json, segments_to_subtitles
from app.speech_pipeline import _group_words
from app.tts import _fit_audio_to_window, build_utterances
from app.artifacts import ArtifactManifest, stable_hash
from app.dialogue import build_dialogue_units, repair_fragment_speakers
from app.speech_plan import build_speech_plans, predict_duration_ms
from app.prosody import apply_ai_prosody, plan_prosody
from app.text_normalizer import normalize_spoken_text, vietnamese_integer
from app.source_subtitle_mask import _candidate_boxes
from app.adaptive_subtitle import FONT_NAME, fit_text, generate_adaptive_ass
from app.source_subtitle_mask import SubtitleRegion
from app.semantic_utterance import build_semantic_utterances


def test_srt_timestamp_conversion():
    cues = segments_to_subtitles([SimpleNamespace(start=1.2344, end=3.4567, text=" hello ")])
    assert cues[0].start == timedelta(milliseconds=1234)
    assert cues[0].end == timedelta(milliseconds=3457)
    assert cues[0].content == "hello"


def test_translation_json_parsing_and_validation():
    assert parse_translation_json('```json\n{"translations":[{"id":1,"text":"Xin chào"}]}\n```', [1]) == {1: "Xin chào"}
    assert parse_translation_json('{"translations":{"1":"Xin chào"}}', [1]) == {1: "Xin chào"}
    assert parse_translation_json('{"1":"Xin chào"}', [1]) == {1: "Xin chào"}
    assert parse_translation_json('{"id":1,"text":"Xin"}\n{"id":2,"text":"chào"}', [1, 2]) == {1: "Xin", 2: "chào"}
    assert parse_translation_json('[{"id":1,"translation":"Xin chào"}]', [1]) == {1: "Xin chào"}
    assert parse_translation_json('{"id":1,"text":"Xin chào"}', [1]) == {1: "Xin chào"}
    with pytest.raises(ValueError, match="missing"):
        parse_translation_json('[]', [1])


def test_safe_job_path(monkeypatch, tmp_path: Path):
    settings = get_settings().model_copy(update={"jobs_dir": tmp_path})
    monkeypatch.setattr("app.jobs.get_settings", lambda: settings)
    expected = (tmp_path / "abc" / "source.mp4").resolve()
    assert safe_job_file("abc", "source.mp4") == expected
    with pytest.raises(ValueError):
        safe_job_file("abc", "../jobs.sqlite3")


def test_utterances_join_only_same_speaker():
    import srt
    cues = [
        srt.Subtitle(1, timedelta(seconds=0), timedelta(seconds=1), "Xin chào"),
        srt.Subtitle(2, timedelta(seconds=1.2), timedelta(seconds=2), "anh nhé"),
        srt.Subtitle(3, timedelta(seconds=2.1), timedelta(seconds=3), "Được rồi"),
    ]
    utterances = build_utterances(cues, {1: "A", 2: "A", 3: "B"})
    assert [(item.content, speaker) for item, speaker in utterances] == [("Xin chào anh nhé", "A"), ("Được rồi", "B")]


def test_utterances_remove_translation_boundary_ellipsis():
    import srt
    cues = [
        srt.Subtitle(1, timedelta(seconds=0), timedelta(seconds=5), "Anh sẽ..."),
        srt.Subtitle(2, timedelta(seconds=5.1), timedelta(seconds=9), "làm việc đó."),
    ]
    utterances = build_utterances(cues, {1: "A", 2: "A"})
    assert utterances[0][0].content == "Anh sẽ làm việc đó."


def test_tts_audio_is_fitted_close_to_speech_window(tmp_path):
    from pydub import AudioSegment
    original = AudioSegment.silent(duration=2000, frame_rate=44100)
    fitted = _fit_audio_to_window(original, 1600, tmp_path / "cue.mp3")
    assert abs(len(fitted) - 1600) < 80


def test_tts_audio_is_never_slow_stretched(tmp_path):
    from pydub import AudioSegment
    original = AudioSegment.silent(duration=1200, frame_rate=44100)
    fitted = _fit_audio_to_window(original, 2000, tmp_path / "cue.mp3")
    assert len(fitted) == 1200


def test_utterances_stop_at_complete_sentence():
    import srt
    cues = [
        srt.Subtitle(1, timedelta(seconds=0), timedelta(seconds=2), "Bí mật đã lộ."),
        srt.Subtitle(2, timedelta(seconds=2.1), timedelta(seconds=4), "Cô lập tức bỏ chạy."),
    ]
    utterances = build_utterances(cues, {1: "A", 2: "A"})
    assert [item.content for item, _ in utterances] == ["Bí mật đã lộ.", "Cô lập tức bỏ chạy."]


def test_dialogue_units_respect_speaker_sentence_and_scene_boundaries():
    cues = [
        srt.Subtitle(1, timedelta(seconds=0), timedelta(seconds=1), "Anh có"),
        srt.Subtitle(2, timedelta(seconds=1.1), timedelta(seconds=2), "nhớ em không?"),
        srt.Subtitle(3, timedelta(seconds=2.1), timedelta(seconds=3), "Tất nhiên."),
        srt.Subtitle(4, timedelta(seconds=6), timedelta(seconds=7), "Ba năm sau"),
    ]
    units = build_dialogue_units(cues, {1: "A", 2: "A", 3: "B", 4: "B"})
    assert [(unit.cue_ids, unit.scene_id) for unit in units] == [([1, 2], 1), ([3], 1), ([4], 2)]


def test_short_contiguous_asr_tail_inherits_previous_speaker():
    cues = [
        srt.Subtitle(1, timedelta(0), timedelta(seconds=1), "拔剑的速"),
        srt.Subtitle(2, timedelta(seconds=1), timedelta(seconds=1.4), "度"),
    ]
    assert repair_fragment_speakers(cues, {1: "A", 2: "B"}) == {1: "A", 2: "A"}


def test_vietnamese_spoken_normalization_handles_units():
    assert vietnamese_integer(10) == "mười"
    assert vietnamese_integer(125) == "một trăm hai mươi lăm"
    assert normalize_spoken_text("Căn hộ rộng 10m², giảm 20%.") == "Căn hộ rộng mười mét vuông, giảm hai mươi phần trăm."


def test_speech_plan_keeps_display_text_separate_from_spoken_text():
    utterance = srt.Subtitle(1, timedelta(0), timedelta(seconds=2), "Phòng rộng 10m2.")
    plans = build_speech_plans([(utterance, "A")], 3000)
    assert plans[0].subtitle_text == "Phòng rộng 10m2."
    assert plans[0].spoken_text == "Phòng rộng mười mét vuông."
    assert plans[0].hard_deadline_ms == 3000
    assert predict_duration_ms("Xin chào!") > 0


def test_ai_prosody_accepts_punctuation_but_never_changed_words():
    cue = srt.Subtitle(1, timedelta(0), timedelta(seconds=2), "Anh yêu em")
    plans = build_speech_plans([(cue, "A")], 2500)
    raw = '[{"id":1,"spoken_text":"Anh... yêu em!","emotion":"warm","intensity":0.6,"style":"doc_truyen","pause_before_ms":40,"pause_after_ms":180}]'
    apply_ai_prosody(plans, raw)
    assert plans[0].spoken_text == "Anh... yêu em!"
    assert plans[0].prosody_source == "ai"
    changed = raw.replace("Anh... yêu em!", "Anh rất yêu em!")
    fallback = build_speech_plans([(cue, "A")], 2500)
    apply_ai_prosody(fallback, changed)
    assert fallback[0].spoken_text == "Anh yêu em"
    assert fallback[0].prosody_source == "fallback"


def test_prosody_provider_failure_is_non_fatal():
    cue = srt.Subtitle(1, timedelta(0), timedelta(seconds=2), "Đừng đi!")
    plans = build_speech_plans([(cue, "A")], 2500)
    result, warning = plan_prosody(plans, "deepseek", lambda *_: (_ for _ in ()).throw(RuntimeError("offline")))
    assert result[0].prosody_source == "fallback"
    assert "offline" in warning


def test_semantic_reflow_merges_fragments_and_preserves_every_cue():
    source = [
        srt.Subtitle(1, timedelta(seconds=1), timedelta(seconds=3), "是你们千灵宗灭"),
        srt.Subtitle(2, timedelta(seconds=4), timedelta(seconds=5), "门之"),
        srt.Subtitle(3, timedelta(seconds=8), timedelta(seconds=10), "日仇魁境突破"),
    ]
    translated = [
        srt.Subtitle(1, source[0].start, source[0].end, "Chính các người đã diệt Càn Linh Tông"),
        srt.Subtitle(2, source[1].start, source[1].end, "chúng ta"),
        srt.Subtitle(3, source[2].start, source[2].end, "Cừu Khôi Cảnh đã đột phá"),
    ]
    raw = '{"utterances":[{"cue_ids":[1,2],"text":"Hôm nay là ngày Càn Linh Tông các ngươi bị diệt môn."},{"cue_ids":[3],"text":"Cừu Khôi Cảnh đã đột phá."}]}'
    result, warning = build_semantic_utterances(translated, source, {1: "A", 3: "A"}, "deepseek", lambda *_: raw)
    assert warning is None
    assert [item.cue_ids for item in result] == [[1, 2], [3]]
    assert result[0].text == "Hôm nay là ngày Càn Linh Tông các ngươi bị diệt môn."


def test_semantic_reflow_invalid_ai_uses_safe_fragment_fallback():
    cues = [
        srt.Subtitle(1, timedelta(0), timedelta(seconds=2), "Anh sẽ cứu em"),
        srt.Subtitle(2, timedelta(seconds=2.2), timedelta(seconds=3), "bằng mọi giá"),
    ]
    result, warning = build_semantic_utterances(cues, cues, {1: "A", 2: "A"}, "deepseek", lambda *_: '{"utterances":[]}')
    assert [item.cue_ids for item in result] == [[1, 2]]
    assert result[0].text == "Anh sẽ cứu em bằng mọi giá"
    assert "fallback" in warning


def test_semantic_fallback_joins_ellipsis_continuation():
    cues = [
        srt.Subtitle(1, timedelta(0), timedelta(seconds=2), "Phụ nữ chỉ khiến ta chậm tay..."),
        srt.Subtitle(2, timedelta(seconds=2), timedelta(seconds=3), "khi rút kiếm."),
    ]
    result, _ = build_semantic_utterances(cues, cues, {1: "A", 2: "A"}, "deepseek", lambda *_: 'invalid')
    assert [item.cue_ids for item in result] == [[1, 2]]
    assert result[0].text == "Phụ nữ chỉ khiến ta chậm tay... khi rút kiếm."


def test_artifact_manifest_requires_matching_fingerprint_and_files(tmp_path):
    output = tmp_path / "output.txt"
    output.write_text("ok")
    manifest = ArtifactManifest(tmp_path)
    fingerprint = stable_hash({"input": 1})
    manifest.complete("step", fingerprint, [output])
    assert ArtifactManifest(tmp_path).valid("step", fingerprint, [output])
    assert not ArtifactManifest(tmp_path).valid("step", stable_hash({"input": 2}), [output])
    output.unlink()
    assert not ArtifactManifest(tmp_path).valid("step", fingerprint, [output])


def test_human_review_updates_translation_and_invalidates_rendered_files(monkeypatch, tmp_path):
    from app.subtitle import read_srt, write_srt
    settings = get_settings().model_copy(update={"jobs_dir": tmp_path})
    monkeypatch.setattr("app.jobs.get_settings", lambda: settings)
    directory = tmp_path / "job-1"
    directory.mkdir()
    subtitles = [srt.Subtitle(1, timedelta(0), timedelta(seconds=1), "Bản cũ")]
    write_srt(directory / "vi.srt", subtitles)
    (directory / "vi-final.json").write_text('[{"id":1,"text":"Bản cũ"}]')
    (directory / "vi-dubbed.mp4").write_bytes(b"old")
    manifest = ArtifactManifest(directory)
    manifest.complete("translation", "fingerprint", [directory / "vi.srt", directory / "vi-final.json"])
    manifest.complete("dub", "old", [directory / "vi-dubbed.mp4"])
    apply_subtitle_review("job-1", {1: "Bản sửa tự nhiên hơn"})
    assert read_srt(directory / "vi.srt")[0].content == "Bản sửa tự nhiên hơn"
    assert not (directory / "vi-dubbed.mp4").exists()
    assert "dub" not in ArtifactManifest(directory).data["artifacts"]


def test_source_subtitle_detector_finds_centered_bright_text_but_not_plain_scene():
    import cv2
    import numpy as np
    frame = np.full((720, 1280, 3), 40, dtype=np.uint8)
    cv2.putText(frame, "SOURCE SUBTITLE", (390, 665), cv2.FONT_HERSHEY_SIMPLEX, 1.4, (255, 255, 255), 4, cv2.LINE_AA)
    boxes = _candidate_boxes(frame)
    assert boxes
    assert boxes[0][1] > 580
    assert not _candidate_boxes(np.full((720, 1280, 3), 40, dtype=np.uint8))


def test_adaptive_subtitle_fits_text_and_uses_same_mask_region(tmp_path):
    short = fit_text("Anh về rồi.", 800, 64, 38)
    long = fit_text("Đây là một câu thoại dài cần tự động thu nhỏ cho vừa vùng che phụ đề.", 500, 64, 38)
    assert short.font_size >= long.font_size
    assert 1 <= len(long.lines) <= 2
    cue = srt.Subtitle(1, timedelta(seconds=1), timedelta(seconds=3), "Xin chào Việt Nam")
    region = SubtitleRegion(100, 600, 900, 80, 0.9, 0, 4)
    report = generate_adaptive_ass([cue], [region], 1280, 720, tmp_path / "vi.ass", tmp_path / "layout.json")
    content = (tmp_path / "vi.ass").read_text()
    assert f"Style: Adaptive,{FONT_NAME}" in content
    assert r"\pos(550,640)" in content
    assert r"\p1" in content
    assert report[0]["region"]["width"] == 900
    assert report[0]["background"]["radius"] > 0
    assert report[0]["background"]["width"] >= 900


def test_adaptive_mask_never_shrinks_below_source_bounds(tmp_path):
    cue = srt.Subtitle(1, timedelta(seconds=1), timedelta(seconds=4), "Ngắn")
    region = SubtitleRegion(250, 620, 780, 70, 0.98, 2, 3)
    report = generate_adaptive_ass([cue], [region], 1280, 720, tmp_path / "vi.ass")
    background = report[0]["background"]
    assert background["x"] <= region.x
    assert background["x"] + background["width"] >= region.x + region.width
    assert background["y"] <= region.y
    assert background["y"] + background["height"] >= region.y + region.height
    assert report[0]["render_start"] == 2
    assert report[0]["render_end"] == 3


def test_adaptive_subtitle_normalizes_vietnamese_and_uses_static_font(tmp_path):
    decomposed = "ye\u0302u em"
    cue = srt.Subtitle(1, timedelta(0), timedelta(seconds=1), decomposed)
    region = SubtitleRegion(100, 600, 900, 80, 0.9, 0, 2)
    generate_adaptive_ass([cue], [region], 1280, 720, tmp_path / "vi.ass")
    content = (tmp_path / "vi.ass").read_text(encoding="utf-8")
    assert "yêu em" in content
    assert r"\b600" not in content


def test_word_grouping_does_not_split_on_noisy_character_speakers():
    words = [
        {"word": "马", "start": 0.0, "end": 0.2, "speaker": "A"},
        {"word": "上", "start": 0.2, "end": 0.4, "speaker": "B"},
        {"word": "回", "start": 0.4, "end": 0.6, "speaker": "A"},
        {"word": "来", "start": 0.6, "end": 0.8, "speaker": "A"},
    ]
    cues, speakers = _group_words(words)
    assert [cue.content for cue in cues] == ["马上回来"]
    assert speakers == {1: "A"}


def test_word_grouping_caps_bad_alignment_and_splits_stable_speaker_turn():
    words = [
        {"start": 0.0, "end": 5.0, "word": "你", "speaker": "A"},
        {"start": 5.0, "end": 5.5, "word": "好", "speaker": "B"},
        {"start": 5.5, "end": 6.0, "word": "吗", "speaker": "B"},
    ]
    cues, speakers = _group_words(words)
    assert cues[0].end.total_seconds() == 1.2
    assert [cue.content for cue in cues] == ["你", "好吗"]
    assert speakers == {1: "A", 2: "B"}
