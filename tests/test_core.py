from datetime import timedelta
from pathlib import Path
from types import SimpleNamespace

import pytest

from app.config import get_settings
from app.jobs import safe_job_file
from app.subtitle import parse_translation_json, segments_to_subtitles
from app.speech_pipeline import _group_words
from app.tts import _fit_audio_to_window, build_utterances


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
