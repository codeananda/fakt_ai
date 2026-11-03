import json
from pathlib import Path

import httpx
import pytest

from fakt_ai import capture as capturing
from fakt_ai import config, ledger, llm
from fakt_ai.capture import OnScreenText, Post, ScreenshotReading
from fakt_ai.models import Capture

_REEL = "https://www.instagram.com/reel/ABC123/"


@pytest.fixture(autouse=True)
def state_dir(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "STATE_DIR", tmp_path)


@pytest.fixture
def keys(mocker):
    return mocker.patch.object(config, "secret", return_value="test-key")


@pytest.fixture
def no_keys(mocker):
    return mocker.patch.object(config, "secret", return_value=None)


def _response(
    method: str = "GET", url: str = "https://example.com", **kwargs
) -> httpx.Response:
    return httpx.Response(
        kwargs.pop("status_code", 200), request=httpx.Request(method, url), **kwargs
    )


@pytest.mark.parametrize(
    ("url", "platform"),
    [
        (_REEL, "instagram"),
        ("https://x.com/a/status/1", "x"),
        ("https://mobile.twitter.com/a/status/1", "x"),
        ("https://fb.watch/abc/", "facebook"),
        ("https://vm.tiktok.com/abc/", "tiktok"),
        ("https://youtu.be/abc", "youtube"),
        ("https://old.reddit.com/r/a/comments/b/", "reddit"),
        ("https://www.netflix.com/title/1", "other"),
        ("https://notinstagram.com/p/1", "other"),
    ],
)
def test_detects_platform_by_whole_domain(url, platform):
    assert capturing._detect_platform(url) == platform


def test_capture_needs_something():
    with pytest.raises(ValueError):
        capturing.capture(text="   ")


def test_text_alone_is_returned():
    found = capturing.capture(text=" Coffee cures colds. ")
    assert found == Capture(platform="text", text="Coffee cures colds.")


def test_unreadable_url_with_nothing_else_fails(mocker):
    mocker.patch.object(capturing, "_first_readable", return_value=Post())
    with pytest.raises(capturing.CaptureFailed):
        capturing.capture(url="https://example.com/a")


def test_selected_text_survives_an_unreadable_url(mocker, no_keys):
    failure = capturing.yt_dlp.utils.DownloadError("login required")
    mocker.patch.object(capturing, "_video_info", side_effect=failure)
    found = capturing.capture(url=_REEL, text="Selected words")
    assert (found.platform, found.url, found.text) == ("instagram", _REEL, "Selected words")


def test_selected_text_already_on_the_page_is_not_repeated(mocker):
    mocker.patch.object(
        capturing, "_first_readable", return_value=Post(text="Long article body")
    )
    found = capturing.capture(url="https://example.com/a", text="article")
    assert found.text == "Long article body"


def test_browser_pages_are_not_fetched():
    found = capturing.capture(url="chrome://settings", text="Selected words")
    assert found.text == "Selected words"


def test_failed_reader_falls_through_to_the_next():
    def broken(url):
        raise httpx.ConnectError("refused")

    post = capturing._first_readable([broken, lambda url: Post(text="found")], "https://a.b")
    assert post.text == "found"


def test_reader_that_finds_nothing_falls_through():
    readers = [lambda url: None, lambda url: Post(), lambda url: Post(media_url="https://a.b")]
    assert capturing._first_readable(readers, "https://a.b").media_url == "https://a.b"


def test_spend_cap_stops_the_capture_rather_than_degrading_it(mocker):
    cap = ledger.SpendCapReached("no budget")
    with pytest.raises(ledger.SpendCapReached):
        capturing._first_readable([mocker.Mock(side_effect=cap, __name__="read")], _REEL)


def test_instagram_falls_back_to_apify_when_ytdlp_needs_a_login(mocker, keys):
    failure = capturing.yt_dlp.utils.DownloadError("login required")
    mocker.patch.object(capturing, "_video_info", side_effect=failure)
    item = {
        "caption": "Reel caption",
        "videoUrl": "https://cdn.example/v.mp4",
        "ownerUsername": "someone",
        "timestamp": "2026-01-02T03:04:05.000Z",
        "videoDuration": 12.5,
    }
    reply = _response("POST", json=[item])
    post_mock = mocker.patch.object(capturing.httpx, "post", return_value=reply)
    watch = mocker.patch.object(capturing, "_watch", return_value=("Spoken", "On screen"))

    found = capturing.capture(url=_REEL)

    assert found.text == "Reel caption"
    assert (found.author, found.transcript, found.image_text) == (
        "someone",
        "Spoken",
        "On screen",
    )
    assert watch.call_args.args[0].media_url == "https://cdn.example/v.mp4"
    assert post_mock.call_args.kwargs["headers"]["Authorization"] == "Bearer test-key"
    assert "test-key" not in post_mock.call_args.args[0]
    assert ledger.summary() == {"apify": 0.02}


def test_missing_apify_key_skips_the_run_without_charging(mocker, no_keys):
    post_mock = mocker.patch.object(capturing.httpx, "post")
    assert capturing._read_instagram(_REEL) is None
    post_mock.assert_not_called()
    assert ledger.summary() == {}


def test_x_embed_yields_plain_tweet_text(mocker):
    html = '<blockquote><p lang="en">Tea &amp; toast<br>daily</p>&mdash; a</blockquote>'
    reply = _response(json={"html": html, "author_name": "a"})
    mocker.patch.object(capturing.httpx, "get", return_value=reply)
    post = capturing._read_x_embed("https://x.com/a/status/1")
    assert (post.text, post.author) == ("Tea & toast\ndaily", "a")


def test_prefers_the_spoken_language_and_manual_captions():
    json3 = {"ext": "json3", "url": "https://manual"}
    info = {
        "language": "de",
        "subtitles": {"live_chat": [{"ext": "json", "url": "x"}], "de-DE": [json3]},
        "automatic_captions": {"de-orig": [{"ext": "json3", "url": "https://auto"}]},
    }
    assert capturing._caption_track(info) == json3


def test_falls_back_to_automatic_vtt_captions():
    vtt = {"ext": "vtt", "url": "https://auto"}
    info = {"automatic_captions": {"en-orig": [{"ext": "srv1", "url": "x"}, vtt]}}
    assert capturing._caption_track(info) == vtt


def test_no_usable_caption_track():
    assert (
        capturing._caption_track({"subtitles": {"fr": [{"ext": "vtt", "url": "x"}]}}) is None
    )


def test_reads_json3_captions():
    body = {
        "events": [
            {"segs": [{"utf8": "Hello"}, {"utf8": " there"}]},
            {},
            {"segs": [{"utf8": "\n"}]},
        ]
    }
    assert capturing._json3_text(json.dumps(body)) == "Hello there"


def test_reads_vtt_captions_without_timings_tags_or_rolling_repeats():
    body = (
        "WEBVTT\n\n1\n00:00:00.000 --> 00:00:02.000\n<c>Hello</c> &amp; hi\n\n"
        "2\n00:00:02.000 --> 00:00:03.000\nHello &amp; hi\n\n"
        "3\n00:00:03.000 --> 00:00:04.000\nworld\n"
    )
    assert capturing._vtt_text(body) == "Hello & hi world"


def test_long_video_with_captions_is_not_downloaded(mocker):
    download = mocker.patch.object(capturing, "_download")
    post = Post(media_url="https://a.b", duration=3600, captions="Spoken words")
    assert capturing._watch(post) == ("Spoken words", "")
    download.assert_not_called()


def test_short_video_is_transcribed_and_its_on_screen_text_read(mocker):
    streams = {"duration": 30.0, "has_video": True, "has_audio": True}
    download = mocker.patch.object(capturing, "_download", return_value=Path("media.mp4"))
    mocker.patch.object(capturing, "_probe", return_value=streams)
    mocker.patch.object(capturing, "_transcribe", return_value="Spoken")
    mocker.patch.object(capturing, "_read_contact_sheet", return_value="Burned in")

    assert capturing._watch(Post(media_url="https://a.b", duration=30)) == (
        "Spoken",
        "Burned in",
    )
    assert download.call_args.args[2] is True


def test_long_video_without_captions_downloads_audio_alone(mocker):
    streams = {"duration": 3600.0, "has_video": False, "has_audio": True}
    download = mocker.patch.object(capturing, "_download", return_value=Path("media.m4a"))
    mocker.patch.object(capturing, "_probe", return_value=streams)
    mocker.patch.object(capturing, "_transcribe", return_value="Spoken")
    sheet = mocker.patch.object(capturing, "_read_contact_sheet")

    assert capturing._watch(Post(media_url="https://a.b", duration=3600)) == ("Spoken", "")
    assert download.call_args.args[2] is False
    sheet.assert_not_called()


def test_failed_transcription_keeps_the_on_screen_text(mocker):
    streams = {"duration": 30.0, "has_video": True, "has_audio": True}
    mocker.patch.object(capturing, "_download", return_value=Path("media.mp4"))
    mocker.patch.object(capturing, "_probe", return_value=streams)
    failure = httpx.ConnectError("refused")
    mocker.patch.object(capturing, "_transcribe", side_effect=failure, __name__="step")
    mocker.patch.object(capturing, "_read_contact_sheet", return_value="Burned in")

    assert capturing._watch(Post(media_url="https://a.b")) == ("", "Burned in")


def test_failed_download_keeps_the_captions(mocker):
    failure = capturing.yt_dlp.utils.DownloadError("gone")
    mocker.patch.object(capturing, "_download", side_effect=failure, __name__="step")
    post = Post(media_url="https://a.b", duration=30, captions="Spoken")
    assert capturing._watch(post) == ("Spoken", "")


def test_small_audio_is_sent_whole(tmp_path):
    audio = tmp_path / "speech.mp3"
    audio.write_bytes(b"x")
    assert capturing._groq_sized_chunks(audio, tmp_path) == [audio]


def test_oversized_audio_is_split_into_twenty_minute_chunks(mocker, tmp_path, monkeypatch):
    monkeypatch.setattr(capturing, "_GROQ_MAX_BYTES", 1)
    audio = tmp_path / "speech.mp3"
    audio.write_bytes(b"xx")

    def split(command):
        (tmp_path / "chunk_001.mp3").touch()
        (tmp_path / "chunk_000.mp3").touch()

    run = mocker.patch.object(capturing, "_run", side_effect=split)
    chunks = capturing._groq_sized_chunks(audio, tmp_path)

    assert [c.name for c in chunks] == ["chunk_000.mp3", "chunk_001.mp3"]
    assert "1200" in run.call_args.args[0]


def test_transcribing_charges_by_the_hour_and_only_once(mocker, tmp_path):
    chunk = tmp_path / "speech.mp3"
    chunk.write_bytes(b"audio")
    mocker.patch.object(capturing, "_probe", return_value={"duration": 1800.0})
    ask = mocker.patch.object(capturing, "_ask_whisper", return_value={"text": "Spoken"})

    assert capturing._transcribe_chunk(chunk, "key") == "Spoken"
    assert capturing._transcribe_chunk(chunk, "key") == "Spoken"

    ask.assert_called_once()
    assert ledger.summary() == {"groq": 0.02}


def test_whisper_output_drops_segments_without_speech():
    reply = {
        "text": "Thanks for watching. Real words",
        "segments": [
            {"text": " Thanks for watching.", "no_speech_prob": 0.95},
            {"text": " Real words", "no_speech_prob": 0.01},
        ],
    }
    assert capturing._spoken_text(reply) == "Real words"


def test_whisper_retries_rate_limits_but_not_bad_requests():
    limited = httpx.HTTPStatusError("429", request=None, response=_response(status_code=429))
    rejected = httpx.HTTPStatusError("400", request=None, response=_response(status_code=400))
    assert capturing._is_transient(limited)
    assert not capturing._is_transient(rejected)


def test_contact_sheet_is_read_with_the_fast_model(mocker, tmp_path):
    run = mocker.patch.object(capturing, "_run")
    ask = mocker.patch.object(llm, "ask", return_value=OnScreenText(text=" Burned in "))

    text = capturing._read_contact_sheet(Path("m.mp4"), {"duration": 30.0}, tmp_path)

    assert text == "Burned in"
    assert "tile=6x2" in " ".join(run.call_args.args[0])
    assert ask.call_args.kwargs["model"] == config.FAST_MODEL
    assert ask.call_args.kwargs["image"] == tmp_path / "contact_sheet.jpg"


def test_reddit_text_post(mocker, keys):
    entry = {"title": "Title", "selftext": "Body", "author": "a", "created_utc": 86400}
    token = _response("POST", json={"access_token": "tok"})
    listing = _response(json=[{"data": {"children": [{"data": entry}]}}])
    mocker.patch.object(capturing.httpx, "post", return_value=token)
    get = mocker.patch.object(capturing.httpx, "get", return_value=listing)

    post = capturing._read_reddit("https://www.reddit.com/r/a/comments/b/c/")

    assert (post.text, post.author, post.published) == ("Title\n\nBody", "a", "1970-01-02")
    assert get.call_args.args[0] == "https://oauth.reddit.com/r/a/comments/b/c.json"
    assert get.call_args.kwargs["headers"]["Authorization"] == "Bearer tok"


def test_reddit_without_keys_uses_the_public_endpoint(mocker, no_keys):
    entry = {"title": "Title"}
    listing = _response(json=[{"data": {"children": [{"data": entry}]}}])
    get = mocker.patch.object(capturing.httpx, "get", return_value=listing)
    capturing._read_reddit("https://www.reddit.com/r/a/comments/b/c/")
    assert get.call_args.args[0] == "https://www.reddit.com/r/a/comments/b/c.json"


def test_reddit_link_post_includes_the_linked_article(mocker, no_keys):
    entry = {"title": "Title", "url_overridden_by_dest": "https://news.example/a"}
    listing = _response(json=[{"data": {"children": [{"data": entry}]}}])
    mocker.patch.object(capturing.httpx, "get", return_value=listing)
    mocker.patch.object(capturing, "_read_article", return_value=Post(text="Article"))
    post = capturing._read_reddit("https://www.reddit.com/r/a/comments/b/c/")
    assert post.text == "Title\n\nArticle"


def test_article_is_extracted_with_metadata(mocker):
    paragraph = "<p>" + "Vitamin D was studied in a large randomised trial. " * 10 + "</p>"
    html = f"<html><head><title>Headline</title></head><body><article>{paragraph}</article></body></html>"
    reply = _response(text=html, headers={"content-type": "text/html"})
    get = mocker.patch.object(capturing.httpx, "get", return_value=reply)

    post = capturing._read_article("https://example.com/a")

    assert post.text.startswith("Headline")
    assert "randomised trial" in post.text
    assert "Mozilla" in get.call_args.kwargs["headers"]["User-Agent"]


def test_thin_page_is_not_treated_as_an_article(mocker):
    reply = _response(text="<html><body><p>Enable JavaScript</p></body></html>")
    mocker.patch.object(capturing.httpx, "get", return_value=reply)
    assert capturing._read_article("https://example.com/a") is None


def test_non_html_is_left_to_exa(mocker):
    reply = _response(content=b"%PDF", headers={"content-type": "application/pdf"})
    mocker.patch.object(capturing.httpx, "get", return_value=reply)
    assert capturing._read_article("https://example.com/a.pdf") is None


def test_exa_reads_the_page_and_charges(mocker, keys):
    result = {"title": "T", "text": "Body", "author": "A", "publishedDate": "2026-01-01"}
    reply = _response("POST", json={"results": [result]})
    post_mock = mocker.patch.object(capturing.httpx, "post", return_value=reply)

    post = capturing._read_with_exa("https://example.com/a")

    assert (post.text, post.author) == ("T\n\nBody", "A")
    assert post_mock.call_args.kwargs["headers"] == {"x-api-key": "test-key"}
    assert ledger.summary() == {"exa": 0.002}


def test_screenshot_fills_in_platform_author_and_text(mocker):
    reading = ScreenshotReading(platform="x", author="a", text="Post words")
    ask = mocker.patch.object(llm, "ask", return_value=reading)

    found = capturing.capture(image="shot.png")

    assert (found.platform, found.author, found.image_text) == ("x", "a", "Post words")
    assert ask.call_args.kwargs["image"].is_absolute()


def test_screenshot_keeps_the_platform_known_from_the_url(mocker):
    mocker.patch.object(capturing, "_first_readable", return_value=Post(text="Caption"))
    reading = ScreenshotReading(platform="other", text="Post words")
    mocker.patch.object(llm, "ask", return_value=reading)

    found = capturing.capture(url=_REEL, image=Path("shot.png"))

    assert (found.platform, found.text, found.image_text) == (
        "instagram",
        "Caption",
        "Post words",
    )


def test_joining_drops_repeats_in_either_order():
    assert capturing._joined("Title", "Title and body", None, "body") == "Title and body"
    assert capturing._joined("One", "", "Two") == "One\n\nTwo"
