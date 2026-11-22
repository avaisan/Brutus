"""Text to speech with AWS Polly."""

from contextlib import closing
from typing import TYPE_CHECKING, cast
from xml.sax.saxutils import escape

if TYPE_CHECKING:
    from mypy_boto3_polly import PollyClient
    from mypy_boto3_polly.literals import VoiceIdType


def build_ssml(text: str, *, rate: int) -> str:
    """Wrap text in SSML with a speaking rate (percent), escaping XML characters."""
    return f"<speak><prosody rate='{rate}%'>{escape(text)}</prosody></speak>"


def synthesize(client: PollyClient, text: str, *, voice: str, rate: int) -> bytes:
    """Speak `text` and return MP3 bytes."""
    response = client.synthesize_speech(
        Text=build_ssml(text, rate=rate),
        TextType="ssml",
        OutputFormat="mp3",
        VoiceId=cast("VoiceIdType", voice),
    )
    with closing(response["AudioStream"]) as stream:
        return stream.read()
