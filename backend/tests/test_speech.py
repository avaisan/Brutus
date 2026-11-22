import io

from botocore.response import StreamingBody
from botocore.stub import Stubber
from mypy_boto3_polly import PollyClient

from brutus.speech import build_ssml, synthesize


class TestBuildSsml:
    def test_given_text_with_xml_characters_when_built_then_escaped(self) -> None:
        assert build_ssml("Fish & <chips>", rate=85) == (
            "<speak><prosody rate='85%'>Fish &amp; &lt;chips&gt;</prosody></speak>"
        )


class TestSynthesize:
    def test_given_text_when_synthesized_then_polly_called_with_ssml_and_bytes_returned(
        self, polly: PollyClient, polly_stub: Stubber
    ) -> None:
        audio = b"ID3-fake-mp3"
        polly_stub.add_response(
            "synthesize_speech",
            {
                "AudioStream": StreamingBody(io.BytesIO(audio), len(audio)),
                "ContentType": "audio/mpeg",
            },
            {
                "Text": "<speak><prosody rate='85%'>Hello</prosody></speak>",
                "TextType": "ssml",
                "OutputFormat": "mp3",
                "VoiceId": "Matthew",
            },
        )

        assert synthesize(polly, "Hello", voice="Matthew", rate=85) == audio
