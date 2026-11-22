import io

from PIL import Image

from brutus.vision.imaging import REKOGNITION_MAX_BYTES, encode_jpeg, normalize
from tests.images import jpeg_bytes, noisy_jpeg_bytes


class TestNormalize:
    def test_given_exif_orientation_6_when_normalized_then_upright(self) -> None:
        image = normalize(jpeg_bytes(200, 100, orientation=6), max_edge=4096)

        assert image.size == (100, 200)

    def test_given_large_image_when_normalized_then_long_edge_limited(self) -> None:
        image = normalize(jpeg_bytes(4000, 1000), max_edge=2048)

        assert image.size == (2048, 512)


class TestEncodeJpeg:
    def test_given_large_noisy_image_when_encoded_then_within_rekognition_limit(
        self,
    ) -> None:
        image = normalize(noisy_jpeg_bytes(3000, 3000), max_edge=3000)

        data = encode_jpeg(image)

        assert len(data) <= REKOGNITION_MAX_BYTES
        assert Image.open(io.BytesIO(data)).format == "JPEG"
