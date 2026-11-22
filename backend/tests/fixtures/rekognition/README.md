# Recorded Rekognition responses

These are real `DetectLabels` (MinConfidence 90) and `DetectFaces` (DEFAULT) responses for each image in `tests/test_data/images/`. They were recorded on 2026-11-22 in eu-central-1, with `ResponseMetadata` removed. `tests/test_recorded.py` replays them through botocore's Stubber, so the counting rules are checked against real output without calling AWS.

To re-record them, for example after Rekognition updates its label model:
1. Run `aws sso login`.
2. Run `uv run task test-integration` to see whether the live counts still hold.
3. Regenerate the JSON files with the same calls as in `brutus.vision.pipeline.analyze`, on images prepared by `normalize(…, max_edge=2048)` and `encode_jpeg`.
