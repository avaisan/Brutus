# Brutus

Brutus is an AI robot: a Raspberry Pi, a camera and a speaker. It takes pictures, analyzes them with AWS Rekognition, shows the results in a browser and speaks through AWS Polly. Read the workspace `../CLAUDE.md` first; this file adds repo-specific guidance.

## Layout and stack

There are three independent **uv** projects, each requiring Python 3.15 or newer (`.python-version` = 3.15). Each has its own `pyproject.toml`, `uv.lock` and `.env`, and they share no code.

| Folder | What it is | Port |
|---|---|---|
| `backend/` | FastAPI app, package `src/brutus` (built with uv_build). It analyzes each uploaded picture once and stores the result. | 8000 |
| `frontend/` | FastAPI + Jinja2 + htmx, package `src/brutus_frontend`. It renders the latest analysis and polls `/panel` every 3 s. | 3000 |
| `raspberrypi_client/` | `brutus_pi.py`, a single module of plain functions: camera (`rpicam-still`), audio (pygame-ce), HTTP (requests). | none |
| `scripts/` | Bash helpers for the Pi (`start_fe_be.sh`, `start_client.sh`, `stop.sh`). | |

- Tooling: a shared root `ruff.toml` for lint and format, mypy `--strict` in each service, and pre-commit running the standard hooks plus ruff. There is no Poetry, black, isort or flake8.
- AWS: SSO profile set through `AWS_PROFILE` and `AWS_REGION` in `backend/.env`. The permissions it needs are in `brutus-iam-role.yaml` (DetectLabels, DetectFaces, CompareFaces, SynthesizeSpeech). If you add an AWS call, update that file.
- Deployment target: a Pi 3B on **64-bit** Raspberry Pi OS. Python 3.15 wheels for Pillow and pygame-ce are published for aarch64 only.

## Design rules (backend)

The backend follows **functional core, imperative shell**. Keep it that way:
- `brutus/vision/models.py` holds frozen pydantic models (`ConfigDict(frozen=True)`) and tuples, not lists. There are no classes with behavior.
- `geometry`, `parse`, `objects`, `people`, `imaging` and `annotate` are pure: no I/O, no AWS, no globals. `annotate` draws on a copy.
- Only `vision/pipeline.py` (Rekognition calls), `store.py` (filesystem), `speech.py` (Polly) and `api.py` / `cli.py` (entry points) do I/O.
- AWS clients are **passed in** (`create_app(settings, rekognition=..., polly=...)`); they are never created at module level. Settings come from `brutus.settings.Settings`, and the paths default to absolute paths under `backend/`, so the working directory doesn't matter.
- Errors raise. The API maps `ClientError`/`BotoCoreError` to 502. Don't return `{"error": ...}` dicts.

### Counting rules (the reason for the refactor)
- People = instances of the `Person` label. Each face is assigned to a person one-to-one, by the share of the face inside the person box (≥ 0.5). A face with no person box becomes a person.
- Identities: CompareFaces against each `reference_library/<Name>.jpg`, tied to detected faces by IoU ≥ 0.5. Each name goes to at most one face.
- Animals and objects: person-related labels are excluded. Animals are recognized by the "Animals and Pets" category or an `Animal` parent. A parent label on the same box as its child (IoU ≥ 0.8) is dropped, so Dog/Pet/Animal becomes Dog.
- People, animals and objects are each numbered from 1, left to right. Captions are `P1 Kirk`, `A1 Dog`, `O1 Chair`.
- AWS calls per picture: 1 DetectLabels + 1 DetectFaces (`ALL` only at `detailed`) + N CompareFaces (none at `basic` or when there are no faces).
- Gender is deliberately **not** reported.
- Thresholds live in `Thresholds` and can be overridden with env vars, for example `THRESHOLDS__SIMILARITY=80`.

## API (backend)

`GET /health` · `POST /images?detail=` (upload JPEG → analysis, 201) · `POST /analyses?detail=` (re-analyze the stored picture) · `GET /analyses/latest` (ETag/304) · `GET /images/latest/annotated` · `POST /speech?text=` (MP3, at most 1500 characters).

The browser never calls the backend: the frontend proxies `/image/{analysis_id}`. The backend has no CORS middleware, and endpoints take no path parameters.

## Commands

Run these inside each service folder:

```bash
uv sync
uv run task dev        # backend/frontend: uvicorn --factory with --reload on 127.0.0.1
uv run task start      # 0.0.0.0 (client: runs the terminal menu)
uv run task test       # unit tests (backend excludes -m integration)
uv run task lint       # ruff check + format --check
uv run task format
uv run task check      # mypy --strict
```

- Backend extras: `uv run task test-integration` calls real Rekognition (needs SSO and costs a little). `uv run brutus-analyze IMG --detail detailed --out annotated.jpg` analyzes a file without a Pi.
- Repo-wide: `pre-commit run --all-files`.

## Testing

- pytest with BDD names: `test_given_…_when_…_then_…`, grouped in `Test*` classes.
- The backend stubs AWS with `botocore.stub.Stubber` on real boto3 clients (fixtures in `tests/conftest.py`), which checks request parameters and response shapes against the service model. Rekognition responses are built in `tests/rekognition_fixtures.py`, and JPEGs in `tests/images.py`.
- The frontend stands in for the backend with an `httpx2.MockTransport` passed to `create_app(transport=...)`.
- The Pi client mocks `subprocess.run`, uses `requests-mock` for HTTP, and patches `play_audio`.
- The test contract and spec are in `../.github/agent-sessions/refactor/python-315-functional/`.

## Raspberry Pi: camera and audio

- Install `rpicam-apps` and enable the camera in `raspi-config`. Install Python with `uv python install 3.15`.
- `capture()`:
  1. Counts down (`COUNTDOWN_SECONDS`).
  2. Runs `rpicam-still -n -o <tmp>` (argument list, `check=True`).
  3. Rotates by `CAMERA_ROTATION` (degrees counter-clockwise; the default -90 means clockwise) and saves to `input/latest.jpg`.
- `upload()` POSTs to `/images?detail=$DETAIL`, and `describe()` prints the "I see …" summary.
- Audio: `play_audio()` uses `pygame.mixer.music` with a file path or a `BytesIO` (speech is played from memory). `audio/by_your_command.mp3` plays at startup, `introduction.mp3` for option 2 and `laugh.mp3` for option 3.
- To troubleshoot: `rpicam-still -o test.jpg`, `speaker-test -t wav`, `alsamixer`.

## Helper scripts

Run them from the repo root on the Pi. They need bash, `ss`, `curl` and `pkill`.
- `start_fe_be.sh`: aborts if 8000 or 3000 is in use, starts the backend, waits for `/health` (15 × 3 s), then starts the frontend. PID files go in `/tmp/brutus/`, and Ctrl-C stops both.
- `start_client.sh`: runs the client in the foreground and writes its PID file.
- `stop.sh`: stops the services by PID file, with `pkill -f` as a fallback.

## Logs

- Logging is **configuration, not code**. Each service has a `logging.json` (`dictConfig` format, using python-json-logger's `pythonjsonlogger.json.JsonFormatter`). Don't add logging setup functions or copy a logging module between services; modules only call `logging.getLogger(__name__)`.
- Backend and frontend: uvicorn loads the file with `--log-config logging.json` (in the `start`/`dev` tasks). Tests and the `brutus-analyze` CLI configure no logging.
- Pi client: `main()` changes into the client folder and calls `logging.config.dictConfig` on `logging.json`.
- Files, rotated daily and kept for 14 days:
  - backend: `backend/logs/brutus.log` (the console gets INFO as plain text);
  - frontend: `frontend/logs/frontend.log` (the console gets WARNING);
  - Pi client: `raspberrypi_client/logs/brutus_pi.log` (file only).
- File paths are relative to the service folder. `logs/.gitkeep` keeps each logs folder in git, because the file handler won't create it.
- Each service has `tests/test_logging_config.py`, which checks that its config writes JSON lines.
- Never log secrets. Expected, repeating failures (such as the backend being down while htmx polls) log one line without a traceback.

## Planned (README "Upcoming")

RuuviTag weather (`RUUVITAG_MAC` may already be in a local `.env`), servo arms through an Arduino, Alexa voice commands, and a Lex/Bedrock discussion mode.
