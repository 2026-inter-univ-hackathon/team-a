import json
import math
import random
import subprocess
import sys
import threading
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlparse


BASE_DIR = Path(__file__).resolve().parent

SONGS_FILE = BASE_DIR / "songs.json"
HISTORY_FILE = BASE_DIR / "play_history.json"
MODEL_FILE = BASE_DIR / "model.json"
NOT_SONG_FILE = BASE_DIR / "not_song_ids.json"
PLAYER_FILE = BASE_DIR / "player.html"
TRAIN_FILE = BASE_DIR / "train.py"

HOST = "127.0.0.1"
PORT = 8000

RECENT_HISTORY_SIZE = 10
RANDOM_TOP_N = 20

FEATURE_COUNT = 57

MIN_TRAINING_DATA = 5
RETRAIN_THRESHOLD = 10

lock = threading.RLock()


# ============================================================
# JSON
# ============================================================

def load_json(path, default=None):
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError, OSError):
        return default


def save_json(path, data):
    tmp_path = path.with_suffix(path.suffix + ".tmp")

    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump(
            data,
            f,
            ensure_ascii=False,
            indent=2,
        )
        f.flush()

    tmp_path.replace(path)


# ============================================================
# Features
# ============================================================

def safe_number(value):
    if isinstance(value, (int, float)):
        value = float(value)

        if math.isfinite(value):
            return value

    return 0.0


def build_features(song):
    """
    songs.json の音響特徴量から
    Random Forest 用の57次元ベクトルを作る。

    順番:
      MFCC          13
      HPCP          36
      duration       1
      bpm            1
      loudness       1
      dynamic_range  1
      spectral_energy 1
      spectral_spread 1
      spectral_flatness 1
      tonality       1

    合計 57
    """

    features = []

    # --------------------------------------------------------
    # MFCC: 13
    # --------------------------------------------------------

    mfcc = song.get("mfcc")

    if not isinstance(mfcc, list):
        mfcc = []

    for i in range(13):
        if i < len(mfcc):
            features.append(safe_number(mfcc[i]))
        else:
            features.append(0.0)

    # --------------------------------------------------------
    # HPCP: 36
    # --------------------------------------------------------

    hpcp = song.get("hpcp")

    if not isinstance(hpcp, list):
        hpcp = []

    for i in range(36):
        if i < len(hpcp):
            features.append(safe_number(hpcp[i]))
        else:
            features.append(0.0)

    # --------------------------------------------------------
    # Scalar features: 8
    # --------------------------------------------------------

    scalar_names = [
        "duration",
        "bpm",
        "loudness",
        "dynamic_range",
        "spectral_energy",
        "spectral_spread",
        "spectral_flatness",
        "tonality",
    ]

    for name in scalar_names:
        features.append(
            safe_number(song.get(name))
        )

    if len(features) != FEATURE_COUNT:
        raise ValueError(
            f"feature count mismatch: "
            f"{len(features)} != {FEATURE_COUNT}"
        )

    return features


# ============================================================
# Recommender
# ============================================================

class Recommender:

    def __init__(self):
        self.model = None
        self.load_model()

    # --------------------------------------------------------
    # Model loading
    # --------------------------------------------------------

    def load_model(self):
        model = load_json(MODEL_FILE)

        if not isinstance(model, dict):
            self.model = None
            print("[Recommender] model.json unavailable")
            return

        if model.get("feature_count") != FEATURE_COUNT:
            self.model = None
            print(
                "[Recommender] invalid feature count:",
                model.get("feature_count"),
            )
            return

        trees = model.get("trees")

        if not isinstance(trees, list) or not trees:
            self.model = None
            print("[Recommender] no trees")
            return

        self.model = model

        print(
            "[Recommender] model loaded:",
            len(trees),
            "trees",
        )

    # --------------------------------------------------------
    # Single tree prediction
    # --------------------------------------------------------

    def predict_tree(self, tree, features):
        """
        現在存在する model.json の形式と、
        train.py が生成する再帰形式の両方に対応する。

        形式1: sklearn 配列形式

        {
            "children_left": [...],
            "children_right": [...],
            "feature": [...],
            "threshold": [...],
            "value": [...]
        }

        形式2: 再帰形式

        {
            "feature": 4,
            "threshold": 0.3,
            "left": {...},
            "right": {...}
        }

        葉:

        {
            "value": 0.8
        }
        """

        # ====================================================
        # Format 1
        # sklearn array format
        # ====================================================

        if "children_left" in tree:

            children_left = tree["children_left"]
            children_right = tree["children_right"]
            feature = tree["feature"]
            threshold = tree["threshold"]
            values = tree["value"]

            node = 0

            while True:

                if node < 0 or node >= len(children_left):
                    raise ValueError(
                        f"invalid tree node: {node}"
                    )

                left = children_left[node]
                right = children_right[node]

                # ------------------------------------------------
                # Leaf
                # ------------------------------------------------

                if left == -1 and right == -1:

                    value = values[node]

                    # sklearn の保存形式によっては
                    # [[value]] になっている場合がある
                    while isinstance(value, list):
                        if not value:
                            raise ValueError(
                                f"empty leaf value at node {node}"
                            )

                        value = value[0]

                    return float(value)

                # ------------------------------------------------
                # Internal node
                # ------------------------------------------------

                feature_index = feature[node]

                if feature_index < 0:
                    raise ValueError(
                        f"invalid feature index at node "
                        f"{node}: {feature_index}"
                    )

                if feature_index >= len(features):
                    raise ValueError(
                        f"feature index out of range: "
                        f"{feature_index} >= {len(features)}"
                    )

                if features[feature_index] <= threshold[node]:
                    node = left
                else:
                    node = right

        # ====================================================
        # Format 2
        # Recursive format
        # ====================================================

        node = tree

        while True:

            # Leaf
            if "value" in node:
                return float(node["value"])

            feature_index = node.get("feature")

            if feature_index is None:
                raise ValueError(
                    "tree node has no feature"
                )

            if feature_index < 0:
                raise ValueError(
                    f"invalid feature index: "
                    f"{feature_index}"
                )

            if feature_index >= len(features):
                raise ValueError(
                    f"feature index out of range: "
                    f"{feature_index} >= {len(features)}"
                )

            threshold = node["threshold"]

            if features[feature_index] <= threshold:
                node = node["left"]
            else:
                node = node["right"]

    # --------------------------------------------------------
    # Forest prediction
    # --------------------------------------------------------

    def predict(self, features):

        if self.model is None:
            return None

        if len(features) != self.model.get("feature_count"):
            raise ValueError(
                f"feature count mismatch: "
                f"{len(features)} != "
                f"{self.model.get('feature_count')}"
            )

        trees = self.model.get("trees", [])

        if not trees:
            return None

        predictions = []

        for index, tree in enumerate(trees):

            try:
                prediction = self.predict_tree(
                    tree,
                    features,
                )

                predictions.append(prediction)

            except Exception as e:
                print(
                    f"[Recommender] "
                    f"tree {index} prediction error: {e}",
                    file=sys.stderr,
                )

        if not predictions:
            return None

        return sum(predictions) / len(predictions)


# ============================================================
# Global recommender
# ============================================================

recommender = Recommender()


# ============================================================
# History
# ============================================================

def load_history():
    history = load_json(
        HISTORY_FILE,
        [],
    )

    if not isinstance(history, list):
        return []

    return history


def save_history(history):
    save_json(
        HISTORY_FILE,
        history,
    )


def get_trainable_history(history):
    result = []

    for record in history:

        if not isinstance(record, dict):
            continue

        if record.get("not_song") is True:
            continue

        if record.get("unrated") is True:
            continue

        liked = record.get("liked")
        skipped = record.get("skipped") is True

        if liked in ("like", "dislike"):
            result.append(record)
        elif skipped:
            result.append(record)

    return result


# ============================================================
# Not-song
# ============================================================

def load_not_song_ids():
    data = load_json(
        NOT_SONG_FILE,
        [],
    )

    if not isinstance(data, list):
        return set()

    return {
        item
        for item in data
        if isinstance(item, int)
    }


def save_not_song_ids(ids):
    save_json(
        NOT_SONG_FILE,
        sorted(ids),
    )


# ============================================================
# Recommendation
# ============================================================

def recommend_song():

    songs = load_json(
        SONGS_FILE,
        [],
    )

    if not isinstance(songs, list):
        return None

    history = load_history()

    not_song_ids = load_not_song_ids()

    # --------------------------------------------------------
    # 最近再生した曲
    # --------------------------------------------------------

    recent_ids = set()

    valid_history = []

    for record in history:

        if not isinstance(record, dict):
            continue

        if record.get("not_song") is True:
            continue

        if record.get("unrated") is True:
            continue

        if record.get("song_id") is None:
            continue

        if (
            record.get("liked") in
            ("like", "dislike")
            or record.get("skipped") is True
        ):
            valid_history.append(record)

    for record in valid_history[-RECENT_HISTORY_SIZE:]:
        recent_ids.add(record["song_id"])

    # --------------------------------------------------------
    # Candidate
    # --------------------------------------------------------

    candidates = []

    for song in songs:

        if not isinstance(song, dict):
            continue

        song_id = song.get("id")

        if song_id is None:
            continue

        # not-song
        if song_id in not_song_ids:
            continue

        # 最近再生
        if song_id in recent_ids:
            continue

        # 短すぎる音源
        duration = safe_number(
            song.get("duration")
        )

        if duration < 5:
            continue

        # ----------------------------------------------------
        # Prediction
        # ----------------------------------------------------

        try:
            features = build_features(song)

            score = recommender.predict(
                features
            )

        except Exception as e:
            print(
                "[Recommender] prediction error:",
                song_id,
                e,
                file=sys.stderr,
            )
            continue

        if score is None:
            continue

        candidates.append(
            (
                float(score),
                song,
            )
        )

    # --------------------------------------------------------
    # 最近除外で候補がなくなった場合
    # --------------------------------------------------------

    if not candidates:

        print(
            "[Recommender] "
            "no candidates after recent exclusion"
        )

        for song in songs:

            if not isinstance(song, dict):
                continue

            song_id = song.get("id")

            if song_id is None:
                continue

            if song_id in not_song_ids:
                continue

            duration = safe_number(
                song.get("duration")
            )

            if duration < 5:
                continue

            try:
                features = build_features(song)

                score = recommender.predict(
                    features
                )

            except Exception as e:
                print(
                    "[Recommender] prediction error:",
                    song_id,
                    e,
                    file=sys.stderr,
                )
                continue

            if score is None:
                continue

            candidates.append(
                (
                    float(score),
                    song,
                )
            )

    if not candidates:
        return None

    # --------------------------------------------------------
    # 高スコア順
    # --------------------------------------------------------

    candidates.sort(
        key=lambda item: item[0],
        reverse=True,
    )

    top_n = candidates[
        :min(RANDOM_TOP_N, len(candidates))
    ]

    # 完全固定推薦にせず、
    # 上位候補からランダムに選ぶ
    score, song = random.choice(top_n)

    result = dict(song)

    result["recommendation_score"] = score

    return result


# ============================================================
# Training
# ============================================================

training_lock = threading.Lock()

training_state = {
    "running": False,
    "started_at": None,
    "finished_at": None,
    "return_code": None,
}


def run_training():

    if not training_lock.acquire(blocking=False):
        return False

    try:

        training_state["running"] = True
        training_state["started_at"] = (
            datetime.now(timezone.utc).isoformat()
        )
        training_state["finished_at"] = None
        training_state["return_code"] = None

        print("[Training] started")

        result = subprocess.run(
            [
                sys.executable,
                str(TRAIN_FILE),
            ],
            cwd=str(BASE_DIR),
        )

        training_state["return_code"] = (
            result.returncode
        )

        training_state["finished_at"] = (
            datetime.now(timezone.utc).isoformat()
        )

        # 新しいモデルをロード
        if result.returncode == 0:
            recommender.load_model()

        print(
            "[Training] finished:",
            result.returncode,
        )

    except Exception as e:

        print(
            "[Training] error:",
            e,
            file=sys.stderr,
        )

        training_state["return_code"] = -1

        training_state["finished_at"] = (
            datetime.now(timezone.utc).isoformat()
        )

    finally:

        training_state["running"] = False
        training_lock.release()

    return True


def start_training():

    if training_state["running"]:
        return False

    thread = threading.Thread(
        target=run_training,
        daemon=True,
    )

    thread.start()

    return True


# ============================================================
# HTTP Handler
# ============================================================

class RequestHandler(BaseHTTPRequestHandler):

    server_version = "MusicRoomServer/1.0"

    # --------------------------------------------------------
    # Common
    # --------------------------------------------------------

    def send_json(
        self,
        data,
        status=200,
    ):

        body = json.dumps(
            data,
            ensure_ascii=False,
        ).encode("utf-8")

        self.send_response(status)

        self.send_header(
            "Content-Type",
            "application/json; charset=utf-8",
        )

        self.send_header(
            "Content-Length",
            str(len(body)),
        )

        self.send_header(
            "Access-Control-Allow-Origin",
            "*",
        )

        self.send_header(
            "Access-Control-Allow-Methods",
            "GET, POST, OPTIONS",
        )

        self.send_header(
            "Access-Control-Allow-Headers",
            "Content-Type",
        )

        self.end_headers()

        self.wfile.write(body)

    # --------------------------------------------------------

    def send_text(
        self,
        data,
        content_type,
        status=200,
    ):

        body = data.encode("utf-8")

        self.send_response(status)

        self.send_header(
            "Content-Type",
            content_type,
        )

        self.send_header(
            "Content-Length",
            str(len(body)),
        )

        self.send_header(
            "Access-Control-Allow-Origin",
            "*",
        )

        self.end_headers()

        self.wfile.write(body)

    # --------------------------------------------------------

    def read_json_body(self):

        length = int(
            self.headers.get(
                "Content-Length",
                "0",
            )
        )

        if length <= 0:
            return {}

        body = self.rfile.read(length)

        try:
            return json.loads(
                body.decode("utf-8")
            )
        except json.JSONDecodeError:
            return {}

    # --------------------------------------------------------
    # OPTIONS
    # --------------------------------------------------------

    def do_OPTIONS(self):

        self.send_response(204)

        self.send_header(
            "Access-Control-Allow-Origin",
            "*",
        )

        self.send_header(
            "Access-Control-Allow-Methods",
            "GET, POST, OPTIONS",
        )

        self.send_header(
            "Access-Control-Allow-Headers",
            "Content-Type",
        )

        self.end_headers()

    # --------------------------------------------------------
    # GET
    # --------------------------------------------------------

    def do_GET(self):

        parsed = urlparse(self.path)

        path = parsed.path

        try:

            # ------------------------------------------------
            # /
            # ------------------------------------------------

            if path == "/":

                if not PLAYER_FILE.exists():
                    self.send_text(
                        "player.html not found",
                        "text/plain; charset=utf-8",
                        404,
                    )
                    return

                with open(
                    PLAYER_FILE,
                    "r",
                    encoding="utf-8",
                ) as f:
                    html = f.read()

                self.send_text(
                    html,
                    "text/html; charset=utf-8",
                )

                return

            # ------------------------------------------------
            # /api/songs
            # ------------------------------------------------

            if path == "/api/songs":

                songs = load_json(
                    SONGS_FILE,
                    [],
                )

                if not isinstance(songs, list):
                    songs = []

                self.send_json(
                    songs
                )

                return

            # ------------------------------------------------
            # /api/recommend
            # ------------------------------------------------

            if path == "/api/recommend":

                with lock:
                    song = recommend_song()

                if song is None:

                    self.send_json(
                        {
                            "error":
                                "no recommendable song",
                            "status": 404,
                        },
                        404,
                    )

                    return

                self.send_json(
                    song
                )

                return

            # ------------------------------------------------
            # /api/history
            # ------------------------------------------------

            if path == "/api/history":

                with lock:
                    history = load_history()

                self.send_json(
                    history
                )

                return

            # ------------------------------------------------
            # /api/not-song
            # ------------------------------------------------

            if path == "/api/not-song":

                ids = load_not_song_ids()

                self.send_json(
                    sorted(ids)
                )

                return

            # ------------------------------------------------
            # /api/status
            # ------------------------------------------------

            if path == "/api/status":

                songs = load_json(
                    SONGS_FILE,
                    [],
                )

                history = load_history()

                trainable = (
                    get_trainable_history(
                        history
                    )
                )

                model_training_count = 0

                if recommender.model:

                    training = recommender.model.get(
                        "training",
                        {},
                    )

                    if isinstance(
                        training,
                        dict,
                    ):
                        model_training_count = int(
                            training.get(
                                "train_count",
                                0,
                            )
                        )

                self.send_json(
                    {
                        "songs": (
                            len(songs)
                            if isinstance(
                                songs,
                                list,
                            )
                            else 0
                        ),
                        "history": len(history),
                        "trainable_history":
                            len(trainable),
                        "model_training_count":
                            model_training_count,
                        "model_loaded":
                            recommender.model
                            is not None,
                        "training":
                            training_state["running"],
                    }
                )

                return

            # ------------------------------------------------
            # /api/train
            # ------------------------------------------------

            if path == "/api/train":

                started = start_training()

                if started:

                    self.send_json(
                        {
                            "status":
                                "training started"
                        }
                    )

                else:

                    self.send_json(
                        {
                            "status":
                                "training already running"
                        },
                        409,
                    )

                return

            # ------------------------------------------------
            # /music/*
            # ------------------------------------------------

            if path.startswith("/music/"):

                relative = path[
                    len("/music/"):
                ]

                # Path traversal 対策
                file_path = (
                    BASE_DIR /
                    "music" /
                    relative
                ).resolve()

                music_dir = (
                    BASE_DIR / "music"
                ).resolve()

                if (
                    music_dir not in
                    file_path.parents
                ):
                    self.send_text(
                        "Forbidden",
                        "text/plain; charset=utf-8",
                        403,
                    )
                    return

                if not file_path.is_file():

                    self.send_text(
                        "Not Found",
                        "text/plain; charset=utf-8",
                        404,
                    )
                    return

                # 簡易的なMIME判定
                suffix = (
                    file_path.suffix.lower()
                )

                mime_types = {
                    ".mp3": "audio/mpeg",
                    ".wav": "audio/wav",
                    ".ogg": "audio/ogg",
                    ".m4a": "audio/mp4",
                    ".aac": "audio/aac",
                    ".flac": "audio/flac",
                }

                content_type = mime_types.get(
                    suffix,
                    "application/octet-stream",
                )

                file_size = file_path.stat().st_size

                self.send_response(200)

                self.send_header(
                    "Content-Type",
                    content_type,
                )

                self.send_header(
                    "Content-Length",
                    str(file_size),
                )

                self.send_header(
                    "Accept-Ranges",
                    "bytes",
                )

                self.send_header(
                    "Access-Control-Allow-Origin",
                    "*",
                )

                self.end_headers()

                with open(
                    file_path,
                    "rb",
                ) as f:

                    while True:

                        chunk = f.read(
                            1024 * 1024
                        )

                        if not chunk:
                            break

                        self.wfile.write(
                            chunk
                        )

                return

            # ------------------------------------------------
            # 404
            # ------------------------------------------------

            self.send_json(
                {
                    "error": "not found",
                    "path": path,
                },
                404,
            )

        except Exception as e:

            print(
                "[HTTP] GET error:",
                e,
                file=sys.stderr,
            )

            self.send_json(
                {
                    "error": str(e),
                },
                500,
            )

    # --------------------------------------------------------
    # POST
    # --------------------------------------------------------

    def do_POST(self):

        parsed = urlparse(self.path)

        path = parsed.path

        try:

            data = self.read_json_body()

            # ------------------------------------------------
            # /api/history
            # ------------------------------------------------

            if path == "/api/history":

                song_id = data.get(
                    "song_id"
                )

                if song_id is None:

                    self.send_json(
                        {
                            "error":
                                "song_id required"
                        },
                        400,
                    )

                    return

                record = {
                    "session_id":
                        data.get(
                            "session_id"
                        ),
                    "song_id":
                        song_id,
                    "started_at":
                        data.get(
                            "started_at"
                        ),
                    "played_seconds":
                        safe_number(
                            data.get(
                                "played_seconds"
                            )
                        ),
                    "skipped":
                        data.get(
                            "skipped",
                            False,
                        ) is True,
                    "liked":
                        data.get(
                            "liked",
                            "none",
                        ),
                    "not_song":
                        data.get(
                            "not_song",
                            False,
                        ) is True,
                }

                with lock:

                    history = load_history()

                    history.append(
                        record
                    )

                    save_history(
                        history
                    )

                self.send_json(
                    {
                        "status": "ok",
                        "record": record,
                    }
                )

                return

            # ------------------------------------------------
            # /api/not-song
            # ------------------------------------------------

            if path == "/api/not-song":

                song_id = data.get(
                    "song_id"
                )

                if song_id is None:

                    self.send_json(
                        {
                            "error":
                                "song_id required"
                        },
                        400,
                    )

                    return

                with lock:

                    ids = load_not_song_ids()

                    ids.add(
                        int(song_id)
                    )

                    save_not_song_ids(
                        ids
                    )

                    # 履歴にも記録
                    history = load_history()

                    history.append(
                        {
                            "session_id":
                                data.get(
                                    "session_id"
                                ),
                            "song_id":
                                int(song_id),
                            "started_at":
                                data.get(
                                    "started_at"
                                ),
                            "played_seconds":
                                safe_number(
                                    data.get(
                                        "played_seconds"
                                    )
                                ),
                            "skipped": False,
                            "liked": "none",
                            "not_song": True,
                        }
                    )

                    save_history(
                        history
                    )

                self.send_json(
                    {
                        "status": "ok",
                        "song_id":
                            int(song_id),
                    }
                )

                return

            # ------------------------------------------------
            # /api/train
            # ------------------------------------------------

            if path == "/api/train":

                started = start_training()

                if started:

                    self.send_json(
                        {
                            "status":
                                "training started"
                        }
                    )

                else:

                    self.send_json(
                        {
                            "status":
                                "training already running"
                        },
                        409,
                    )

                return

            # ------------------------------------------------
            # 404
            # ------------------------------------------------

            self.send_json(
                {
                    "error": "not found",
                    "path": path,
                },
                404,
            )

        except Exception as e:

            print(
                "[HTTP] POST error:",
                e,
                file=sys.stderr,
            )

            self.send_json(
                {
                    "error": str(e),
                },
                500,
            )

    # --------------------------------------------------------

    def log_message(
        self,
        format,
        *args,
    ):

        print(
            "[HTTP]",
            format % args,
        )


# ============================================================
# Main
# ============================================================

def main():

    print(
        "========================================"
    )
    print(
        "Music Room Server"
    )
    print(
        "========================================"
    )

    print(
        f"Listening on http://{HOST}:{PORT}/"
    )

    print(
        f"Songs: {SONGS_FILE}"
    )

    print(
        f"History: {HISTORY_FILE}"
    )

    print(
        f"Model: {MODEL_FILE}"
    )

    print()

    server = ThreadingHTTPServer(
        (HOST, PORT),
        RequestHandler,
    )

    try:

        server.serve_forever()

    except KeyboardInterrupt:

        print()
        print(
            "Shutting down..."
        )

    finally:

        server.server_close()


if __name__ == "__main__":
    main()
