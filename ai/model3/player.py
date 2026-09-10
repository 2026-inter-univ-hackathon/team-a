import json
import math
import random
import sys
import uuid
from datetime import datetime
from pathlib import Path

from PySide6.QtCore import Qt, QProcess, QTimer, QUrl
from PySide6.QtGui import QFont
from PySide6.QtMultimedia import QAudioOutput, QMediaPlayer
from PySide6.QtWidgets import (
    QApplication,
    QDialog,
    QFileDialog,
    QHBoxLayout,
    QLabel,
    QMessageBox,
    QPushButton,
    QVBoxLayout,
    QWidget,
)


BASE_DIR = Path(__file__).resolve().parent

SONGS_FILE = BASE_DIR / "songs.json"
HISTORY_FILE = BASE_DIR / "play_history.json"
MODEL_FILE = BASE_DIR / "model.json"
NOT_SONG_FILE = BASE_DIR / "not_song_ids.json"

RECENT_HISTORY_SIZE = 10
RANDOM_TOP_N = 20
FEATURE_COUNT = 57

# 新しい教師データがこの数以上増えたら自動再学習
RETRAIN_THRESHOLD = 10

# model.json が存在しない場合でも、
# この数以上の有効な教師データがあれば学習開始
MIN_TRAINING_DATA = 5


# ============================================================
# Utility
# ============================================================

def now_iso():
    return datetime.now().astimezone().isoformat()


def generate_session_id():
    return str(uuid.uuid4())


def load_json(path, default):
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError, OSError):
        return default


def save_json(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)

    tmp_path = path.with_suffix(path.suffix + ".tmp")

    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

    tmp_path.replace(path)


# ============================================================
# Recommender
# ============================================================

class Recommender:
    def __init__(self, model_path):
        self.model_path = model_path
        self.model = None
        self.trees = []

        self.reload()

    # --------------------------------------------------------
    # Model loading
    # --------------------------------------------------------

    def reload(self):
        model = load_json(self.model_path, None)

        if not isinstance(model, dict):
            raise ValueError("model.json が不正です")

        feature_count = model.get("feature_count")

        if feature_count != FEATURE_COUNT:
            raise ValueError(
                f"feature_count が不正です: "
                f"{feature_count} != {FEATURE_COUNT}"
            )

        trees = model.get("trees")

        if not isinstance(trees, list) or not trees:
            raise ValueError("trees が存在しません")

        self.model = model
        self.trees = trees

    # --------------------------------------------------------
    # Tree prediction
    # --------------------------------------------------------

    def predict_tree(self, tree, features):
        node = tree

        while True:
            if "value" in node:
                return float(node["value"])

            feature_index = node.get("feature")
            threshold = node.get("threshold")

            if feature_index is None or threshold is None:
                raise ValueError("決定木ノードが不正です")

            if feature_index < 0 or feature_index >= len(features):
                raise ValueError("feature index が範囲外です")

            if features[feature_index] <= threshold:
                node = node.get("left")
            else:
                node = node.get("right")

            if node is None:
                raise ValueError("決定木の子ノードが存在しません")

    # --------------------------------------------------------
    # Prediction
    # --------------------------------------------------------

    def predict(self, features):
        if not self.trees:
            raise ValueError("モデルが読み込まれていません")

        predictions = []

        for tree in self.trees:
            predictions.append(
                self.predict_tree(tree, features)
            )

        return sum(predictions) / len(predictions)

    # --------------------------------------------------------
    # History
    # --------------------------------------------------------

    def get_recent_history(self, history):
        valid = []

        for record in history:
            if record.get("not_song") is True:
                continue

            if record.get("unrated") is True:
                continue

            liked = record.get("liked")
            skipped = record.get("skipped") is True

            if liked not in ("like", "dislike") and not skipped:
                continue

            valid.append(record)

        valid.sort(
            key=lambda x: x.get("started_at", ""),
            reverse=True
        )

        return valid[:RECENT_HISTORY_SIZE]

    # --------------------------------------------------------
    # Current state
    # --------------------------------------------------------

    def build_current_state(self, history):
        recent = self.get_recent_history(history)

        likes = []
        dislikes = []
        skips = []

        for record in recent:
            liked = record.get("liked")
            skipped = record.get("skipped") is True

            if liked == "like":
                likes.append(record)
            elif liked == "dislike":
                dislikes.append(record)
            elif skipped:
                skips.append(record)

        return {
            "recent": recent,
            "likes": likes,
            "dislikes": dislikes,
            "skips": skips,
        }

    # --------------------------------------------------------
    # Recommendation
    # --------------------------------------------------------

    def recommend(self, songs, history, not_song_ids):
        recent_history = self.get_recent_history(history)

        recent_ids = set()

        for record in recent_history:
            song_id = record.get("song_id")

            if song_id is not None:
                recent_ids.add(song_id)

        candidates = []

        for song in songs:
            song_id = song.get("id")

            if song_id is None:
                continue

            if song_id in not_song_ids:
                continue

            if song_id in recent_ids:
                continue

            try:
                duration = float(song.get("duration", 0))

                if duration < 5:
                    continue
            except (TypeError, ValueError):
                continue

            features = song.get("features")

            if not isinstance(features, list):
                continue

            if len(features) != FEATURE_COUNT:
                continue

            try:
                score = self.predict(features)
            except (TypeError, ValueError, OverflowError):
                continue

            candidates.append(
                {
                    "song": song,
                    "score": score,
                }
            )

        # 履歴除外を解除して再試行
        if not candidates:
            for song in songs:
                song_id = song.get("id")

                if song_id is None:
                    continue

                if song_id in not_song_ids:
                    continue

                try:
                    duration = float(song.get("duration", 0))

                    if duration < 5:
                        continue
                except (TypeError, ValueError):
                    continue

                features = song.get("features")

                if not isinstance(features, list):
                    continue

                if len(features) != FEATURE_COUNT:
                    continue

                try:
                    score = self.predict(features)
                except (TypeError, ValueError, OverflowError):
                    continue

                candidates.append(
                    {
                        "song": song,
                        "score": score,
                    }
                )

        if not candidates:
            return None

        candidates.sort(
            key=lambda x: x["score"],
            reverse=True
        )

        top_n = min(RANDOM_TOP_N, len(candidates))

        return random.choice(candidates[:top_n])["song"]


# ============================================================
# Music Player
# ============================================================

class MusicPlayer(QWidget):

    def __init__(self):
        super().__init__()

        # ----------------------------------------------------
        # Data
        # ----------------------------------------------------

        self.songs = load_json(SONGS_FILE, [])
        self.history = load_json(HISTORY_FILE, [])
        self.not_song_ids = set(
            load_json(NOT_SONG_FILE, [])
        )

        # ----------------------------------------------------
        # Session
        # ----------------------------------------------------

        self.session_id = generate_session_id()

        # ----------------------------------------------------
        # Current song state
        # ----------------------------------------------------

        self.current_song = None

        self.liked = "none"
        self.skipped = False

        self.playback_started_at = None
        self.playback_elapsed_seconds = 0.0
        self.playback_segment_started_at = None

        # ----------------------------------------------------
        # User playback state
        #
        # QMediaPlayer の状態とは独立して管理する。
        #
        # True  = 再生状態
        # False = 停止状態
        #
        # 曲終了や曲変更では勝手に変更しない。
        # ----------------------------------------------------

        self.is_playing = False

        # ----------------------------------------------------
        # Model
        # ----------------------------------------------------

        self.recommender = None

        self.training_process = None
        self.training_in_progress = False

        # 現在使用しているモデルの教師データ数
        self.model_training_count = 0

        self.load_recommender()

        # ----------------------------------------------------
        # Media player
        # ----------------------------------------------------

        self.audio_output = QAudioOutput(self)
        self.audio_output.setVolume(1.0)

        self.media_player = QMediaPlayer(self)
        self.media_player.setAudioOutput(self.audio_output)

        self.media_player.mediaStatusChanged.connect(
            self.on_media_status_changed
        )

        self.media_player.errorOccurred.connect(
            self.on_media_error
        )

        # ----------------------------------------------------
        # UI
        # ----------------------------------------------------

        self.setWindowTitle("部室BGM")
        self.resize(720, 420)

        self.build_ui()

        # ----------------------------------------------------
        # Initial song
        # ----------------------------------------------------

        self.play_next()

        # 起動直後に再学習条件を確認
        QTimer.singleShot(
            1000,
            self.check_auto_retrain
        )

    # ========================================================
    # Model
    # ========================================================

    def load_recommender(self):
        try:
            self.recommender = Recommender(MODEL_FILE)

            self.model_training_count = (
                self.get_model_training_count()
            )

            return True

        except (ValueError, TypeError, OSError, KeyError) as e:
            print(f"[model] モデル読み込み失敗: {e}")

            self.recommender = None
            self.model_training_count = 0

            return False

    def get_model_training_count(self):
        if self.recommender is None:
            return 0

        training = self.recommender.model.get(
            "training",
            {}
        )

        if not isinstance(training, dict):
            return 0

        try:
            return (
                int(training.get("train_count", 0))
                + int(training.get("validation_count", 0))
                + int(training.get("test_count", 0))
            )
        except (TypeError, ValueError):
            return 0

    # ========================================================
    # Training data validation
    # ========================================================

    @staticmethod
    def get_history_label(record):
        """
        train.py と同じラベル判定。
        """

        if record.get("not_song") is True:
            return None

        if record.get("unrated") is True:
            return None

        liked = record.get("liked")
        skipped = record.get("skipped") is True

        if liked == "like":
            return 1.0

        if liked == "dislike":
            return 0.0

        if skipped:
            return 0.25

        return None

    @staticmethod
    def valid_timestamp(record):
        """
        train.py と同様に timestamp が有効か確認。
        """

        timestamp = record.get("started_at")

        if not timestamp:
            return False

        try:
            datetime.fromisoformat(timestamp)
            return True
        except (TypeError, ValueError):
            return False

    @staticmethod
    def valid_song_features(song):
        """
        train.py の教師データとして使用できる
        57次元特徴量か確認する。
        """

        features = song.get("features")

        if not isinstance(features, list):
            return False

        if len(features) != FEATURE_COUNT:
            return False

        for value in features:
            try:
                value = float(value)
            except (TypeError, ValueError):
                return False

            if not math.isfinite(value):
                return False

        return True

    # ========================================================
    # Training data count
    # ========================================================

    def count_trainable_history(self):
        """
        train.py と同じ条件で、実際に教師データとして
        使用できる履歴件数を数える。
        """

        song_map = {}

        for song in self.songs:
            song_id = song.get("id")

            if song_id is not None:
                song_map[song_id] = song

        count = 0

        for record in self.history:
            # ラベル
            label = self.get_history_label(record)

            if label is None:
                continue

            # 曲の存在
            song_id = record.get("song_id")

            song = song_map.get(song_id)

            if song is None:
                continue

            # 特徴量
            if not self.valid_song_features(song):
                continue

            # timestamp
            if not self.valid_timestamp(record):
                continue

            count += 1

        return count

    # ========================================================
    # Automatic retraining
    # ========================================================

    def should_retrain(self):
        """
        自動再学習が必要か判定する。

        教師データが最低件数に達していない場合は、
        絶対に再学習を開始しない。
        """

        if self.training_in_progress:
            return False

        trainable_count = self.count_trainable_history()

        print(
            f"[train] trainable history: "
            f"{trainable_count}"
        )

        # 教師データ不足なら何もしない
        if trainable_count < MIN_TRAINING_DATA:
            return False

        # モデルが存在しない場合
        if self.recommender is None:
            return True

        # モデルの教師データ数より減っている場合
        if trainable_count < self.model_training_count:
            return False

        new_data_count = (
            trainable_count
            - self.model_training_count
        )

        return new_data_count >= RETRAIN_THRESHOLD

    def check_auto_retrain(self):
        """
        自動再学習条件を確認して、必要なら開始する。
        """

        if self.training_in_progress:
            return

        if not self.should_retrain():
            return

        self.start_retraining()

    def start_retraining(self):
        """
        train.py を QProcess でバックグラウンド実行する。
        """

        if self.training_in_progress:
            return

        # 開始直前にも必ず確認する。
        # ここで不足していれば QProcess を起動しない。
        trainable_count = self.count_trainable_history()

        if trainable_count < MIN_TRAINING_DATA:
            print(
                f"[train] 教師データ不足: "
                f"{trainable_count}/{MIN_TRAINING_DATA}"
            )

            self.restore_status()
            return

        train_py = BASE_DIR / "train.py"

        if not train_py.exists():
            print("[train] train.py が存在しません")
            return

        self.training_in_progress = True

        self.update_status(
            "モデルを更新中..."
        )

        print("[train] 自動再学習を開始します")

        process = QProcess(self)

        self.training_process = process

        process.setWorkingDirectory(
            str(BASE_DIR)
        )

        process.setProgram(
            sys.executable
        )

        process.setArguments(
            [str(train_py)]
        )

        process.readyReadStandardOutput.connect(
            self.on_training_stdout
        )

        process.readyReadStandardError.connect(
            self.on_training_stderr
        )

        process.errorOccurred.connect(
            self.on_training_error
        )

        process.finished.connect(
            self.on_training_finished
        )

        process.start()

        if not process.waitForStarted(1000):
            print(
                "[train] train.py の起動に失敗しました"
            )

            self.training_in_progress = False
            self.training_process = None

            self.restore_status()

    def on_training_stdout(self):
        if self.training_process is None:
            return

        data = self.training_process.readAllStandardOutput()

        text = bytes(data).decode(
            "utf-8",
            errors="replace"
        )

        if text:
            print("[train]", text.rstrip())

    def on_training_stderr(self):
        if self.training_process is None:
            return

        data = self.training_process.readAllStandardError()

        text = bytes(data).decode(
            "utf-8",
            errors="replace"
        )

        if text:
            print("[train:stderr]", text.rstrip())

    def on_training_error(self, error):
        print(
            f"[train] QProcess error: {error}"
        )

    def on_training_finished(
        self,
        exit_code,
        exit_status
    ):
        print(
            f"[train] train.py finished: "
            f"exit_code={exit_code}, "
            f"exit_status={exit_status}"
        )

        success = (
            exit_code == 0
            and exit_status == QProcess.ExitStatus.NormalExit
        )

        if success:
            if self.load_recommender():
                print(
                    "[train] 新しいモデルを読み込みました"
                )

                self.update_status(
                    "モデル更新完了"
                )
            else:
                print(
                    "[train] 学習は成功しましたが、"
                    "新しいモデルの読み込みに失敗しました"
                )

                self.restore_status()

        else:
            print(
                "[train] 再学習に失敗しました。"
                "既存モデルを維持します"
            )

            self.restore_status()

        self.training_in_progress = False

        process = self.training_process
        self.training_process = None

        if process is not None:
            process.deleteLater()

        # 失敗時に即座に再実行しない。
        #
        # 履歴追加時の check_auto_retrain() で十分。
        if success:
            QTimer.singleShot(
                100,
                self.check_auto_retrain
            )

    def restore_status(self):
        if self.current_song is None:
            self.update_status("")
            return

        if self.playback_started_at is None:
            self.update_status(
                "再生ボタンを押してください"
            )
        else:
            self.update_status("")

    # ========================================================
    # UI
    # ========================================================

    def build_ui(self):
        layout = QVBoxLayout(self)

        layout.setContentsMargins(
            40,
            30,
            40,
            30
        )

        layout.setSpacing(18)

        # ----------------------------------------------------
        # Title
        # ----------------------------------------------------

        title = QLabel("部室BGM")

        title_font = QFont()
        title_font.setPointSize(24)
        title_font.setBold(True)

        title.setFont(title_font)
        title.setAlignment(Qt.AlignmentFlag.AlignCenter)

        title.setContextMenuPolicy(
            Qt.ContextMenuPolicy.CustomContextMenu
        )

        title.customContextMenuRequested.connect(
            self.open_debug_dialog
        )

        layout.addWidget(title)

        # ----------------------------------------------------
        # Song title
        # ----------------------------------------------------

        self.song_title_label = QLabel("")

        song_font = QFont()
        song_font.setPointSize(22)
        song_font.setBold(True)

        self.song_title_label.setFont(song_font)

        self.song_title_label.setAlignment(
            Qt.AlignmentFlag.AlignCenter
        )

        self.song_title_label.setWordWrap(True)

        layout.addWidget(
            self.song_title_label
        )

        # ----------------------------------------------------
        # Artist
        # ----------------------------------------------------

        self.artist_label = QLabel("")

        artist_font = QFont()
        artist_font.setPointSize(14)

        self.artist_label.setFont(
            artist_font
        )

        self.artist_label.setAlignment(
            Qt.AlignmentFlag.AlignCenter
        )

        layout.addWidget(
            self.artist_label
        )

        # ----------------------------------------------------
        # Status
        # ----------------------------------------------------

        self.status_label = QLabel("")

        status_font = QFont()
        status_font.setPointSize(10)

        self.status_label.setFont(
            status_font
        )

        self.status_label.setAlignment(
            Qt.AlignmentFlag.AlignCenter
        )

        layout.addWidget(
            self.status_label
        )

        # ----------------------------------------------------
        # Progress
        # ----------------------------------------------------

        self.progress_label = QLabel("0:00 / 0:00")

        self.progress_label.setAlignment(
            Qt.AlignmentFlag.AlignCenter
        )

        layout.addWidget(
            self.progress_label
        )

        # ----------------------------------------------------
        # Play / Pause
        # ----------------------------------------------------

        self.play_button = QPushButton("▶ 再生")

        play_font = QFont()
        play_font.setPointSize(18)
        play_font.setBold(True)

        self.play_button.setFont(
            play_font
        )

        self.play_button.setMinimumHeight(60)

        self.play_button.setCheckable(True)

        self.play_button.clicked.connect(
            self.toggle_playback
        )

        layout.addWidget(
            self.play_button
        )

        # ----------------------------------------------------
        # Feedback buttons
        # ----------------------------------------------------

        feedback_layout = QHBoxLayout()

        self.like_button = QPushButton("👍 高評価")
        self.dislike_button = QPushButton("👎 低評価")
        self.skip_button = QPushButton("⏭ Skip")

        self.like_button.clicked.connect(
            self.like_current
        )

        self.dislike_button.clicked.connect(
            self.dislike_current
        )

        self.skip_button.clicked.connect(
            self.skip_current
        )

        feedback_layout.addWidget(
            self.like_button
        )

        feedback_layout.addWidget(
            self.dislike_button
        )

        feedback_layout.addWidget(
            self.skip_button
        )

        layout.addLayout(
            feedback_layout
        )

    # ========================================================
    # UI helpers
    # ========================================================

    def update_status(self, text):
        self.status_label.setText(text)

    def update_song_labels(self):
        if self.current_song is None:
            self.song_title_label.setText("")
            self.artist_label.setText("")
            return

        title = (
            self.current_song.get("name")
            or self.current_song.get("title")
            or "Unknown"
        )

        artist = (
            self.current_song.get("username")
            or self.current_song.get("artist")
            or "Unknown Artist"
        )

        self.song_title_label.setText(
            str(title)
        )

        self.artist_label.setText(
            str(artist)
        )

    # ========================================================
    # Song selection
    # ========================================================

    def select_song(self, song):
        """
        曲を選択する。

        曲を選択しただけでは再生しない。

        重要:
            self.is_playing は変更しない。
            これによって Skip や曲終了時に、
            現在のユーザー再生状態を維持できる。
        """

        self.media_player.stop()

        self.current_song = song

        self.liked = "none"
        self.skipped = False

        self.playback_started_at = None
        self.playback_elapsed_seconds = 0.0
        self.playback_segment_started_at = None

        # ----------------------------------------------------
        # 再生状態は変更しない
        # ----------------------------------------------------

        if self.is_playing:
            self.play_button.setChecked(True)
            self.play_button.setText(
                "⏸ 停止"
            )
        else:
            self.play_button.setChecked(False)
            self.play_button.setText(
                "▶ 再生"
            )

        self.update_song_labels()

        preview_url = song.get("preview_url")

        if preview_url:
            self.media_player.setSource(
                QUrl(str(preview_url))
            )
        else:
            self.media_player.setSource(
                QUrl()
            )

        self.update_progress()

        if self.is_playing:
            self.update_status("")
        else:
            self.update_status(
                "再生ボタンを押してください"
            )

    def play_next(self):
        """
        次の曲を選択する。

        現在の self.is_playing は変更しない。
        """

        song = None

        # ----------------------------------------------------
        # Model recommendation
        # ----------------------------------------------------

        if self.recommender is not None:
            try:
                song = self.recommender.recommend(
                    self.songs,
                    self.history,
                    self.not_song_ids
                )
            except Exception as e:
                print(
                    f"[model] recommendation failed: {e}"
                )

        # ----------------------------------------------------
        # Random fallback
        # ----------------------------------------------------

        if song is None:
            candidates = []

            for item in self.songs:
                song_id = item.get("id")

                if song_id is None:
                    continue

                if song_id in self.not_song_ids:
                    continue

                try:
                    duration = float(
                        item.get("duration", 0)
                    )

                    if duration < 5:
                        continue

                except (TypeError, ValueError):
                    continue

                candidates.append(item)

            if candidates:
                song = random.choice(candidates)

        if song is None:
            self.current_song = None

            self.song_title_label.setText(
                "再生可能な曲がありません"
            )

            self.artist_label.setText("")

            self.is_playing = False

            self.play_button.setChecked(False)
            self.play_button.setText(
                "▶ 再生"
            )

            self.update_status("")

            return

        self.select_song(song)

        # ----------------------------------------------------
        # すでに再生状態だった場合は次曲も自動再生
        # ----------------------------------------------------

        if self.is_playing:
            self.start_playback()

    # ========================================================
    # Playback
    # ========================================================

    def toggle_playback(self):
        """
        再生ボタンの状態をユーザー操作として切り替える。

        QMediaPlayer.playbackState() ではなく、
        self.is_playing を基準にする。
        """

        if self.current_song is None:
            self.is_playing = False
            self.play_button.setChecked(False)
            self.play_button.setText(
                "▶ 再生"
            )
            return

        if self.is_playing:
            self.pause_playback()
        else:
            self.start_playback()

    def start_playback(self):
        """
        再生開始。

        self.is_playing を True にする。
        """

        if self.current_song is None:
            self.is_playing = False

            self.play_button.setChecked(False)
            self.play_button.setText(
                "▶ 再生"
            )

            return

        self.is_playing = True

        # 初回再生
        if self.playback_started_at is None:
            self.playback_started_at = datetime.now()

        # 再生区間開始
        if self.playback_segment_started_at is None:
            self.playback_segment_started_at = datetime.now()

        self.media_player.play()

        self.play_button.setChecked(True)
        self.play_button.setText(
            "⏸ 停止"
        )

        self.update_status("")

    def pause_playback(self):
        """
        再生停止。

        self.is_playing を False にする。
        """

        self.is_playing = False

        self.accumulate_playback_time()

        self.media_player.pause()

        self.play_button.setChecked(False)
        self.play_button.setText(
            "▶ 再生"
        )

    def accumulate_playback_time(self):
        if self.playback_segment_started_at is None:
            return

        now = datetime.now()

        elapsed = (
            now
            - self.playback_segment_started_at
        ).total_seconds()

        if elapsed > 0:
            self.playback_elapsed_seconds += elapsed

        self.playback_segment_started_at = None

    # ========================================================
    # Feedback
    # ========================================================

    def like_current(self):
        if self.current_song is None:
            return

        self.liked = "like"

        print(
            f"[feedback] like: "
            f"{self.current_song.get('id')}"
        )

        self.update_status("👍 高評価")

    def dislike_current(self):
        if self.current_song is None:
            return

        self.liked = "dislike"

        print(
            f"[feedback] dislike: "
            f"{self.current_song.get('id')}"
        )

        self.update_status("👎 低評価")

    def skip_current(self):
        """
        現在の曲をスキップする。

        再生状態は維持する。

        再生中:
            現在曲を保存
            ↓
            次曲へ
            ↓
            自動再生

        停止中:
            現在曲を保存
            ↓
            次曲へ
            ↓
            停止状態のまま
        """

        if self.current_song is None:
            return

        print(
            f"[playback] skip: "
            f"{self.current_song.get('id')}"
        )

        # Skip 前の再生状態を保存
        was_playing = self.is_playing

        self.skipped = True

        # 現在の曲を履歴保存
        self.finish_playback()

        # 次の曲を選択
        #
        # finish_playback() の後も
        # self.is_playing は変更されない。
        self.play_next()

        # play_next() でも self.is_playing は維持される。
        #
        # 念のため Skip 前の状態を明示的に復元する。
        self.is_playing = was_playing

        if was_playing:
            self.start_playback()
        else:
            self.play_button.setChecked(False)
            self.play_button.setText(
                "▶ 再生"
            )

    # ========================================================
    # History
    # ========================================================

    def finish_playback(self):
        """
        現在の曲の履歴を保存する。

        一度も再生していない場合は保存しない。

        注意:
            このメソッドでは self.is_playing を
            変更しない。
        """

        if self.current_song is None:
            return

        if self.playback_started_at is None:
            return

        self.accumulate_playback_time()

        record = {
            "session_id": self.session_id,
            "song_id": self.current_song.get("id"),
            "started_at": (
                self.playback_started_at
                .astimezone()
                .isoformat()
            ),
            "played_seconds": round(
                self.playback_elapsed_seconds,
                3
            ),
            "skipped": self.skipped,
            "liked": self.liked,
            "not_song": (
                self.current_song.get("id")
                in self.not_song_ids
            ),
        }

        self.history.append(record)

        save_json(
            HISTORY_FILE,
            self.history
        )

        print(
            "[history] saved:",
            json.dumps(
                record,
                ensure_ascii=False
            )
        )

        # 履歴追加時にだけ再学習条件を確認
        QTimer.singleShot(
            0,
            self.check_auto_retrain
        )

        self.playback_started_at = None
        self.playback_elapsed_seconds = 0.0
        self.playback_segment_started_at = None

    # ========================================================
    # Media events
    # ========================================================

    def on_media_status_changed(self, status):
        if status == (
            QMediaPlayer.MediaStatus.EndOfMedia
        ):
            print(
                f"[playback] end: "
                f"{self.current_song.get('id')}"
                if self.current_song else 'unknown'
            )

            # ------------------------------------------------
            # 曲終了時に self.is_playing を変更しない
            #
            # 再生中だったなら、そのまま次曲も再生する。
            # ------------------------------------------------

            was_playing = self.is_playing

            self.finish_playback()

            # 次曲を選択
            self.play_next()

            # 念のため終了前の再生状態を維持
            self.is_playing = was_playing

            if was_playing:
                self.start_playback()
            else:
                self.play_button.setChecked(False)
                self.play_button.setText(
                    "▶ 再生"
                )

        self.update_progress()

    def on_media_error(self, error, error_string):
        if error == QMediaPlayer.Error.NoError:
            return

        print(
            f"[media] error: "
            f"{error} {error_string}"
        )

        # メディア自体のエラーなので、
        # 実際に再生できていない状態として扱う。
        self.is_playing = False

        self.play_button.setChecked(False)
        self.play_button.setText(
            "▶ 再生"
        )

        self.update_status(
            "音源を読み込めませんでした"
        )

    # ========================================================
    # Progress
    # ========================================================

    def update_progress(self):
        duration_ms = self.media_player.duration()
        position_ms = self.media_player.position()

        duration = max(
            0,
            duration_ms // 1000
        )

        position = max(
            0,
            position_ms // 1000
        )

        self.progress_label.setText(
            f"{self.format_time(position)} / "
            f"{self.format_time(duration)}"
        )

    @staticmethod
    def format_time(seconds):
        minutes = seconds // 60
        seconds = seconds % 60

        return f"{minutes}:{seconds:02d}"

    # ========================================================
    # Debug dialog
    # ========================================================

    def open_debug_dialog(self):
        dialog = QDialog(self)

        dialog.setWindowTitle(
            "Debug"
        )

        dialog.resize(
            500,
            400
        )

        layout = QVBoxLayout(dialog)

        # ----------------------------------------------------
        # Session
        # ----------------------------------------------------

        session_label = QLabel(
            f"session_id: {self.session_id}"
        )

        session_label.setWordWrap(True)

        layout.addWidget(
            session_label
        )

        # ----------------------------------------------------
        # History count
        # ----------------------------------------------------

        history_label = QLabel(
            f"history count: {len(self.history)}"
        )

        layout.addWidget(
            history_label
        )

        # ----------------------------------------------------
        # Training count
        # ----------------------------------------------------

        training_label = QLabel(
            f"trainable history: "
            f"{self.count_trainable_history()}"
        )

        layout.addWidget(
            training_label
        )

        model_label = QLabel(
            f"model training count: "
            f"{self.model_training_count}"
        )

        layout.addWidget(
            model_label
        )

        # ----------------------------------------------------
        # Training status
        # ----------------------------------------------------

        if self.training_in_progress:
            training_status = "再学習中"
        else:
            training_status = "待機"

        training_status_label = QLabel(
            f"model training status: "
            f"{training_status}"
        )

        layout.addWidget(
            training_status_label
        )

        # ----------------------------------------------------
        # Export history
        # ----------------------------------------------------

        export_button = QPushButton(
            "履歴をエクスポート"
        )

        export_button.clicked.connect(
            lambda: self.export_history(dialog)
        )

        layout.addWidget(
            export_button
        )

        # ----------------------------------------------------
        # Clear history
        # ----------------------------------------------------

        clear_button = QPushButton(
            "ローカル履歴を削除"
        )

        clear_button.clicked.connect(
            lambda: self.clear_history(
                dialog,
                history_label,
                training_label
            )
        )

        layout.addWidget(
            clear_button
        )

        # ----------------------------------------------------
        # Toggle not-song
        # ----------------------------------------------------

        not_song_button = QPushButton(
            "現在の曲を Not Song に切り替え"
        )

        not_song_button.clicked.connect(
            lambda: self.toggle_not_song(
                not_song_button
            )
        )

        layout.addWidget(
            not_song_button
        )

        # ----------------------------------------------------
        # Current song
        # ----------------------------------------------------

        if self.current_song is not None:
            current_id = self.current_song.get(
                "id"
            )

            current_name = self.current_song.get(
                "name",
                ""
            )

            current_song_label = QLabel(
                f"current song: "
                f"{current_id} / "
                f"{current_name}"
            )

            current_song_label.setWordWrap(
                True
            )

            layout.addWidget(
                current_song_label
            )

        # ----------------------------------------------------
        # Close
        # ----------------------------------------------------

        close_button = QPushButton(
            "閉じる"
        )

        close_button.clicked.connect(
            dialog.accept
        )

        layout.addWidget(
            close_button
        )

        dialog.exec()

    def export_history(self, parent):
        path, _ = QFileDialog.getSaveFileName(
            parent,
            "履歴をエクスポート",
            "play_history.json",
            "JSON Files (*.json)"
        )

        if not path:
            return

        try:
            save_json(
                Path(path),
                self.history
            )

            QMessageBox.information(
                parent,
                "完了",
                "履歴をエクスポートしました。"
            )

        except OSError as e:
            QMessageBox.critical(
                parent,
                "エラー",
                f"保存に失敗しました。\n{e}"
            )

    def clear_history(
        self,
        parent,
        history_label,
        training_label
    ):
        result = QMessageBox.question(
            parent,
            "確認",
            "ローカルの再生履歴を削除しますか？"
        )

        if result != QMessageBox.StandardButton.Yes:
            return

        self.history = []

        save_json(
            HISTORY_FILE,
            self.history
        )

        history_label.setText(
            "history count: 0"
        )

        training_label.setText(
            "trainable history: 0"
        )

        print(
            "[history] local history cleared"
        )

    def toggle_not_song(self, button):
        if self.current_song is None:
            return

        song_id = self.current_song.get(
            "id"
        )

        if song_id is None:
            return

        if song_id in self.not_song_ids:
            self.not_song_ids.remove(
                song_id
            )

            button.setText(
                "現在の曲を Not Song に切り替え"
            )

            print(
                f"[not_song] removed: {song_id}"
            )

        else:
            self.not_song_ids.add(
                song_id
            )

            button.setText(
                "現在の曲を Song に戻す"
            )

            print(
                f"[not_song] added: {song_id}"
            )

        save_json(
            NOT_SONG_FILE,
            sorted(self.not_song_ids)
        )

    # ========================================================
    # Close
    # ========================================================

    def closeEvent(self, event):
        """
        アプリ終了時に現在の再生履歴を保存する。
        """

        self.finish_playback()

        if self.training_process is not None:
            print(
                "[train] training process is still running"
            )

        self.media_player.stop()

        event.accept()


# ============================================================
# Main
# ============================================================

def main():
    app = QApplication(sys.argv)

    player = MusicPlayer()
    player.show()

    timer = QTimer(player)
    timer.timeout.connect(
        player.update_progress
    )
    timer.start(500)

    sys.exit(
        app.exec()
    )


if __name__ == "__main__":
    main()
