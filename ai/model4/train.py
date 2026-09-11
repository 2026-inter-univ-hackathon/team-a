import json
import math
import sys
from datetime import datetime, timezone
from pathlib import Path

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error


# ============================================================
# Paths
# ============================================================

BASE_DIR = Path(__file__).resolve().parent

SONGS_FILE = BASE_DIR / "songs.json"
HISTORY_FILE = BASE_DIR / "play_history.json"
MODEL_FILE = BASE_DIR / "model.json"


# ============================================================
# Configuration
# ============================================================

FEATURE_COUNT = 57

N_ESTIMATORS = 300
RANDOM_STATE = 42

TRAIN_RATIO = 0.60
VALIDATION_RATIO = 0.20
TEST_RATIO = 0.20

MIN_SAMPLES_LEAF = 2

# 最低限必要な教師データ数
MIN_DATASET_SIZE = 5


# ============================================================
# Utility
# ============================================================

def load_json(path, default=None):
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)

    except (
        FileNotFoundError,
        json.JSONDecodeError,
        OSError,
    ):
        return default


# ============================================================
# Feature validation
# ============================================================

def validate_features(features):
    """
    57次元の特徴量が正しい形式か確認する。
    """

    if not isinstance(features, list):
        return False

    if len(features) != FEATURE_COUNT:
        return False

    for value in features:
        if not isinstance(value, (int, float)):
            return False

        if not math.isfinite(float(value)):
            return False

    return True


# ============================================================
# Label
# ============================================================

def get_label(record):
    """
    再生履歴から教師ラベルを取得する。

    like    -> 1.0
    skip    -> 0.25
    dislike -> 0.0

    not_song / unrated / 未評価 -> 学習対象外
    """

    if record.get("not_song") is True:
        return None

    if record.get("unrated") is True:
        return None

    liked = record.get("liked")

    skipped = (
        record.get("skipped") is True
    )

    if liked == "like":
        return 1.0

    if liked == "dislike":
        return 0.0

    if skipped:
        return 0.25

    return None


# ============================================================
# Timestamp
# ============================================================

def parse_timestamp(record):
    """
    履歴の started_at を datetime に変換する。

    学習時の時系列分割に使用する。
    """

    value = record.get("started_at")

    if not isinstance(value, str):
        return None

    try:
        return datetime.fromisoformat(value)

    except ValueError:
        return None


# ============================================================
# Dataset preparation
# ============================================================

def build_dataset(songs, history):
    """
    songs.json と play_history.json から
    RandomForest 用の教師データを作成する。
    """

    song_map = {}

    for song in songs:
        song_id = song.get("id")

        if song_id is None:
            continue

        song_map[song_id] = song

    dataset = []

    skipped_not_song = 0
    skipped_unrated = 0
    skipped_missing_song = 0
    skipped_invalid_features = 0
    skipped_invalid_timestamp = 0
    skipped_no_label = 0

    for record in history:

        # ----------------------------------------------------
        # not_song
        # ----------------------------------------------------

        if record.get("not_song") is True:
            skipped_not_song += 1
            continue

        # ----------------------------------------------------
        # unrated
        # ----------------------------------------------------

        if record.get("unrated") is True:
            skipped_unrated += 1
            continue

        # ----------------------------------------------------
        # Label
        # ----------------------------------------------------

        label = get_label(record)

        if label is None:
            skipped_no_label += 1
            continue

        # ----------------------------------------------------
        # Song ID
        # ----------------------------------------------------

        song_id = record.get("song_id")

        if song_id not in song_map:
            skipped_missing_song += 1
            continue

        song = song_map[song_id]

        # ----------------------------------------------------
        # Features
        # ----------------------------------------------------

        features = song.get("features")

        if not validate_features(features):
            skipped_invalid_features += 1
            continue

        # ----------------------------------------------------
        # Timestamp
        # ----------------------------------------------------

        timestamp = parse_timestamp(record)

        if timestamp is None:
            skipped_invalid_timestamp += 1
            continue

        # ----------------------------------------------------
        # Dataset entry
        # ----------------------------------------------------

        dataset.append(
            {
                "timestamp": timestamp,
                "features": [
                    float(value)
                    for value in features
                ],
                "label": float(label),
                "song_id": song_id,
            }
        )

    # 時系列順
    dataset.sort(
        key=lambda item: item["timestamp"]
    )

    # --------------------------------------------------------
    # Statistics
    # --------------------------------------------------------

    print("========================================")
    print("Dataset")
    print("========================================")

    print(
        f"history records:           {len(history)}"
    )

    print(
        f"usable teacher data:       {len(dataset)}"
    )

    print(
        f"skipped not_song:          {skipped_not_song}"
    )

    print(
        f"skipped unrated:           {skipped_unrated}"
    )

    print(
        f"skipped missing song:      {skipped_missing_song}"
    )

    print(
        f"skipped invalid features:  {skipped_invalid_features}"
    )

    print(
        f"skipped invalid timestamp: {skipped_invalid_timestamp}"
    )

    print(
        f"skipped no label:          {skipped_no_label}"
    )

    return dataset


# ============================================================
# Dataset split
# ============================================================

def split_dataset(dataset):
    """
    時系列順に

        60% train
        20% validation
        20% test

    に分割する。
    """

    total = len(dataset)

    train_count = int(
        total * TRAIN_RATIO
    )

    validation_count = int(
        total * VALIDATION_RATIO
    )

    test_count = (
        total
        - train_count
        - validation_count
    )

    # 最低1件ずつになるよう調整
    if train_count < 1:
        train_count = 1

    if validation_count < 1:
        validation_count = 1

    if test_count < 1:
        test_count = 1

    # 合計が total を超えた場合
    while (
        train_count
        + validation_count
        + test_count
        > total
    ):
        if train_count > 1:
            train_count -= 1

        elif validation_count > 1:
            validation_count -= 1

        elif test_count > 1:
            test_count -= 1

        else:
            break

    # 合計が total 未満の場合は train に追加
    while (
        train_count
        + validation_count
        + test_count
        < total
    ):
        train_count += 1

    train = dataset[
        :train_count
    ]

    validation_start = train_count

    validation_end = (
        validation_start
        + validation_count
    )

    validation = dataset[
        validation_start:validation_end
    ]

    test = dataset[
        validation_end:
    ]

    print()
    print("========================================")
    print("Split")
    print("========================================")

    print(
        f"train:       {len(train)}"
    )

    print(
        f"validation:  {len(validation)}"
    )

    print(
        f"test:        {len(test)}"
    )

    return train, validation, test


# ============================================================
# Dataset conversion
# ============================================================

def xy(dataset):
    X = [
        item["features"]
        for item in dataset
    ]

    y = [
        item["label"]
        for item in dataset
    ]

    return X, y


# ============================================================
# Model
# ============================================================

def train_model(train):
    """
    RandomForestRegressor を学習する。
    """

    X_train, y_train = xy(train)

    model = RandomForestRegressor(
        n_estimators=N_ESTIMATORS,
        random_state=RANDOM_STATE,
        n_jobs=-1,
        min_samples_leaf=MIN_SAMPLES_LEAF,
    )

    print()
    print("========================================")
    print("Training")
    print("========================================")

    print(
        f"n_estimators:      {N_ESTIMATORS}"
    )

    print(
        f"min_samples_leaf:  {MIN_SAMPLES_LEAF}"
    )

    print(
        f"train samples:     {len(X_train)}"
    )

    model.fit(
        X_train,
        y_train
    )

    return model


# ============================================================
# Evaluation
# ============================================================

def evaluate_model(model, dataset, name):
    """
    指定データセットに対する MAE を計算する。
    """

    if not dataset:
        print(
            f"{name} MAE: N/A"
        )

        return None

    X, y = xy(dataset)

    predictions = model.predict(X)

    mae = mean_absolute_error(
        y,
        predictions
    )

    print(
        f"{name} MAE: {mae:.4f}"
    )

    return float(mae)


# ============================================================
# Tree serialization
# ============================================================

def serialize_tree(estimator):
    """
    sklearn の DecisionTreeRegressor を
    player.py が読める JSON 形式へ変換する。
    """

    tree = estimator.tree_

    def build_node(node_id):
        left_child = tree.children_left[
            node_id
        ]

        right_child = tree.children_right[
            node_id
        ]

        # ----------------------------------------------------
        # Leaf
        # ----------------------------------------------------

        if left_child == right_child:
            value = tree.value[
                node_id
            ][0][0]

            return {
                "value": float(value)
            }

        # ----------------------------------------------------
        # Internal node
        # ----------------------------------------------------

        feature = tree.feature[
            node_id
        ]

        threshold = tree.threshold[
            node_id
        ]

        return {
            "feature": int(feature),
            "threshold": float(threshold),
            "left": build_node(
                left_child
            ),
            "right": build_node(
                right_child
            ),
        }

    return build_node(0)


# ============================================================
# Model export
# ============================================================

def export_model(
    model,
    train_count,
    validation_count,
    test_count,
    train_mae,
    validation_mae,
    test_mae,
):
    """
    RandomForest を model.json に保存する。

    直接 model.json に書き込まず、
    model.json.tmp に書いてから replace() する。

    これにより player.py が再学習途中の
    壊れた JSON を読み込むことを防ぐ。
    """

    trees = []

    print()
    print("========================================")
    print("Export")
    print("========================================")

    print(
        f"serializing {len(model.estimators_)} trees..."
    )

    for estimator in model.estimators_:
        trees.append(
            serialize_tree(estimator)
        )

    model_data = {
        "model_type": "random_forest_regressor",

        "version": 2,

        "feature_count": FEATURE_COUNT,

        "n_estimators": len(
            model.estimators_
        ),

        "random_state": RANDOM_STATE,

        "min_samples_leaf": MIN_SAMPLES_LEAF,

        "label_mapping": {
            "like": 1.0,
            "skip": 0.25,
            "dislike": 0.0,
            "unrated": None,
        },

        "training": {
            "train_count": train_count,
            "validation_count": validation_count,
            "test_count": test_count,
        },

        "metrics": {
            "train_mae": train_mae,
            "validation_mae": validation_mae,
            "test_mae": test_mae,
        },

        "trained_at": (
            datetime.now(
                timezone.utc
            ).isoformat()
        ),

        "trees": trees,
    }

    # --------------------------------------------------------
    # Atomic replacement
    # --------------------------------------------------------

    tmp_file = MODEL_FILE.with_suffix(
        MODEL_FILE.suffix + ".tmp"
    )

    print(
        f"writing temporary model: {tmp_file}"
    )

    with open(
        tmp_file,
        "w",
        encoding="utf-8"
    ) as f:
        json.dump(
            model_data,
            f,
            ensure_ascii=False,
            indent=2
        )

        f.flush()

    print(
        f"replacing model: {MODEL_FILE}"
    )

    tmp_file.replace(
        MODEL_FILE
    )

    print(
        "model.json updated successfully"
    )


# ============================================================
# Main
# ============================================================

def main():
    print("========================================")
    print("Music Recommendation Model Training")
    print("========================================")

    # --------------------------------------------------------
    # Load data
    # --------------------------------------------------------

    songs = load_json(
        SONGS_FILE,
        []
    )

    history = load_json(
        HISTORY_FILE,
        []
    )

    if not isinstance(songs, list):
        print(
            "ERROR: songs.json が配列ではありません",
            file=sys.stderr
        )

        return 1

    if not isinstance(history, list):
        print(
            "ERROR: play_history.json が配列ではありません",
            file=sys.stderr
        )

        return 1

    print(
        f"songs:   {len(songs)}"
    )

    print(
        f"history: {len(history)}"
    )

    # --------------------------------------------------------
    # Build dataset
    # --------------------------------------------------------

    dataset = build_dataset(
        songs,
        history
    )

    if len(dataset) < MIN_DATASET_SIZE:
        print()
        print(
            f"ERROR: 教師データが少なすぎます。"
        )

        print(
            f"必要: {MIN_DATASET_SIZE}"
        )

        print(
            f"現在: {len(dataset)}"
        )

        return 1

    # --------------------------------------------------------
    # Split
    # --------------------------------------------------------

    train, validation, test = (
        split_dataset(dataset)
    )

    if len(train) < 1:
        print(
            "ERROR: train dataset が空です",
            file=sys.stderr
        )

        return 1

    # --------------------------------------------------------
    # Train
    # --------------------------------------------------------

    model = train_model(
        train
    )

    # --------------------------------------------------------
    # Evaluate
    # --------------------------------------------------------

    print()
    print("========================================")
    print("Evaluation")
    print("========================================")

    train_mae = evaluate_model(
        model,
        train,
        "train"
    )

    validation_mae = evaluate_model(
        model,
        validation,
        "validation"
    )

    test_mae = evaluate_model(
        model,
        test,
        "test"
    )

    # --------------------------------------------------------
    # Export
    # --------------------------------------------------------

    export_model(
        model=model,
        train_count=len(train),
        validation_count=len(validation),
        test_count=len(test),
        train_mae=train_mae,
        validation_mae=validation_mae,
        test_mae=test_mae,
    )

    print()
    print("========================================")
    print("Training complete")
    print("========================================")

    print(
        f"teacher data: {len(dataset)}"
    )

    print(
        f"train:        {len(train)}"
    )

    print(
        f"validation:   {len(validation)}"
    )

    print(
        f"test:         {len(test)}"
    )

    print(
        f"model:        {MODEL_FILE}"
    )

    return 0


if __name__ == "__main__":
    sys.exit(
        main()
    )
