import json
from pathlib import Path

import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import train_test_split


BASE_DIR = Path(__file__).resolve().parent

SONGS_FILE = BASE_DIR / "songs.json"
HISTORY_FILE = BASE_DIR / "play_history.json"
MODEL_FILE = BASE_DIR / "model.json"


# ============================================================
# 設定
# ============================================================

N_ESTIMATORS = 100
RANDOM_STATE = 42

RECENT_HISTORY_SIZE = 10


# ============================================================
# JSON
# ============================================================

def load_json(path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


# ============================================================
# 特徴量
# 57 dimensions
# ============================================================

def get_features(song):
    bpm = song.get("bpm")

    if bpm is None:
        bpm_value = 0.0
        bpm_missing = 1.0
    else:
        bpm_value = float(bpm)
        bpm_missing = 0.0

    mfcc = song.get("mfcc")
    hpcp = song.get("hpcp")

    if mfcc is None or hpcp is None:
        return None

    return [
        float(song.get("duration", 0.0)),
        bpm_value,
        bpm_missing,
        float(song.get("loudness", 0.0)),
        float(song.get("dynamic_range", 0.0)),
        *[float(x) for x in mfcc],
        *[float(x) for x in hpcp],
        float(song.get("spectral_energy", 0.0)),
        float(song.get("spectral_spread", 0.0)),
        float(song.get("spectral_flatness", 0.0)),
    ]


# ============================================================
# ラベル
# ============================================================

def get_label(record):
    """
    行動を数値化する。

    like     -> 1.0
    none     -> 0.5
    skip     -> 0.3
    dislike  -> 0.0
    not_song -> 除外
    """

    if record.get("not_song", False):
        return None

    if record.get("liked") == "dislike":
        return 0.0

    if record.get("liked") == "like":
        return 1.0

    if record.get("skipped", False):
        return 0.3

    if record.get("liked") == "none":
        return 0.5

    return None


# ============================================================
# データセット作成
# ============================================================

def build_dataset(songs, history):
    song_map = {
        song["id"]: song
        for song in songs
    }

    X = []
    y = []

    excluded = 0

    for record in history:
        label = get_label(record)

        if label is None:
            excluded += 1
            continue

        song = song_map.get(record.get("song_id"))

        if song is None:
            excluded += 1
            continue

        features = get_features(song)

        if features is None:
            excluded += 1
            continue

        X.append(features)
        y.append(label)

    return np.array(X, dtype=np.float64), np.array(y, dtype=np.float64), excluded


# ============================================================
# RandomForest → JSON
# ============================================================

def export_model(model, feature_count):
    """
    sklearnのRandomForestを
    JavaScriptから実行できるJSONへ変換する。
    """

    trees = []

    for estimator in model.estimators_:
        tree = estimator.tree_

        trees.append({
            "children_left": tree.children_left.tolist(),
            "children_right": tree.children_right.tolist(),
            "feature": tree.feature.tolist(),
            "threshold": tree.threshold.tolist(),
            "value": [
                float(v[0][0])
                for v in tree.value
            ],
        })

    model_data = {
        "model_type": "random_forest_regressor",
        "version": 1,

        "feature_count": feature_count,

        "n_estimators": len(model.estimators_),

        "random_state": RANDOM_STATE,

        "trees": trees,
    }

    with open(MODEL_FILE, "w", encoding="utf-8") as f:
        json.dump(
            model_data,
            f,
            ensure_ascii=False,
            separators=(",", ":"),
        )

    print(f"モデルを {MODEL_FILE.name} に保存しました")


# ============================================================
# メイン
# ============================================================

def main():
    songs = load_json(SONGS_FILE)
    history = load_json(HISTORY_FILE)

    print(f"曲数: {len(songs)}")
    print(f"履歴数: {len(history)}")

    X, y, excluded = build_dataset(
        songs,
        history,
    )

    print()
    print("========== 学習データ ==========")
    print(f"学習データ数: {len(X)}")
    print(f"除外データ数: {excluded}")
    print(f"特徴量数: {X.shape[1] if len(X) else 0}")

    if len(X) < 2:
        print("学習データが不足しています")
        return

    print()
    print("========== 評価値の内訳 ==========")
    print(f"高評価 (1.0): {np.sum(y == 1.0)} 件")
    print(f"通常再生 (0.5): {np.sum(y == 0.5)} 件")
    print(f"Skip   (0.3): {np.sum(y == 0.3)} 件")
    print(f"低評価 (0.0): {np.sum(y == 0.0)} 件")

    # --------------------------------------------------------
    # train / test split
    # --------------------------------------------------------

    if len(X) >= 10:
        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=0.2,
            random_state=RANDOM_STATE,
        )
    else:
        X_train = X
        y_train = y
        X_test = X
        y_test = y

    # --------------------------------------------------------
    # 学習
    # --------------------------------------------------------

    model = RandomForestRegressor(
        n_estimators=N_ESTIMATORS,
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )

    model.fit(X_train, y_train)

    print()
    print("学習完了")

    # --------------------------------------------------------
    # 評価
    # --------------------------------------------------------

    predictions = model.predict(X_test)

    mae = mean_absolute_error(
        y_test,
        predictions,
    )

    rmse = np.sqrt(
        mean_squared_error(
            y_test,
            predictions,
        )
    )

    print()
    print("========== モデル評価 ==========")
    print(f"MAE : {mae:.4f}")
    print(f"RMSE: {rmse:.4f}")

    # --------------------------------------------------------
    # モデル保存
    # --------------------------------------------------------

    export_model(
        model,
        X.shape[1],
    )


if __name__ == "__main__":
    main()
