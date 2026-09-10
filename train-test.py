import json
import numpy as np
from sklearn.ensemble import RandomForestRegressor


SONGS_FILE = "songs.json"
HISTORY_FILE = "play_history.json"


# ========================================
# データ読み込み
# ========================================

with open(SONGS_FILE, "r", encoding="utf-8") as f:
    songs = json.load(f)

with open(HISTORY_FILE, "r", encoding="utf-8") as f:
    history = json.load(f)


print(f"曲数: {len(songs)}")
print(f"履歴数: {len(history)}")


# ========================================
# song_id → 曲データ
# ========================================

song_map = {
    song["id"]: song
    for song in songs
}


# ========================================
# 音響特徴量
# ========================================

def get_features(song):

    required = [
        "loudness",
        "dynamic_range",
        "mfcc",
        "hpcp",
        "spectral_energy",
        "spectral_spread",
        "spectral_flatness",
    ]

    # 必須特徴量が欠損している曲は使用しない
    if any(song.get(key) is None for key in required):
        return None

    # BPMは欠損する場合がある
    #
    # bpm_missing:
    #   0 = BPMあり
    #   1 = BPMなし
    #
    # BPMなしを単純に「0 BPM」として扱わないためのフラグ
    if song.get("bpm") is None:
        bpm = 0
        bpm_missing = 1
    else:
        bpm = song["bpm"]
        bpm_missing = 0

    features = [
        song["duration"],
        bpm,
        bpm_missing,
        song["loudness"],
        song["dynamic_range"],
    ]

    # MFCC 13次元
    features.extend(song["mfcc"])

    # HPCP 36次元
    features.extend(song["hpcp"])

    features.extend([
        song["spectral_energy"],
        song["spectral_spread"],
        song["spectral_flatness"],
    ])

    return features


# ========================================
# 履歴 → 評価値
#
# like       → 1.0
# none       → 0.5
# skip       → 0.3
# dislike    → 0.0
# not_song   → 学習対象外
# ========================================

def get_label(record):

    # 「曲ではない」は学習対象外
    if record.get("not_song", False):
        return None

    # 明示的な低評価
    #
    # Skipと同時に記録されていた場合でも、
    # 明示的なdislikeを優先する。
    if record.get("liked") == "dislike":
        return 0.0

    # 明示的な高評価
    if record.get("liked") == "like":
        return 1.0

    # Skip
    if record.get("skipped", False):
        return 0.3

    # 評価なしで再生された
    if record.get("liked") == "none":
        return 0.5

    # 不明な状態
    return None


# ========================================
# 学習データ作成
# ========================================

X = []
y = []

excluded_history = 0

for record in history:

    song_id = record["song_id"]

    # songs.jsonに存在しない曲
    if song_id not in song_map:
        excluded_history += 1
        continue

    # 評価値を取得
    label = get_label(record)

    if label is None:
        excluded_history += 1
        continue

    # 曲データ
    song = song_map[song_id]

    # 音響特徴量
    features = get_features(song)

    if features is None:
        excluded_history += 1
        continue

    X.append(features)
    y.append(label)


X = np.array(X)
y = np.array(y)


print()
print("========== 学習データ ==========")
print(f"学習データ数: {len(X)}")
print(f"除外データ数: {excluded_history}")

if len(X) == 0:
    raise RuntimeError(
        "学習に使用できるデータがありません"
    )

print(f"特徴量数: {X.shape[1]}")


# ========================================
# 評価値の内訳
# ========================================

print()
print("========== 評価値の内訳 ==========")

print(f"高評価 (1.0): {np.sum(y == 1.0)} 件")
print(f"通常再生 (0.5): {np.sum(y == 0.5)} 件")
print(f"Skip   (0.3): {np.sum(y == 0.3)} 件")
print(f"低評価 (0.0): {np.sum(y == 0.0)} 件")


# ========================================
# Random Forest
# ========================================

model = RandomForestRegressor(
    n_estimators=100,
    random_state=42,
    n_jobs=-1
)


# ========================================
# 学習
# ========================================

model.fit(X, y)

print()
print("学習完了")


# ========================================
# 全曲に対して予測
# ========================================

valid_songs = []
all_X = []

excluded_songs = 0

for song in songs:

    features = get_features(song)

    if features is None:
        excluded_songs += 1
        continue

    valid_songs.append(song)
    all_X.append(features)


all_X = np.array(all_X)


print()
print("========== 予測 ==========")
print(f"予測対象曲数: {len(valid_songs)}")
print(f"予測対象外曲数: {excluded_songs}")


predictions = model.predict(all_X)


# ========================================
# 予測結果
# ========================================

results = []

for song, score in zip(valid_songs, predictions):

    results.append({
        "id": song["id"],
        "name": song["name"],
        "username": song["username"],
        "score": float(score),
    })


# スコアの高い順
results.sort(
    key=lambda x: x["score"],
    reverse=True
)


# ========================================
# 上位20曲表示
# ========================================

print()
print("========== おすすめ上位20曲 ==========")

for i, song in enumerate(results[:20], 1):

    print(
        f"{i:2d}. "
        f"{song['score']:.3f}  "
        f"{song['name']}  "
        f"(ID: {song['id']})"
    )


# ========================================
# 上位10曲を保存
# ========================================

top10 = results[:10]


with open(
    "top10.json",
    "w",
    encoding="utf-8"
) as f:

    json.dump(
        top10,
        f,
        ensure_ascii=False,
        indent=2
    )


print()
print("上位10曲を top10.json に保存しました")
