import json
import os
import sys
import time

import requests


API_URL = "https://freesound.org/apiv2/search/"

API_KEY = os.environ.get("FREESOUND_API_KEY")

if not API_KEY:
    print("ERROR: FREESOUND_API_KEY is not set.", file=sys.stderr)
    print("Example: export FREESOUND_API_KEY='your_api_key'", file=sys.stderr)
    sys.exit(1)


FIELDS = ",".join([
    # 音源メタデータ
    "id",
    "name",
    "username",
    "duration",
    "tags",
    "type",
    "channels",
    "samplerate",
    "filesize",
    "license",
    "url",

    # プレビュー
    "previews",

    # 音響特徴量
    "bpm",
    "loudness",
    "dynamic_range",
    "mfcc",
    "hpcp",
    "spectral_energy",
    "spectral_spread",
    "spectral_flatness",
    "tonality",
])


def fetch_page(page, page_size=150, max_retries=5):
    """
    Freesound APIから1ページ取得する。
    接続エラーが発生した場合はリトライする。
    """

    params = {
        "token": API_KEY,
        "query": "music",
        "filter": 'license:"Creative Commons 0" category:Music',
        "fields": FIELDS,
        "page": page,
        "page_size": page_size,
    }

    for attempt in range(1, max_retries + 1):
        try:
            response = requests.get(
                API_URL,
                params=params,
                timeout=30,
            )

            response.raise_for_status()

            return response.json()

        except requests.RequestException as e:
            print(
                f"  Request failed "
                f"(attempt {attempt}/{max_retries}): {e}"
            )

            if attempt == max_retries:
                raise

            wait = attempt * 5

            print(
                f"  Retrying in {wait} seconds..."
            )

            time.sleep(wait)


def get_preview_url(previews):
    """
    Freesoundのpreviewsから再生用URLを1つ選択する。
    """

    if not previews:
        return None

    candidates = [
        "preview-hq-mp3",
        "preview-hq-ogg",
        "preview-lq-mp3",
        "preview-lq-ogg",
    ]

    for key in candidates:
        url = previews.get(key)

        if url:
            return url

    return None


def normalize_sound(sound):
    """
    APIレスポンスをsongs.json用の形式に変換する。
    """

    return {
        # 音源メタデータ
        "id": sound.get("id"),
        "name": sound.get("name"),
        "username": sound.get("username"),
        "duration": sound.get("duration"),
        "tags": sound.get("tags"),
        "type": sound.get("type"),
        "channels": sound.get("channels"),
        "samplerate": sound.get("samplerate"),
        "filesize": sound.get("filesize"),
        "license": sound.get("license"),
        "url": sound.get("url"),

        # 再生用URL
        "preview_url": get_preview_url(
            sound.get("previews")
        ),

        # 音響特徴量
        "bpm": sound.get("bpm"),
        "loudness": sound.get("loudness"),
        "dynamic_range": sound.get("dynamic_range"),
        "mfcc": sound.get("mfcc"),
        "hpcp": sound.get("hpcp"),
        "spectral_energy": sound.get("spectral_energy"),
        "spectral_spread": sound.get("spectral_spread"),
        "spectral_flatness": sound.get("spectral_flatness"),
        "tonality": sound.get("tonality"),
    }


def save_songs(songs, filename="songs.json"):
    """
    現在取得できている曲をJSONに保存する。
    """

    with open(
        filename,
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            songs,
            f,
            ensure_ascii=False,
            indent=2,
        )


def main():
    all_songs = []

    page = 1
    page_size = 150

    print("Starting Freesound search...")
    print("Filter: CC0 + Music")
    print()

    while True:
        print(f"Fetching page {page}...")

        try:
            data = fetch_page(
                page=page,
                page_size=page_size,
            )

        except requests.RequestException as e:
            print()
            print("ERROR: Failed to fetch page.")
            print(e)
            print()

            # ここまで取得したデータを保存
            if all_songs:
                save_songs(all_songs)

                print(
                    f"Saved {len(all_songs)} songs "
                    f"to songs.json"
                )

            sys.exit(1)

        results = data.get("results", [])

        if not results:
            print("No more results.")
            break

        for sound in results:
            song = normalize_sound(sound)
            all_songs.append(song)

        print(
            f"  Retrieved: {len(results)}"
        )

        print(
            f"  Total:     {len(all_songs)}"
        )

        # 取得した時点で途中保存
        save_songs(all_songs)

        print(
            f"  Saved:     songs.json"
        )

        if not data.get("next"):
            print("Reached the last page.")
            break

        page += 1

        # APIへのアクセス間隔
        time.sleep(0.5)

    print()
    print("Done!")
    print(f"Songs:  {len(all_songs)}")
    print("Output: songs.json")


if __name__ == "__main__":
    main()
