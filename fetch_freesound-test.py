import json
import os
import requests


API_URL = "https://freesound.org/apiv2/search/"
API_KEY = os.environ["FREESOUND_API_KEY"]

FIELDS = ",".join([
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
    "previews",
    "bpm",
    "loudness",
])


params = {
    "token": API_KEY,
    "query": "music",
    "filter": 'license:"Creative Commons 0" category:Music',
    "fields": FIELDS,
    "page": 1,
    "page_size": 10,
}


response = requests.get(
    API_URL,
    params=params,
    timeout=30,
)

response.raise_for_status()

data = response.json()


songs = []

for sound in data["results"]:
    previews = sound.get("previews", {})

    preview_url = (
        previews.get("preview-hq-mp3")
        or previews.get("preview-hq-ogg")
        or previews.get("preview-lq-mp3")
        or previews.get("preview-lq-ogg")
    )

    songs.append({
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
        "preview_url": preview_url,
        "bpm": sound.get("bpm"),
        "loudness": sound.get("loudness"),
    })


with open(
    "songs.json",
    "w",
    encoding="utf-8",
) as f:
    json.dump(
        songs,
        f,
        ensure_ascii=False,
        indent=2,
    )


print(f"取得件数: {len(songs)}")
print("songs.json に保存しました")
