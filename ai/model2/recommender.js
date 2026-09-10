// ============================================================
// recommender.js
//
// Pythonで生成したRandomForest model.jsonを
// ブラウザ上で実行する推薦エンジン
// ============================================================

const Recommender = (() => {
    let model = null;

    const FEATURE_COUNT = 57;
    const RECENT_HISTORY_SIZE = 10;
    const RANDOM_TOP_N = 20;

    async function loadModel(path = "model.json") {
        const response = await fetch(path);

        if (!response.ok) {
            throw new Error(
                `モデルを読み込めませんでした: ${response.status}`
            );
        }

        model = await response.json();

        if (model.feature_count !== FEATURE_COUNT) {
            throw new Error(
                `特徴量数が一致しません: ` +
                `${model.feature_count} != ${FEATURE_COUNT}`
            );
        }

        console.log(
            `推薦モデル読み込み完了: ${model.n_estimators} trees`
        );
    }

    function getFeatures(song) {
        const bpm = song.bpm;

        const bpmValue =
            bpm === null || bpm === undefined
                ? 0.0
                : Number(bpm);

        const bpmMissing =
            bpm === null || bpm === undefined
                ? 1.0
                : 0.0;

        if (!Array.isArray(song.mfcc)) {
            return null;
        }

        if (!Array.isArray(song.hpcp)) {
            return null;
        }

        const features = [
            Number(song.duration ?? 0),
            bpmValue,
            bpmMissing,
            Number(song.loudness ?? 0),
            Number(song.dynamic_range ?? 0),

            ...song.mfcc.map(Number),
            ...song.hpcp.map(Number),

            Number(song.spectral_energy ?? 0),
            Number(song.spectral_spread ?? 0),
            Number(song.spectral_flatness ?? 0),
        ];

        if (features.length !== FEATURE_COUNT) {
            console.warn(
                "特徴量数が不正:",
                song.id,
                features.length
            );

            return null;
        }

        return features;
    }

    function predictTree(tree, features) {
        let node = 0;

        while (true) {
            const left = tree.children_left[node];
            const right = tree.children_right[node];

            if (left === -1 && right === -1) {
                return tree.value[node];
            }

            const featureIndex = tree.feature[node];
            const threshold = tree.threshold[node];

            if (features[featureIndex] <= threshold) {
                node = left;
            } else {
                node = right;
            }
        }
    }

    function predict(features) {
        if (!model) {
            throw new Error("モデルが読み込まれていません");
        }

        let total = 0;

        for (const tree of model.trees) {
            total += predictTree(tree, features);
        }

        return total / model.trees.length;
    }

    function getRecentHistory(
        history,
        count = RECENT_HISTORY_SIZE
    ) {
        return history
            .filter(record => !record.not_song)
            .slice(-count);
    }

    function getRecentSongs(history, songs) {
        const songMap = new Map(
            songs.map(song => [song.id, song])
        );

        return getRecentHistory(history)
            .map(record => songMap.get(record.song_id))
            .filter(Boolean);
    }

    function buildCurrentState(history, songs) {
        const recentSongs = getRecentSongs(
            history,
            songs
        );

        if (recentSongs.length === 0) {
            return null;
        }

        const featureVectors = recentSongs
            .map(getFeatures)
            .filter(Boolean);

        if (featureVectors.length === 0) {
            return null;
        }

        const dimensions = featureVectors[0].length;

        const state = new Array(dimensions).fill(0);

        let totalWeight = 0;

        for (let i = 0; i < featureVectors.length; i++) {
            const weight = i + 1;

            for (let j = 0; j < dimensions; j++) {
                state[j] +=
                    featureVectors[i][j] * weight;
            }

            totalWeight += weight;
        }

        for (let j = 0; j < dimensions; j++) {
            state[j] /= totalWeight;
        }

        return state;
    }

    function buildCandidates(songs, history) {
        const recentIds = new Set(
            getRecentHistory(history)
                .map(record => record.song_id)
        );

        return songs.filter(song => {
            if (!getFeatures(song)) {
                return false;
            }

            if (song.duration < 5) {
                return false;
            }

            if (recentIds.has(song.id)) {
                return false;
            }

            return true;
        });
    }

    function recommend(songs, history) {
        if (!model) {
            throw new Error("モデルが読み込まれていません");
        }

        const currentState =
            buildCurrentState(history, songs);

        const candidates =
            buildCandidates(songs, history);

        if (candidates.length === 0) {
            return null;
        }

        if (!currentState) {
            return randomChoice(candidates);
        }

        const scored = [];

        for (const song of candidates) {
            const features = getFeatures(song);

            if (!features) {
                continue;
            }

            const score = predict(features);

            scored.push({
                song,
                score,
            });
        }

        scored.sort(
            (a, b) => b.score - a.score
        );

        const top = scored.slice(
            0,
            Math.min(RANDOM_TOP_N, scored.length)
        );

        const selected =
            top[Math.floor(Math.random() * top.length)];

        console.log(
            "推薦:",
            selected.song.name,
            "score:",
            selected.score
        );

        return selected.song;
    }

    function randomChoice(array) {
        return array[
            Math.floor(Math.random() * array.length)
        ];
    }

    return {
        loadModel,
        getFeatures,
        predict,
        recommend,
        buildCurrentState,
        buildCandidates,
    };
})();
