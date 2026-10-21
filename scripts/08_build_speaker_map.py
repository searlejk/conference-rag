"""
Step 8 (custom feature): Speaker Semantic Map
=============================================
Projects every talk's concept vector (from 07_build_talk_concepts.py) from
1,536 dimensions down to 2 with UMAP, groups talks into topic clusters, and
places each speaker at the average meaning of their talks.

No API calls: uses the vectors already saved in scripts/output/talk_vectors.npz.

Usage:
    python scripts/08_build_speaker_map.py

Output:
    map_data.json  — talk/speaker/cluster positions for map.html (public: titles,
                     speakers and 2D coordinates only — no embeddings)
"""

import json
import os
import re
from collections import defaultdict

import numpy as np
import umap
from sklearn.cluster import KMeans

OUT = os.path.join('scripts', 'output')
ALPHA = 0.7  # same blend as talk_concepts in the database
N_CLUSTERS = 9

# Sustainings, audit and statistical reports are procedural, not about a subject
PROCEDURAL = re.compile(r'^(The )?Sustaining of|Auditing Department Report|^Statistical Report', re.I)

# Cluster names chosen by reading each cluster's most distinctive words and titles
CLUSTER_LABELS = {
    0: "God's love, hope, and healing",
    1: 'Commandments and repentance',
    2: 'Peacemaking and unity',
    3: 'Book of Mormon and the Restoration',
    4: 'Trials and the Atonement',
    5: 'Service and missionary work',
    6: 'Youth, parents, and guidance',
    7: 'Testimony of Christ and the Holy Ghost',
    8: 'Temples and priesthood keys',
}

# As sustained in general conference, October 2026
FIRST_PRESIDENCY = ['Dallin H. Oaks', 'Henry B. Eyring', 'D. Todd Christofferson']
TWELVE = ['Dieter F. Uchtdorf', 'David A. Bednar', 'Quentin L. Cook', 'Neil L. Andersen',
          'Ronald A. Rasband', 'Gary E. Stevenson', 'Dale G. Renlund', 'Gerrit W. Gong',
          'Ulisses Soares', 'Patrick Kearon', 'Gérald Caussé', 'Clark G. Gilbert']


def clean_name(speaker):
    name = speaker.replace('\xa0', ' ')
    name = re.sub(r'^(Presented by|By)\s+', '', name)
    return re.sub(r'^(President|Elder|Bishop|Sister|Brother)\s+', '', name).strip()


def unit(v):
    return v / np.linalg.norm(v, axis=-1, keepdims=True)


def main():
    vectors = np.load(os.path.join(OUT, 'talk_vectors.npz'), allow_pickle=True)
    ids = list(vectors['ids'])
    concept = unit(ALPHA * vectors['content'] + (1 - ALPHA) * vectors['full'])

    meta = {}
    with open(os.path.join(OUT, 'sentences.json'), 'r', encoding='utf-8') as f:
        for row in json.load(f):
            meta.setdefault(row['talk_id'], row)

    keep = [n for n, i in enumerate(ids) if not PROCEDURAL.search(meta[i]['title'])]
    ids = [ids[n] for n in keep]
    X = concept[keep]
    print(f"Mapping {len(ids)} talks ({len(vectors['ids']) - len(ids)} procedural talks excluded)")

    clusters = KMeans(N_CLUSTERS, n_init=10, random_state=42).fit_predict(X)
    reducer = umap.UMAP(n_neighbors=15, min_dist=0.15, metric='cosine', random_state=42)
    xy = reducer.fit_transform(X)

    # Speakers: project the AVERAGE of their talk vectors, not the average of 2D points,
    # so their position reflects meaning rather than UMAP's layout quirks
    by_speaker = defaultdict(list)
    for n, i in enumerate(ids):
        by_speaker[clean_name(meta[i]['speaker'])].append(n)
    names = sorted(by_speaker)
    speaker_xy = reducer.transform(unit(np.array([X[by_speaker[s]].mean(axis=0) for s in names])))

    def role(name):
        if name in FIRST_PRESIDENCY:
            return 'First Presidency'
        if name in TWELVE:
            return 'Quorum of the Twelve'
        return 'Other'

    data = {
        'talks': [{
            'id': i,
            'x': round(float(xy[n, 0]), 3), 'y': round(float(xy[n, 1]), 3),
            'title': meta[i]['title'], 'speaker': clean_name(meta[i]['speaker']),
            'year': meta[i]['year'], 'season': meta[i]['season'],
            'url': meta[i]['url'], 'cluster': int(clusters[n]),
        } for n, i in enumerate(ids)],
        'clusters': [{
            'id': c, 'label': CLUSTER_LABELS[c], 'size': int((clusters == c).sum()),
            'x': round(float(xy[clusters == c, 0].mean()), 3), 'y': round(float(xy[clusters == c, 1].mean()), 3),
        } for c in range(N_CLUSTERS)],
        'speakers': [{
            'name': s, 'x': round(float(speaker_xy[k, 0]), 3), 'y': round(float(speaker_xy[k, 1]), 3),
            'talks': len(by_speaker[s]), 'role': role(s),
            'order': (FIRST_PRESIDENCY + TWELVE).index(s) if role(s) != 'Other' else None,
        } for k, s in enumerate(names)],
    }

    missing = [s for s in FIRST_PRESIDENCY + TWELVE if s not in by_speaker]
    with open('map_data.json', 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False)
    print(f"✅ map_data.json: {len(data['talks'])} talks, {len(names)} speakers, {N_CLUSTERS} topics")
    if missing:
        print(f"   ⚠️ Leaders with no talks in this data: {missing}")


if __name__ == '__main__':
    main()
