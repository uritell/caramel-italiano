"""Licence records for the clean-pilot sources.

Attribution sentences are the suggested wordings already written in
DATA_LICENSES.md. They are copied here so every manifest row carries the
same text. Nothing in this file is a new legal conclusion.
"""
from __future__ import annotations

CC_BY_4 = "CC BY 4.0"
CC_BY_URL = "https://creativecommons.org/licenses/by/4.0/"

# Official OpenSLR / Facebook host for the Italian MLS tarball, and the
# md5 line published at https://dl.fbaipublicfiles.com/mls/md5sum.txt
MLS_URL = "https://dl.fbaipublicfiles.com/mls/mls_italian.tar.gz"
MLS_MD5 = "dc77f5805aecc7182aa20786032e5dc1"
MLS_BYTES = 15395281399

FLEURS_REPO = "google/fleurs"
DATAHIVE_REPO = "datahiveai/italian-conversations-sample"

SOURCES = {
    "datahive_it_conversations_sample": {
        "id": "datahive_it_conversations_sample",
        "title": "DataHive Italian Conversations sample",
        "url": "https://huggingface.co/datasets/datahiveai/italian-conversations-sample",
        "licence": CC_BY_4,
        "licence_url": CC_BY_URL,
        "attribution": (
            "This work uses the Italian Conversations sample by DataHive, "
            "licensed under CC BY 4.0 "
            "(https://huggingface.co/datasets/datahiveai/italian-conversations-sample)."
        ),
        "layout": "stereo_caller_left_callee_right",
        "needs_account": False,
    },
    "fleurs_it_it": {
        "id": "fleurs_it_it",
        "title": "FLEURS Italian (it_it)",
        "url": "https://huggingface.co/datasets/google/fleurs",
        "licence": CC_BY_4,
        "licence_url": CC_BY_URL,
        "attribution": (
            "This work uses FLEURS (Conneau, A., et al., 2022, Few-shot Learning "
            "Evaluation of Universal Representations of Speech; "
            "https://huggingface.co/datasets/google/fleurs), licensed under CC BY 4.0. "
            "FLEURS is a product of Google."
        ),
        "layout": "mono",
        "needs_account": False,
    },
    "mls_it": {
        "id": "mls_it",
        "title": "Multilingual LibriSpeech, Italian",
        "url": "https://www.openslr.org/94/",
        "download_url": MLS_URL,
        "licence": CC_BY_4,
        "licence_url": CC_BY_URL,
        "attribution": (
            "This work uses the Italian portion of Multilingual LibriSpeech (MLS) "
            "(Pratap, V., et al., 2020, https://www.openslr.org/94/), licensed under "
            "CC BY 4.0. MLS is derived from public-domain LibriVox and Project Gutenberg material."
        ),
        "layout": "mono",
        "needs_account": False,
    },
    "common_voice_it_27": {
        "id": "common_voice_it_27",
        "title": "Common Voice Scripted Speech 27.0, Italian",
        "url": "https://mozilladatacollective.com/datasets/cmu5xagm300dyo1075qdhf20m",
        "licence": "CC0",
        "licence_url": "https://creativecommons.org/publicdomain/zero/1.0/",
        "attribution": (
            "This work uses Mozilla Common Voice Scripted Speech 27.0 (Italian), "
            "a CC0 dataset obtained through Mozilla Data Collective."
        ),
        "layout": "mono",
        "needs_account": True,
    },
}

OPEN_SOURCES = (
    "datahive_it_conversations_sample",
    "fleurs_it_it",
    "mls_it",
)
