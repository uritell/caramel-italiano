# Data and codec licences

This file records the corpora and model weights the Caramel Italiano
**clean pilot** is allowed to use, as audited on 27 November 2026
(Europe/Zurich). The audit is research, not legal advice. No dataset was
downloaded, purchased, or signed up for when the audit was written.
Full notes, rejected/GREY corpora, and hour counts are in
`docs/data-audit.md`.

**Scope of the pilot.** Open single-speaker Italian (FLEURS, Multilingual
LibriSpeech, Common Voice) used to build synthetic two-track dialogues,
plus the ~3 hour DataHive Italian Conversations stereo sample. Kyutai
Mimi (8 codebooks) is the tokenizer. YODAS, KIParla, CLIPS, Nexdata,
ITALIC (voice-cloning caveat), VoxPopuli, and other GREY or out-of-scope
sets are **not** part of this pilot.

Audio and tokens must stay on the machine that processes them. Do not
commit them to this repository. Common Voice additionally forbids
re-hosting the corpus and speaker re-identification.

Attribution wording that is not copied from a source page is marked
**suggested** and should be reviewed before a public model or paper
release.

## Datasets

| Dataset | Source URL | Licence (as stated at the source) | Attribution text |
|---|---|---|---|
| DataHive Italian Conversations (sample) | [Hugging Face dataset card](https://huggingface.co/datasets/datahiveai/italian-conversations-sample) | CC BY 4.0. Card: “Free for research and commercial use, including model training, with attribution to DataHive.” Participants “explicitly consent to recording, sharing with customers, and AI model training.” Stereo: caller left, callee right, shared timeline. Sample only (~3 h 3 m 4 s, 6 conversations); larger collections are custom/paid and are out of scope. | The card requires attribution to DataHive but does not give a formal citation string. **Suggested:** “This work uses the Italian Conversations sample by DataHive, licensed under CC BY 4.0 (https://huggingface.co/datasets/datahiveai/italian-conversations-sample).” |
| Mozilla Common Voice, Scripted Speech 27.0, Italian | [MDC datasheet](https://mozilladatacollective.com/datasets/cmu5xagm300dyo1075qdhf20m); [Common Voice Legal Terms](https://commonvoice.mozilla.org/cv/terms) | Creative Commons Zero (CC0 / CC-0) public domain dedication, plus MDC conditions: do not determine the identity of speakers; “it is forbidden to re-host or re-share this dataset”; obtain the data via Mozilla Data Collective. Terms (effective 31 October 2025): Mozilla makes the datasets available “exclusively through MDC … unless otherwise specified” and asks that you “not post, distribute, or mirror any Common Voice dataset, in whole or in part, on other platforms or services.” | Attribution is not required under CC0. **Courtesy credit (suggested):** “This work uses Mozilla Common Voice Scripted Speech 27.0 (Italian), a CC0 dataset obtained through Mozilla Data Collective.” The no-rehost and no-speaker-identification conditions still apply. Access requires an MDC account; this project has not created one yet. |
| Multilingual LibriSpeech (MLS), Italian | [OpenSLR 94](https://www.openslr.org/94/); hours and provenance: [Pratap et al., 2020](https://arxiv.org/abs/2012.03411) | OpenSLR: “License: CC BY 4.0.” The paper states the set is derived from LibriVox / Project Gutenberg recordings “released in the public domain.” Train 247.38 h, dev 5.18 h, test 5.27 h (257.83 h total). Prefer MLS over overlapping LibriVox-derived releases (M-AILABS, CML-TTS). | OpenSLR does not publish a mandatory attribution sentence. **Suggested**, combining the licence and the standard academic citation: “This work uses the Italian portion of Multilingual LibriSpeech (MLS) (Pratap, V., et al., 2020, https://www.openslr.org/94/), licensed under CC BY 4.0. MLS is derived from public-domain LibriVox and Project Gutenberg material.” |
| FLEURS, Italian (`it_it`) — training / synthesis source and codec-test set | [google/fleurs](https://huggingface.co/datasets/google/fleurs); [Conneau et al., 2022](https://arxiv.org/abs/2205.12446) | Hugging Face dataset card YAML: `license: cc-by-4.0`. Paper: “approximately 12 hours of speech supervision per language”; no Italian-specific hour figure is stated. Single-speaker read FLoRes sentences. Option A plans on about 14 h. | The card does not publish a mandatory one-line attribution string. **Suggested:** “This work uses FLEURS (Conneau, A., et al., 2022, Few-shot Learning Evaluation of Universal Representations of Speech; https://huggingface.co/datasets/google/fleurs), licensed under CC BY 4.0. FLEURS is a product of Google.” |
| Emozionalmente | [OpenSLR 161](https://www.openslr.org/161/); [Zenodo 12616095](https://zenodo.org/records/12616095) | OpenSLR: “License: Creative Commons Attribution 4.0 International.” Zenodo `license: cc-by-4.0`. 6,902 acted clips, mean 3.81 s (about 7.3 h), mono 16 kHz. Option A downloads the whole zip. | **Suggested:** “This work uses Emozionalmente (Catania, F., Wilke, J. W., and Garzotto, F., 2025), CC BY 4.0, https://www.openslr.org/161/ and https://zenodo.org/records/12616095.” |
| FLEURS, English (`en_us`) — codec comparison only, not training data | Same as above | Same CC BY 4.0 | Same suggested attribution, noting the English test split was used only to compare Mimi reconstructions, not to train Caramel Italiano. |
| Multilingual LibriSpeech (MLS), Italian — lean remainder | Same as the MLS row above | Same CC BY 4.0. The mono store already holds about 100 h. The lean run keeps the rest of the deterministic archive order, about 158 h, and does not re-encode the first 100 h. | Same suggested MLS sentence, noting this slice is the portion after the first 100 h. |
| VoxPopuli Italian, unlabelled, 2015 only — lean run | Official tarballs `it_2015.tar` and `it_2015_2.tar` at `https://dl.fbaipublicfiles.com/voxpopuli/audios/`. Index: [facebookresearch/voxpopuli](https://github.com/facebookresearch/voxpopuli). | README: “VoxPopuli Data \| CC0” (see the [European Parliament legal notice](https://www.europarl.europa.eu/legal-notice/en/) for the raw recordings). Pre-trained models and the VoxPopuli code are CC BY-NC 4.0 and are **not** downloaded. The lean run keeps at most 1,000 h and checks that the tar members are 2015 events. | **Suggested:** “This work uses VoxPopuli Italian unlabelled parliamentary audio from 2015 (Wang et al., 2021; https://github.com/facebookresearch/voxpopuli), CC0. The recordings are European Parliament material; the source is acknowledged as the Parliament’s legal notice requires. VoxPopuli models and code (CC BY-NC 4.0) are not used.” |

## Codec weights (not a dataset)

| Work | Source URL | Licence (as stated) | Attribution text |
|---|---|---|---|
| Kyutai Mimi (tokenizer / codec), 8 codebooks in the caramel-1 recipe | [kyutai/mimi](https://huggingface.co/kyutai/mimi); companion code [kyutai-labs/moshi](https://github.com/kyutai-labs/moshi) | YAML `license: cc-by-4.0`. Card: “Developed by: Kyutai”; “License: CC-BY.” Weights are CC BY 4.0; Moshi Python code is separately MIT (not vendored here). | Official card citation (use this): Défossez, A., Mazaré, L., Orsini, M., Royer, A., Pérez, P., Jégou, H., Grave, E., and Zeghidour, N. (2024). *Moshi: a speech-text foundation model for real-time dialogue*. Kyutai. https://arxiv.org/abs/2410.00037. Prose notice used in NOTICE: “This project uses the Mimi neural audio codec developed by Kyutai, available at https://huggingface.co/kyutai/mimi, licensed under CC BY 4.0.” |

## Upstream model (recipe, not Italian training data)

| Work | Source URL | Licence | Attribution text |
|---|---|---|---|
| caramel / caramel-1 | [github.com/naklecha/caramel](https://github.com/naklecha/caramel), pinned at `84a9271d265841543287d4044d7ef80991a2b003`; weights at [huggingface.co/naklecha/caramel-1](https://huggingface.co/naklecha/caramel-1) | Apache License 2.0 | Credit the caramel project and Nishant Aklecha / naklecha. The warmstart run initialises from these public bf16 weights (fresh optimizer, no optimizer state in the file) and retrains on the Option A Italian tokens. See NOTICE for the unfilled upstream copyright-line caveat. |

## What is deliberately not listed as a pilot source

VoxPopuli models and code stay out (CC BY-NC 4.0). The lean data run, approved separately, may use the **2015 unlabelled Italian audio only** (CC0, at most 1,000 h, from `it_2015.tar` and `it_2015_2.tar`) plus the MLS Italian hours that are not already in the mono store. ITALIC (CC BY 4.0 with an explicit “do not clone their voices” download condition), YODAS and other YouTube-derived sets (legally GREY), and research-only or non-commercial dialogue corpora (KIParla, CLIPS, and similar) are documented in `docs/data-audit.md` and stay out of scope.
