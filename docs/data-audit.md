_Snapshot of the project data audit, copied into this draft on 27 November 2026 (Europe/Zurich). This is research, not legal advice._

# Caramel Italiano — Italian speech data legal/access audit

**Project:** Caramel Italiano (retraining [caramel-1](https://github.com/naklecha/caramel) on Italian)  
**Audit date:** 27 November 2026 (Europe/Zurich)  
**Scope:** web research only. No datasets were downloaded, no accounts were created, and no money was spent.  
**This is not legal advice.** Licence quotes are taken from public pages as they stood on the audit date. Counsel should review any training plan, especially YouTube-sourced data, GDPR/consent for conversational audio, and paid vendor contracts.

## Method and verdicts

For every corpus the table records only what the **source** states. If a figure or licence was not found at the source, it is written **UNKNOWN** — nothing is invented. Hours from secondary surveys or third-party blogs are used only when flagged as such, and they do not enter the OK totals.

**Verdicts**

| Verdict | Meaning used here |
|---|---|
| **OK** | The public licence text clearly allows commercial use and model training (e.g. CC0, CC BY, a BSD-style grant of commercial use, or an explicit “commercial use including model training” statement), and access is open or openly downloadable. Residual conditions (attribution, no speaker re-identification, EP source acknowledgement) are noted but do not by themselves make the set GREY. |
| **GREY** | Usable only with material caveats: NC/ND/ShareAlike, research-only, institutional agreement, GDPR/audio-download ban, YouTube Terms vs CC compilation, GPL copyleft uncertainty, or a paid product whose contract was not readable without signup/purchase. |
| **NO** | No Italian speech, no audio (transcripts only), or the resource does not exist as described. |

**Overlap rule.** Several “datasets” are views of the same audio (LibriVox → MLS / M-AILABS / CML-TTS; European Parliament → VoxPopuli / Europarl-ST / MOSEL labels; YouTube CC → YODAS / YouTube-Commons / MOSEL v2). Unique-hour totals in the summary **de-duplicate** these families. Collection/relabel sets (MOSEL, Granary) are catalogued but **not** added as extra audio.

**Original caramel-1 recipe (approximate / unverified, as given in the brief):** ~5,700 h of two-person conversations with each speaker on their own track, plus ~5,800 h of one-person speech.

**Group A** is conversational / dialogue speech (two or more speakers), ideally separate channels or reliably splittable. **Group B** is single-speaker / monologic / read speech. Parliamentary floor audio is placed in B, with a cross-check note under A, because it is not two-person conversation with per-speaker tracks.

---

## Group A — conversational / dialogue

### Compact table

| Dataset | URL | Licence (as stated) | Commercial / training | Italian hours (as stated) | Separate tracks | Access | Verdict |
|---|---|---|---|---|---|---|---|
| DataHive Italian Conversations (sample) | [HF card](https://huggingface.co/datasets/datahiveai/italian-conversations-sample) | CC BY 4.0; “Free for research and commercial use, including model training, with attribution to DataHive.” | Yes, with attribution | **3 h 3 m 4 s** (6 conversations) | **Yes** — stereo, caller left / callee right, shared timeline | Open download (HF) | **OK** (sample only; larger collections are custom/paid) |
| YODAS / YODAS2 (IT shards) | [espnet/yodas](https://huggingface.co/datasets/espnet/yodas), [yodas2](https://huggingface.co/datasets/espnet/yodas2) | Dataset card YAML: `license: cc-by-3.0`. Authors: “We made sure that our dataset only consisted of videos with CC licenses during our downloading.” | Dataset licence is CC BY 3.0, but underlying YouTube rights are unsettled | **it000 1,454.98 h** (manual captions) + **it100 4,989.62 h** + **it101 4,242.87 h** (automatic) = **10,687.47 h** | Not as released (typically mixed YouTube audio). Some videos may be stereo; not documented as per-speaker tracks. Timestamps on utterances; diarization would be required. | Open download / streaming on Hugging Face | **GREY** — YouTube ToS vs CC compilation; mixed speech types, not a clean dialogue set; captions “manual” means user-uploaded, not necessarily human |
| YouTube-Commons (IT) | [PleIAs/YouTube-Commons](https://huggingface.co/datasets/PleIAs/YouTube-Commons) | “All the transcripts are part of a video shared under a CC-By license.” Maintainers “recommend that this set be preferably used for open research.” | Unclear for audio (this release is transcripts) | Italian-original **video count** is discussed in a [HF thread](https://huggingface.co/datasets/PleIAs/YouTube-Commons/discussions/14); **audio hours of Italian speech are not stated on the dataset card** | No audio in this release | Open download of transcripts; audio would have to be re-fetched from YouTube | **NO** as an audio corpus (transcripts only). **GREY** if one were to rebuild audio from YouTube (same YouTube-CC issues as YODAS) |
| MOSEL (IT, collection) | [FBK-MT/mosel](https://huggingface.co/datasets/FBK-MT/mosel), [FBK page](https://mt.fbk.eu/mosel/) | “CC-BY-4.0” / HF YAML `license: cc-by-4.0` for the collection (pseudo-labels). Audio remains under each **source** licence (VoxPopuli, YouTube-Commons, etc.) | Collection licence is CC BY 4.0; audio follows sources | Paper/card table: **labeled 3,756 h**, **unlabeled 21,933 h**, **total 25,689 h**. HF repo transcripts cover **16,713 h** of that subset | Not a new recording set; source-dependent | Open download of labels; audio from sources | **GREY** as a unique conversational source (mostly VoxPopuli + YouTube-Commons labels; do not add hours) |
| KIParla | [kiparla.it](https://kiparla.it/en/), [search/terms](https://kiparla.it/en/search/), [GitHub collection](https://github.com/KIParla/KIParla-collection) | Consultation “exclusively for scientific research purposes and without commercial gain.” Downloaded NoSketch Engine data “shareable under the Creative Commons BY-NC-SA 4.0 license.” Audio: GDPR / consent-form restrictions; “It is not possible to download the corpus audio files and/or record them using personal means.” | No (research / NC; no audio download) | Homepage: “**More than 350 hours** of transcribed recordings… totaling 994 speakers.” (Do not treat older module totals as the current whole.) | No separate channels stated. Time-aligned ELAN/Jefferson transcripts with speaker codes — **splittable in principle from timestamps**, if audio were obtainable | Transcripts open (GitHub/CLARIN); audio **request / institutional registration**, listen-in-browser only | **GREY** — NC, research-only, audio not downloadable under stated terms |
| CLIPS (Corpora e Lessici dell’Italiano Parlato e Scritto) | [clips.unina.it](http://www.clips.unina.it/), [structure](http://www.clips.unina.it/corpus-structure), [ELRA-U-S 0014](http://universal.elra.info/product_info.php?cPath=37_46&products_id=1624), [Leipzig description](https://home.uni-leipzig.de/burr/CorpusLing/Portal/Korpus/Korpusbeschreibung_CLIPS.htm) | Leipzig (quoting the project): corpus “fully downloadable… **free for research purposes**.” Italian: “l’accesso e il download completo del materiale **per finalità scientifiche** sono gratuiti.” No commercial-use licence found on pages retrieved. Official portal timed out during this audit (HTTP 504). | Not stated as allowed | “**about 100 hours**” / “**100 hours** of recorded speech” (project / ELRA descriptions), including radiotelevision, **240 dialogues** (map task + spot-the-difference), read speech, telephone (≈300 speakers vs a simulated hotel desk), and anechoic read speech | **UNKNOWN**. Documentation describes dialogue and telephone sub-corpora but does **not** state separate per-speaker channels. Some material is orthographically transcribed / phonetically labelled (timestamps possible for a subset) | Free **registration** then download (as described historically). Portal availability was not confirmed live. | **GREY** — research-only wording; commercial use unstated; portal/licence text not fully recovered live |
| API / AP (Archivio del Parlato Italiano, map task) | Historical: CIRASS / Università di Napoli Federico II. [LINGUIST List 15.3192](https://linguistlist.org/issues/15/3192/); review: [Romano, RdL](https://linguistica.sns.it/RdL/19.2/Romano.pdf) | **UNKNOWN** at a current official source. Originally distributed free on DVD. No current licence statement found. | UNKNOWN | Review literature: **18 adult map-task dialogues, “more than 4 hours”**, plus child and read material. Treat as secondary; no live official hour page found | Map-task corpora are often dual-mic; **not confirmed** for API at a current official source | Current download endpoint **UNKNOWN** (historical DVD / possibly defunct links) | **GREY** — access and licence not established today |
| LABLITA reference corpus of spoken Italian | [corpus.lablita.eu](http://corpus.lablita.eu/?locale=en); [Cresti et al., “The LABLITA Speech Resources”](https://www.societadilinguisticaitaliana.net/wp-content/uploads/2022/11/005_CrestiEtAlii_Atti_LIV_Congresso_SLI.pdf) | Paper: corpus is “**open and continuously updated**” and “delivered to the users online and for downloading” via the ORFEO platform (metadata, transcripts, alignments). **No SPDX / Creative Commons / commercial-use clause was found.** Includes the Italian section of C-ORAL-ROM, which is also sold by ELRA (see next row) — rights stack is unclear | UNKNOWN (open for research use as described; commercial training not stated) | **Total audio hours UNKNOWN.** Paper: “approx. **700,000 token**”; **422** samples; includes C-ORAL-ROM Italian (“approx. **300,000 words**”) plus earlier Crusca / GRIT / Stammerjohann material. “**40 hours** of original recordings” refers to the **Stammerjohann 1965** source, not the whole corpus. Face-to-face interaction is “more then 2/3” of the collection; types include monologue, two-speaker dialogue, and multi-dialogue; also telephone and broadcasting | **UNKNOWN** as separate tracks. Transcripts aligned to audio by terminated sequences, with speaker participation documented — **splittable in principle from alignments** | Online search + download via ORFEO (as described). Exact click-through terms **UNKNOWN** | **GREY** — no commercial licence found; overlap with ELRA-sold C-ORAL-ROM |
| C-ORAL-ROM (Italian component) | [ELRA-S0172](https://catalogue.elra.info/en-us/repository/browse/ELRA-S0172/); [Cresti & Moneglia 2004/2005](https://benjamins.com/catalog/scl.15) | ELRA paid end-user licences (academic and commercial price tiers are listed in catalogue descriptions in secondary reports). The ELRA catalogue page itself **timed out** in this audit, so **exact current licence wording and prices were not captured from the live page**. A subset also appears inside LABLITA/ORFEO (above) | Commercial use is typically available from ELRA **for a fee**, but the contract was not reviewed | Italian word count “approx. **300,000 words**” (LABLITA paper). Full four-language package duration is cited in catalogue descriptions as **123:27:35** total — **Italian-only hours UNKNOWN** (not stated separately at sources successfully retrieved) | **UNKNOWN** as separate channels; spontaneous dialogue/monologue with transcripts | **Paid** via ELRA (and/or overlapping ORFEO access for a subset) | **GREY** — paid agreement; Italian hours not stated separately; rights vs LABLITA release unclear |
| DIA — Dialogic ItAlian | [unibz record](https://bia.unibz.it/esploro/outputs/dataset/DIA-Dialogic-ItAlian-corpus/991006608798401241); [The Language Archive](https://archive.mpi.nl/tla/islandora/object/tla%3A1839_277c2e49_96ed_49fc_964e_7699e78e83cd) | **UNKNOWN** (no licence text retrieved from the catalogue pages in this audit). TLA listing indicates **restricted** access in secondary descriptions | UNKNOWN | “Approximately **10 hours**” of recorded dialogues (survey / paper descriptions). Confirm on the record before use | **UNKNOWN**; spontaneous two-party dialogue, likely mixed or single-mic unless the record says otherwise | **Request** (TLA / contact Alessandro Vietti, as listed on TLA) | **GREY** — licence and channel format not established; research-style restricted access |
| ASR-ItaCSC (MagicHub / Magic Data) | [magichub.com](https://magichub.com/datasets/italian-conversational-speech-corpus/) | Page displays the **CC BY-NC-ND 4.0** badge. A full prose licence paragraph was not captured in the fetch; treat the badge as the stated licence and re-read the page before use | No (NC); ND also restricts derivatives | “**10.43 hours** of transcribed Italian conversational speech… **28 conversations between three pairs of speakers**” | **No** — “16 kHz, 16 bits, **mono**” | Open download (vendor “open-source” page), subject to the NC-ND licence | **GREY** — CC BY-NC-ND 4.0; mono |
| VoxPopuli, Italian (parliamentary) — checked as an A candidate | [facebookresearch/voxpopuli](https://github.com/facebookresearch/voxpopuli) | See Group B for the full licence quote (data **CC0**, plus [EP legal notice](https://www.europarl.europa.eu/legal-notice/en/)) | Data: yes (CC0 / EP notice allows commercial reuse with source acknowledgement). Released **models/code** are CC BY-NC 4.0 — do not reuse those commercially | See Group B: **91 h** transcribed; **4.6K / 21.9K h** unlabelled (v1/v2) | Not two-person conversation tracks. Floor speeches are typically one speaker at a time. The **speech-to-speech interpretation** tables (e.g. It→* 961 h total across targets; *→It 1.3K h) are source+interpreter, not dialogue | Open (GitHub / Fair scripts) | **Not a Group A match.** Legally closer to **OK** as monologic/parliamentary audio — counted in Group B, not in A totals |
| Europarl-ST (IT as source) | [mllp.upv.es/europarl-st](https://mllp.upv.es/europarl-st/), [README](https://www.mllp.upv.es/europarl-st/README.md) | “The work carried out for constructing the Europarl-ST corpus is released under a **Creative Commons Attribution-NonCommercial 4.0 International license (CC BY-NC 4.0)**.” “All rights of the data belong to the European Union and respective copyright holders.” README also reproduces the EP notice (commercial dissemination of EU items with acknowledgement) — the **dataset work** is nonetheless CC BY-NC 4.0 | Dataset packaging: **no** (CC BY-NC). Underlying EP recordings may have a different reuse path (see EP notice / VoxPopuli CC0) | Train IT→en **37 h**, IT→other directions **19–21 h** each (same source speeches, not additive). Train-noisy IT→* **15 h** each. Dev/test “all between 3 and 6 hours.” These overlap EP audio used in VoxPopuli | No (segmented floor speeches) | Open download | **GREY** — NC on the corpus work; overlap with VoxPopuli |
| Nexdata Italian telephone conversations | [nexdata.ai …/1232](https://www.nexdata.ai/datasets/speechrecog/1232) | No full contract on the public page. The page markets the set for “conversational AI… and multilingual speech model training” and states GDPR/CCPA/PIPL compliance. Licence name/text **UNKNOWN** without purchase | Claimed suitable for model training; **contract not reviewed** | “**499 hours**… **676** native speakers” | **No** — “8 kHz 8 bit, a-law/u-law pcm, **mono channel**.” Annotations include “timestamp, speaker ID” — **splittable from timestamps**, not separate tracks | **Paid** | **GREY** — paid, contract unread; mono (not Fisher-style dual-channel) |
| Defined.ai Italian spontaneous dialogue / podcast catalogues | e.g. [italian-spontaneous-dialogue](https://www.defined.ai/datasets/italian-spontaneous-dialogue), domain pages (telco, banking, insurance), [lifestyle podcast](https://www.defined.ai/datasets/italian-lifestyle-podcast) | **UNKNOWN** — vendor pages did not return licence text in this audit (JS-rendered / not retrieved) | UNKNOWN (commercial vendor; terms not reviewed) | Official aggregate hours **UNKNOWN** at source in this audit. A [2025 survey CSV](https://github.com/marco-giordano/spoken-italian-datasets) (secondary) lists 1,797 h spontaneous dialogue, 60 h lifestyle podcast, 2,531 h scripted monologue — **do not treat as verified** | UNKNOWN at source (telephone/dialogue products are often dual-channel, but this was **not** confirmed here) | **Paid** | **GREY** — paid; hours and licence not verified at the vendor page |
| FutureBee AI Italian conversation / call-centre sets | Catalogue under [futurebeeai.com](https://www.futurebeeai.com/) (general conversation and domain call-centre pages) | **UNKNOWN** (commercial listings; terms not reviewed) | UNKNOWN | Official hours **UNKNOWN** here. Survey CSV (secondary) lists e.g. 50 h general conversation and 30 h per call-centre domain — **unverified** | UNKNOWN | **Paid** | **GREY** — paid; not verified at source |
| PortMedia French and Italian (ELRA-S0371) | [ELRA-S0371](https://catalogue.elra.info/en-us/repository/browse/ELRA-S0371/) | ELRA paid academic/commercial licences (standard ELRA terms; live page not fully captured) | Typically yes under a **paid commercial ELRA licence**; contract not reviewed | “**604**” Italian transcribed dialogues (catalogue / survey). **Hours UNKNOWN** at sources retrieved | UNKNOWN (wizard-of-Oz / spoken dialogue systems often close-talk or stereo — not confirmed) | **Paid** (ELRA) | **GREY** |
| Fisher-style LDC Italian telephone | — | — | — | — | — | — | **NO** — no Italian Fisher (or LDC Fisher-like) corpus was found. Closest analogues are paid telephone sets above, and CLIPS telephone (research-only) |
| Common Voice Spontaneous Speech 5.0, Italian | [SS v5 announcement](https://community.mozilladatacollective.com/commonvoice-27/) | Common Voice datasets are CC0 (see Group B), **if** an Italian SS set exists | Would be yes, if present | SS 5.0 covers **80 languages**, 89,754 clips, **302 h** transcribed/validated **across all languages**. A language-level Italian SS 5.0 datasheet was **not found**. Pages were found for English, French, Catalan, **Ligurian**, etc. Italian in the 21-language SS ASR shared task list was **not** found | Spontaneous answers to prompts are typically **single-speaker**, not two-person dialogue | Open via Mozilla Data Collective, if released | **UNKNOWN / not confirmed** for Italian. Even if present, it would be closer to Group B (prompted monologue) than to two-person conversation |
| Emilia / Emilia-YODAS | [amphion/Emilia-Dataset](https://huggingface.co/datasets/amphion/Emilia-Dataset) | Original Emilia: CC BY-NC 4.0; Emilia-YODAS: CC BY 4.0 (as stated on the project card / discussions) | N/A for Italian | **No Italian.** Languages covered: English, Chinese, German, French, Japanese, Korean | — | Open (with licence split) | **NO** — no IT split |

### Notes on Group A quality vs the caramel-1 recipe

The recipe wants **two-person conversation with each speaker on its own track** (telephone stereo or separate mics). Among open, commercially usable sets, only the **DataHive 3-hour stereo sample** clearly meets that technical bar. Academic Italian dialogue corpora (KIParla, CLIPS, LABLITA, C-ORAL-ROM, API, DIA, ASR-ItaCSC) are the right *genre* but are research-only, NC, paid-via-ELRA, or currently unobtainable. YODAS is large but is not a speaker-separated dialogue corpus. The Nexdata 499 h telephone set is **mono**.

---

## Group B — single-speaker / monologic / read speech

### Compact table

| Dataset | URL | Licence (as stated) | Commercial / training | Italian hours (as stated) | Separate tracks | Access | Verdict |
|---|---|---|---|---|---|---|---|
| Common Voice Scripted Speech 27.0 — Italian | [MDC datasheet](https://mozilladatacollective.com/datasets/cmu5xagm300dyo1075qdhf20m); [Legal Terms](https://commonvoice.mozilla.org/cv/terms) | “This dataset is released under the **Creative Commons Zero (CC-0) licence**.” Terms: “We make Mozilla’s Common Voice dataset available under the Creative Commons CC0 public domain dedication.” Datasheet: “By downloading this data you agree to not determine the identity of speakers… It is forbidden to re-host or re-share this dataset.” Access is via Mozilla Data Collective. | Yes (CC0). Do not identify speakers; do not re-host the corpus; obtain it via MDC | “**430.5 hours** of recorded speech (**363.9 hours validated**) from **7350** speakers… 284667 clips… cv-corpus-27.0-2026-09-11.” Release date on the card: 17 Sep 2026 | Single-speaker read clips (not conversation) | Open via MDC (account on that platform; this audit did not sign up) | **OK** |
| Multilingual LibriSpeech (MLS), Italian | [OpenSLR 94](https://www.openslr.org/94/); hours: [Pratap et al.](https://arxiv.org/abs/2012.03411) | OpenSLR: “**License: CC BY 4.0**.” Paper: derived from LibriVox / Gutenberg, “released in the public domain, which allows us to release MLS freely to everyone.” | Yes, with attribution (CC BY 4.0) | Train **247.38 h**, dev **5.18 h**, test **5.27 h** (**257.83 h** total). 22+43 train speakers (M/F) | Single-speaker audiobook read speech (one reader per clip; some books have mixed readers upstream) | Open download (hosted on Facebook public S3, not on OpenSLR itself) | **OK** |
| LibriVox Italian (upstream of MLS / M-AILABS / CML-TTS) | [librivox.org](https://librivox.org/); counts in [MLS paper Table 1](https://arxiv.org/html/2012.03411v2) | LibriVox recordings of public-domain texts are released as **public domain** (project policy). Always check the page for each book. | Yes, for public-domain items | MLS Table 1: **279.43 h**, 61 books, 28 speakers in the LibriVox Italian pool **at the time of MLS collection**. Current catalogue size **UNKNOWN** (not counted here) | Single-speaker (per book/section) | Open | **OK** (public-domain items), but **overlaps** MLS / M-AILABS / CML-TTS. Prefer MLS as the cleaned, cited training release |
| M-AILABS Speech Dataset, it_IT | [i-celeste-aurora/m-ailabs-dataset](https://github.com/i-celeste-aurora/m-ailabs-dataset) | Custom BSD-style grant: “Redistribution and use in any form, **including any commercial use**, with or without modification are permitted provided that…” copyright notice and no-endorsement conditions are kept. Underlying LibriVox/Gutenberg: public domain (Ukrainian excepted — not relevant to IT) | Yes, subject to the notice conditions | “**Italian / Italy / it_IT / 127h 40m**” | Single-speaker read clips (wav, mono, 16 kHz) | Open (links in that README) | **OK**, but **heavy overlap** with LibriVox / MLS — do not add to unique totals |
| CML-TTS, Italian | [OpenSLR 146](https://www.openslr.org/146/); [Oliveira et al.](https://arxiv.org/html/2306.10097) | OpenSLR: “**License: CC-BY 4.0 license**.” Paper: “The dataset is publicly available under the CC-BY 4.0 license.” “**CML-TTS is based on Multilingual LibriSpeech (MLS)** and adapted for training TTS models.” | Yes, with attribution | Train **73.78 h**, test **1.47 h**, dev **0.40 h** (**75.65 h**) | Single-speaker (TTS-oriented segmentation of MLS/LibriVox) | Open download | **OK**, but **derived from MLS** — do not add unique hours |
| VoxPopuli, Italian | [GitHub README](https://github.com/facebookresearch/voxpopuli); [HF card](https://huggingface.co/datasets/facebook/voxpopuli) | README: “VoxPopuli Data \| **CC0** (see also European Parliament’s [legal notice](https://www.europarl.europa.eu/legal-notice/en/) for the raw data).” HF: “The dataset is distributet under CC0 license, see also European Parliament's legal notice for the raw data.” **Pre-trained models and code: CC BY-NC 4.0.** EP notice: reuse of EU-owned multimedia “is authorised, for personal use or for further **non-commercial or commercial dissemination**, provided that the entire item is reproduced and the source is acknowledged.” | **Data: yes.** Do not commercially reuse the released VP **checkpoints/code**. Acknowledge EP source as the notice requires | Unlabelled **4.6K h (v1) / 21.9K h (v2)** — versions, **not** additive; use **v2 = 21.9K h**. Transcribed **91 h**, 306 speakers, 757K tokens. Interpretation: Italian as source **961 h** total across 14 targets; Italian as target **1.3K h** (these are parallel S2S hours, not extra unique Italian floor audio) | Typically one official speaker per floor take. Not stereo conversation | Open | **OK** (data). Domain: formal parliamentary speech, not casual monologue |
| FLEURS, Italian (`it_it`) | [google/fleurs](https://huggingface.co/datasets/google/fleurs); [Conneau et al.](https://arxiv.org/abs/2205.12446) | HF YAML: `license: cc-by-4.0` | Yes, with attribution | Paper: “**approximately 12 hours of speech supervision per language**” (102 languages). **No Italian-specific hour figure** is given. HF `it_it` clip counts: train 3,030, validation 391, test 865 | Single-speaker read (n-way parallel FLoRes sentences) | Open (HF / TFDS) | **OK**. Hours for IT specifically: **UNKNOWN** beyond the “≈12 h per language” statement |
| ITALIC | [Zenodo 8040649](https://zenodo.org/records/8040649); [paper](https://arxiv.org/html/2306.08502) | Zenodo: “The ITALIC dataset is released under the **Creative Commons Attribution 4.0 International License**.” “you agree not to attempt to determine the identity of speakers… or to clone their voices.” | Training is allowed under CC BY 4.0; the Zenodo note asks users **not to clone speakers’ voices** — relevant if caramel-style voice cloning is in scope | Paper: “**15.46 hours** of recording… 16,521 audio recordings by 70 distinct volunteers.” Massive split: 10.80 / 1.90 / 2.76 h | Single-speaker short prompts (assistant-style utterances), 1.14–38.34 s | Open (Zenodo / GitHub) | **OK** for ASR-style training; **GREY for voice cloning** if the “do not clone” request is treated as binding (it is a stated condition of download, not a CC clause) |
| Emozionalmente | [OpenSLR 161](https://www.openslr.org/161/) | “**License: Creative Commons Attribution 4.0 International**” | Yes, with attribution | “**6,902** labeled samples… 431 amateur actors… average **3.81 seconds** (SD = 0.99).” **Total hours are not stated** on the OpenSLR page (this audit does not compute a total) | Single-speaker acted emotional read speech, mono 16 kHz | Open download | **OK**. Total hours: **UNKNOWN** (clip count + mean duration only) |
| VoxForge Italian | [MDC datasheet](https://mozilladatacollective.com/datasets/cmp2ghayo00lzmp07eh8fssxm); [voxforge.org](https://www.voxforge.org/home/downloads/speech/italian-speech-files) | MDC: “**GNU General Public License v3.0 or later (GPL-3.0)**.” Each session includes `GPL_license.txt` | GPL permits use, but **copyleft** may be argued to reach some distributed derivatives. Whether a trained speech-to-speech **model** is a GPL derivative is legally unsettled. MDC “Forbidden Usage: N/A” does not resolve that | “**20 hours** (10643 utterances)… at least 207 speakers.” Release date on the card: 12 May 2026 | Single-speaker read prompts | Open (VoxForge historically; also packaged on MDC) | **GREY** — GPL-3.0-or-later copyleft risk for a shipped model |
| Multilingual TEDx (mTEDx), Italian | [OpenSLR 100](https://www.openslr.org/100/) | “**License: Creative Commons Attribution-NonCommercial-NoDerivs 4.0 (CC BY-NC-ND 4.0)**” | **No** (NC and ND) | Official OpenSLR page states file size (`mtedx_it.tgz` 19G) but **does not state Italian hours**. Paper (Salesky et al., Interspeech 2021) is the usual source for hours — treat any figure not copied from that paper’s table as **UNKNOWN here** (the paper PDF was not fully tabulated in this pass). Do not use unofficial 107 h figures without citing the paper table | Single-speaker talks (audience / residual overlap possible) | Open download | **GREY** — CC BY-NC-ND 4.0 |
| MOSEL IT labels | See Group A | CC-BY-4.0 on labels; audio from sources | Depends on source audio | Labeled 3,756 / unlabeled 21,933 / total 25,689 h (includes VoxPopuli IT and other surveyed open sources; v2 adds YouTube-Commons transcripts) | Source-dependent | Open labels | **Not extra audio.** Useful as transcripts for VoxPopuli / YouTube-CC. Verdict follows the **audio** source |
| YODAS IT | See Group A | CC BY 3.0 on the compilation; YouTube CC videos | GREY | 10,687.47 h | Not per-speaker tracks | Open | **GREY** (same YouTube issue). Mixed monologue/dialogue/other. Can feed a diarized B or weak-A pipeline only with legal caution |
| Emilia | See Group A | — | — | No Italian | — | — | **NO** |

**Other Group B items checked, not used in OK totals**

| Dataset | Finding | Verdict |
|---|---|---|
| DEMoS (emotional) | [Zenodo 2544829](https://zenodo.org/record/2544829). Survey: research EULA, academic use. Hours not restated here without the record’s licence text. | **GREY** (research EULA, as described) |
| EMOVO | 588 acted utterances; originally distributed on request ([LREC 2014 paper](https://www.lrec-conf.org/proceedings/lrec2014/pdf/591_Paper.pdf)). **Dataset licence UNKNOWN** (mirrors exist; do not assume the paper’s CC BY-NC applies to the audio). About “one hour” is a paper-style description in some write-ups — confirm in the paper before quoting. | **GREY** |
| IBNC (Italian Broadcast News, ELRA-S0093) | ~15 h transcribed broadcast news; ELRA paid. | **GREY** (paid) |
| Italian Speecon, SpeechDat-Car, Appen Italian Speech Corpus, and other ELRA read/telephone sets | Paid ELRA resources (hours vary; e.g. survey lists 44 h Appen Italian Speech Corpus, Speecon 600 speakers). Licences are paid academic/commercial ELRA contracts, not reviewed. | **GREY** (paid) |
| CoVoST 2 / CVSS (IT pairs) | Built on Common Voice; licence follows CV (CC0) plus any translation-layer terms. Audio **overlaps Common Voice**. | Do not add unique hours |
| MuST-C | English TED audio with translations including Italian **text**. Survey notes distribution “temporarily suspended” pending TED policy. Not Italian speech. | **NO** as Italian audio |

---

## Licence quotes (fuller text)

### Common Voice (CC0 + MDC conditions)

From the [MDC Italian 27.0 datasheet](https://mozilladatacollective.com/datasets/cmu5xagm300dyo1075qdhf20m):

> This dataset is released under the Creative Commons Zero (CC-0) licence. By downloading this data you agree to not determine the identity of speakers in the dataset.

From [Common Voice Legal Terms](https://commonvoice.mozilla.org/cv/terms) (effective 31 October 2025):

> We make Mozilla’s Common Voice dataset available under the Creative Commons CC0 public domain dedication.  
> … Mozilla makes the Common Voice datasets available to the general public exclusively through MDC under a Creative Common Zero (CC0) license, unless otherwise specified. … we ask that you not post, distribute, or mirror any Common Voice dataset, in whole or in part, on other platforms or services, and that access by others take place only through MDC.

### VoxPopuli data (CC0) and EP notice

From the [project README](https://github.com/facebookresearch/voxpopuli):

> VoxPopuli Data | [CC0](https://creativecommons.org/share-your-work/public-domain/cc0/) (see also European Parliament's [legal notice](https://www.europarl.europa.eu/legal-notice/en/) for the raw data)  
> Pre-trained Models | CC BY-NC 4.0  
> Code | CC BY-NC 4.0

From the [EP legal notice](https://www.europarl.europa.eu/legal-notice/en/):

> As a general rule, the reuse (reproduction or use) of textual data and multimedia items which are the property of the European Union … is authorised, for personal use or for further non-commercial or commercial dissemination, provided that the entire item is reproduced and the source is acknowledged.

### MLS (CC BY 4.0)

[OpenSLR 94](https://www.openslr.org/94/): “License: CC BY 4.0.”

### M-AILABS (commercial-use BSD-style)

From the [dataset README](https://raw.githubusercontent.com/i-celeste-aurora/m-ailabs-dataset/master/README.md):

> Redistribution and use in any form, including any commercial use, with or without modification are permitted provided that the following conditions are met:  
> - Redistributions of source data must retain the above copyright notice, this list of conditions and the following disclaimer.  
> - Neither the name of the copyright holder nor the names of its contributors may be used to endorse or promote products derived from this downloaded data, source-code or binary-code without specific prior written permission.

### KIParla (research / CC BY-NC-SA; no audio download)

From [kiparla.it/en/search](https://kiparla.it/en/search/):

> the consultation is carried out exclusively for scientific research purposes and without commercial gain;  
> … The data from the KIParla corpus downloaded via the NoSketch Engine platform are shareable under the Creative Commons BY-NC-SA 4.0 license.  
> ATTENTION! Researchers wishing to access the audio files for research purposes must contact the resource managers via their institutional email addresses. It is not possible to download the corpus audio files and/or record them using personal means, due to reasons related to the release form used during data collection and the European regulation on personal data protection (GDPR).

### DataHive sample (CC BY 4.0, training allowed)

From the [HF dataset card](https://huggingface.co/datasets/datahiveai/italian-conversations-sample):

> License: CC BY 4.0  
> Released under CC BY 4.0. Free for research and commercial use, including model training, with attribution to DataHive.  
> Channels: Caller left, callee right, on a shared timeline  
> Participants explicitly consent to recording, sharing with customers, and AI model training.

### Europarl-ST (CC BY-NC 4.0 on the corpus work)

From [README.md](https://www.mllp.upv.es/europarl-st/README.md):

> The work carried out for constructing the Europarl-ST corpus is released under a Creative Commons Attribution-NonCommercial 4.0 International license (CC BY-NC 4.0)  
> All rights of the data belong to the European Union and respective copyright holders.

### YODAS (CC BY 3.0 compilation of YouTube CC videos)

HF dataset cards for [yodas](https://huggingface.co/datasets/espnet/yodas/raw/main/README.md) and [yodas2](https://huggingface.co/datasets/espnet/yodas2/raw/main/README.md):

> license: cc-by-3.0  
> We made sure that our dataset only consisted of videos with CC licenses during our downloading.

### mTEDx (CC BY-NC-ND 4.0)

[OpenSLR 100](https://www.openslr.org/100/): “License: Creative Commons Attribution-NonCommercial-NoDerivs 4.0 (CC BY-NC-ND 4.0).”

### ITALIC (CC BY 4.0 + no voice-cloning request)

[Zenodo](https://zenodo.org/records/8040649):

> The ITALIC dataset is released under the Creative Commons Attribution 4.0 International License.  
> By downloading and accessing the dataset, you agree not to attempt to determine the identity of speakers in the ITALIC dataset or to clone their voices.

---

## Overlaps (do not double-count)

| Family | Members | Unique-hour treatment |
|---|---|---|
| LibriVox public-domain audiobooks | Raw LibriVox (~279.43 h at MLS collection), **MLS 257.83 h**, M-AILABS 127 h 40 m, CML-TTS 75.65 h | Count **MLS 257.83 h** once in OK unique totals |
| European Parliament 2008–2020 multimedia | **VoxPopuli** unlabelled 21.9K h (v2) + transcribed 91 h, Europarl-ST (NC packaging), MOSEL labels, interpretation subsets | Count **VoxPopuli 21.9K + 91 h** once. Do not add Europarl-ST or MOSEL |
| YouTube Creative Commons | YODAS 10,687.47 h, YouTube-Commons transcripts, MOSEL v2 YTC labels, Granary views | **Not OK unique.** If ever used, count YODAS once, GREY |
| Common Voice | Scripted 27.0 IT, CoVoST/CVSS derivatives | Count **363.9 validated h** (or 430.5 recorded, if unvalidated clips are in scope) once |
| C-ORAL-ROM Italian | ELRA-S0172 paid package; also included in LABLITA/ORFEO | Hours UNKNOWN; do not add twice |

NVIDIA **Granary / Granary-v2** is a further relabelling/packaging of these families (YODAS, VoxPopuli, YouTube-Commons, etc.). It is not additional unique Italian audio.

---

## Summary totals

### Unique **OK** hours

**Group A (conversational, commercially usable, open):** **3 hours 3 minutes 4 seconds**  
(DataHive Italian Conversations sample only.)

No other open corpus in this audit both (i) is clearly licensed for commercial model training and (ii) is documented as two-or-more-speaker conversation with separate channels or a reliable official split. Academic dialogue corpora are larger but **GREY**. YODAS is large but **GREY** and not speaker-separated dialogue.

**Group B (single-speaker / monologic / read, unique OK):**

| Source | Hours used in the unique total | Note |
|---|---|---|
| VoxPopuli IT unlabelled v2 | 21,900 | Formal parliamentary; v1 4.6K is superseded, not added |
| VoxPopuli IT transcribed | 91 | Official transcripts |
| Common Voice Scripted 27.0 IT | 363.9 validated (430.5 recorded) | Validated figure used in the headline total |
| MLS IT train+dev+test | 257.83 | Canonical LibriVox-derived set |
| FLEURS IT | ≈12 | “Per language” figure; IT-specific total not stated |
| ITALIC | 15.46 | Short assistant-style prompts; voice-cloning caveat |
| Emozionalmente | ≈7.3 | Option A downloads all of OpenSLR 161. 6,902 × 3.81 s from the OpenSLR page; not a published total |
| **Headline unique OK Group B** | **≈ 22,640 h** | Using 363.9 + 21,900 + 91 + 257.83 + 12 + 15.46 |

If unvalidated Common Voice clips are included, add ~66.6 h (430.5 − 363.9). If VoxPopuli unlabelled is rejected on domain grounds (parliament ≠ “one-person speech” in the caramel sense), unique OK Group B falls to **≈ 649 h** (CV + MLS + FLEURS + ITALIC) — far closer to, or below, the ~5,800 h recipe.

### Gap vs the original recipe

| Recipe bucket | Target (approx., unverified) | Unique OK found | Gap |
|---|---|---|---|
| A. Two-person, **separate-track** conversation | ~5,700 h | **~3 h** | **~5,700 h missing** |
| B. One-person speech | ~5,800 h | **~22,640 h** if VP unlabelled is accepted; **~649 h** if only read/spontaneous open sets are accepted | Hours can exceed the recipe only by counting formal EP audio. Casual/read open speech is much smaller than 5,800 h if EP is excluded. Domain match is still weak (audiobooks, read sentences, parliament) |

**The binding gap is Group A, especially separate-track conversational audio.** There is no open, commercially clear Italian equivalent of Fisher / dual-channel telephone at scale. CLIPS telephone and map-task material is the historically closest academic design, but it is research-only and not confirmed as dual-channel. The only open stereo, per-speaker, commercially licensed Italian conversations found are a **3-hour vendor sample**.

---

## Options to close the Group A gap

### 1. Diarize (or stereo-split where applicable) CC conversational media — mainly YODAS IT (~10,687 h)

Use the Italian YODAS / YODAS2 shards, detect multi-speaker segments, diarize, and optionally keep only high-confidence two-speaker scenes. Some YouTube uploads are already stereo; most are not true separate mics.

- **Legal caveats:** The compilation is CC BY 3.0 and the authors filtered for YouTube CC licences at scrape time, but **YouTube Terms of Service**, licence changes after scrape, “CC” covering the video vs the soundtrack, and whether training a commercial model is within the uploader’s licence are all live issues. This remains **GREY**. YouTube-Commons does not ship audio. Counsel should sign off before any large scrape or YODAS training run. Attribution of original channels is required under CC BY.
- **Quality caveats:** This is not Fisher. Overlap, music, ads, language bleed, auto-captions (it100/it101), and diarization error will all pollute a speech-to-speech model. “Manual” captions are user-uploaded, not guaranteed human. Expect a large yield drop after filtering for actual two-person Italian conversation.

### 2. Synthesize dialogues from unique OK Group B audio (Common Voice, MLS, optionally VoxPopuli)

Write or generate Italian dialogue scripts and render them with TTS / voice conversion / concatenative turn-taking from CC0 / CC BY / CC0-EP voices, producing artificial left/right tracks.

- **Legal caveats:** Cleanest path if every component is OK: CV (CC0, **no speaker identification / no re-hosting**), MLS (CC BY), VoxPopuli data (CC0 + EP acknowledgement). **Do not** use ITALIC if the work involves **cloning those 70 volunteers’ voices** (explicit Zenodo condition). **Do not** use mTEDx, Europarl-ST packaging, KIParla, or MagicHub audio. Any TTS system used must itself be commercially licensable (VoxPopuli **released checkpoints are CC BY-NC**).
- **Quality caveats:** Synthetic turn-taking will miss authentic overlap, backchannels, barging-in, Lombard speech, telephone channel, and informal Italian. This can bootstrap a model but is a poor substitute for thousands of hours of real dual-track conversation. Mixing read-domain (MLS, CV) with formal EP speech will bias style.

### 3. Buy dual-channel telephone / dialogue data, and/or commission a consented stereo collection

Paid catalogues (Nexdata 499 h Italian telephone — **mono** on the public spec; Defined.ai dialogue/podcast products; ELRA PortMedia and similar; custom collection from DataHive or another consented vendor) can add hundreds to low thousands of hours, still short of ~5,700 h unless a large custom collect is funded. DataHive already documents a consented, per-speaker-track pipeline and sells custom collections; their public sample is only 3 h.

- **Legal caveats:** Contracts were **not** reviewed. Do not assume “commercial dataset” means “train and ship a speech-to-speech model, including voice transfer.” Require written rights for model training and distribution, speaker consent that covers that use, and GDPR warranties. The Nexdata page advertises model-training use but the public audio format is **mono**, so it does not by itself close the **separate-track** gap.
- **Quality caveats:** 8 kHz telephone mono (Nexdata as specified) is a poor match to a modern speech-to-speech stack that wants full-band, per-speaker tracks. Prefer documented stereo / multi-mic, 16–48 kHz collections, and inspect overlap and speaker balance. Even 2,000 h of bought dialogue would leave a multi-thousand-hour shortfall vs the English recipe.

A fourth, slower path — **new consented Italian stereo conversations under CC BY or CC0** — is legally the cleanest way to actually reproduce the caramel-1 recipe. Academic corpora (CLIPS, KIParla, LABLITA) should be approached only if the rights holders will negotiate a **commercial training** grant; their public terms do not provide one.

---

## Sources consulted (primary)

- Mozilla Data Collective, Common Voice Scripted Speech 27.0 Italian, and Common Voice Legal Terms  
- facebookresearch/voxpopuli README and HF dataset card; europarl.europa.eu legal notice  
- OpenSLR 94 (MLS), 100 (mTEDx), 146 (CML-TTS), 161 (Emozionalmente); MLS, FLEURS, and CML-TTS papers  
- M-AILABS README (i-celeste-aurora/m-ailabs-dataset)  
- FBK MOSEL page and HF card  
- ESPnet YODAS / YODAS2 dataset cards  
- PleIAs YouTube-Commons card  
- KIParla website and GitHub collection  
- CLIPS project pages and Leipzig / ELRA descriptions  
- LABLITA “Speech Resources” paper (Cresti et al.) and ORFEO/LABLITA portals  
- ELRA catalogue entries (C-ORAL-ROM, PortMedia, and others)  
- MagicHub ASR-ItaCSC page  
- Zenodo / arXiv ITALIC  
- DataHive Italian Conversations sample card  
- Nexdata Italian telephone product page  
- VoxForge / MDC VoxForge Italian datasheet  
- Emilia dataset card (language list)  
- Europarl-ST README  
- Giordano & Rinaldi, “A Survey on Spoken Italian Datasets and Corpora” (arXiv:2501.06557) and [inventory CSV](https://github.com/marco-giordano/spoken-italian-datasets) — used only as a finding aid; hours/licences were checked at sources above  

The official CLIPS portal and the live ELRA C-ORAL-ROM catalogue page did not return complete content when fetched (timeout / 504). Defined.ai product pages did not return licence or hour text. Those gaps are marked UNKNOWN rather than filled from memory.
