# RunPod pilot

One H100 SXM pod, one Italian training pass, then the pod is deleted.
Nothing here rents a machine until you pass `--launch`.

The retry uses **one GPU** and a smaller pool: **100 hours** of MLS/FLEURS
mono, **80 hours** of synthetic stereo, and the **3 hour** DataHive sample
(about 2.9 h of that stays in training). Those sizes are
`runpod/pilot_config.py`. The pass targets are the same as the 2-GPU plan,
so the step counts change: stage 1 stops at **156**, stage 2 ends at **622**,
stage 3 ends at **1254**. Keeping 401 / 1072 / 1981 would train a different
number of passes on this smaller pool and on one GPU.

The owner approved **$25** total. This retry stops at **2.5 hours or $12**,
whichever comes first. At $4.39 per hour the 2.5 hour mark bills **$10.98**,
so the clock hits first. A run that finishes in about 2 hours is about **$9**.

## What you need

- A RunPod API key in `RUNPOD_API_KEY`. The same account is enough. This
  does not create a new account.
- The repository is public, so the pod downloads `runpod/bootstrap.sh` from
  GitHub. No GitHub token is required.

Common Voice is not downloaded. It needs a Mozilla Data Collective account.

## Option A ($9, one H100, the open pools)

This does not spend money:

```bash
python3 runpod/launch.py --config optionA --dry-run
```

The dry run prints a phase estimate from the retry's measured rates and
refuses the plan when that GPU estimate is above $8. The pod, if you
later launch it, is one Secure H100 SXM. The hard cap is **$9.00** or
**1.9 hours**, whichever comes first. Container disk is 250 GB at
$0.10/GB/month while the pod is running, and there is no volume disk.

```bash
python3 runpod/launch.py --config optionA --launch
```

`--launch` is the command that rents the machine. The run uses all of
MLS Italian, FLEURS Italian, and Emozionalmente, about 218 h of synthetic
stereo, and the same DataHive hold-out as the retry. It starts from
scratch: the retry checkpoint does not contain the optimizer state
`train.py` needs to resume.

## Warm start from caramel-1 ($3)

This continues from the English caramel-1 weights and the Option A token
stores. It does not download audio and it does not tokenize. The dry run
does not spend money:

```bash
python3 runpod/launch.py --config warmstart --stores runpod/retrieved/mviebvzw3ht7hn/stores --dry-run
```

The pod, if you launch it, downloads `naklecha/caramel-1` itself (public,
Apache 2.0). The stores are uploaded in 32 MB pieces. The hard cap is
**$3.00** or **38 minutes**, whichever comes first. The dry run refuses
the plan when the GPU estimate is above $2.50.

```bash
python3 runpod/launch.py --config warmstart --stores runpod/retrieved/mviebvzw3ht7hn/stores --launch
```

## Check, then launch

From the repo root:

```bash
python runpod/launch.py --dry-run
```

That checks the step counts, the dollar math, the shell scripts, and that
the public data URLs answer. It does not send your API key and it does not
create a pod.

```bash
export RUNPOD_API_KEY=...          # already issued by RunPod
export CARAMEL_REPO_REF=main       # the branch whose bootstrap.sh the pod curls
python runpod/launch.py --launch
```

`--launch` is the command that spends money. With no flag, the script only
does the dry run.

The pod is one Secure Cloud **NVIDIA H100 80GB HBM3** (the SXM 80 GB part).
The image is `runpod/pytorch:2.2.0-cu1281-torch291-ubuntu2404`: Ubuntu 24.04,
CUDA 12.8.1, PyTorch 2.9.1 built for CUDA 12.8. That matches
`third_party/caramel/setup.sh` (Python 3.12, torch 2.9.1, CUDA 12.8+).
The pod asks for a host CUDA version of at least 12.8.

Container disk is 160 GB. `volumeInGb` is 0 and there is no network volume.
Raw MLS and FLEURS files are deleted after the dialogues are built.
`terminateAfter` is set to 2.5 hours on the RunPod side as well.

## How the money cap works

Quoted rate: **$4.39 per pod-hour** (439 cents), one GPU.

The clock starts when the create call returns. Both the launcher and a
loop inside the pod compute the bill the same way, in whole cents, rounded up:

```text
cents = (elapsed_seconds * hourly_cents + 3599) / 3600
```

They terminate when either of these is true:

- elapsed time is **9000 seconds** (2.5 h), or
- the rounded bill reaches **1200 cents** ($12).

At $4.39/h the 2.5 hour mark is $10.98, so the **clock hits first**. If the
live `costPerHr` on the pod is higher, the $12 mark hits first. Example: at
$20/h the dollar cap fires at about 36 minutes.

Before create, the launcher reads the Secure Cloud price from GraphQL. It
refuses to launch if that price is above **$4.90 per GPU-hour**. One GPU at
$4.90 for 2.5 h is $12.25, just over the $12 cap, so a quote above $4.90
needs a new yes. The user's API key is not copied into the pod. RunPod
injects a pod-scoped key, and the in-pod watchdog uses that key to
terminate.

While the pod is up the launcher prints a line about every 20 seconds:

```text
[cost] 1200s  $1.47  rate $4.39/h  status RUNNING  via graphql
```

The cap **terminates** the pod (`podTerminate`). It does not stop it. Stop
can leave disk billing. Terminate deletes the pod and its container disk.
There is no network volume to outlive that.

Three copies of the cap run at once:

1. The launcher, watching the clock and calling `podTerminate`.
2. `runpod/watchdog.sh` inside the pod, same formula, same terminate.
3. `terminateAfter` on the create call, so RunPod itself deletes the pod
   at 2.5 hours even if both of the above die.

`podStop` is implemented and is not used by the cap.

## Which API

All four operations use GraphQL at `https://api.runpod.io/graphql`:

| Action | GraphQL |
|---|---|
| Create | `podFindAndDeployOnDemand` |
| Status | `pod` |
| Stop (not used by the cap) | `podStop` |
| Terminate | `podTerminate` |

The price check reads `securePrice`, plus the public `uninterruptablePrice` and `stockStatus`. It does not request live stock counts. If one of those extra fields is rejected and `securePrice` is still in the response, the quote is kept.

The header is `Authorization: Bearer $RUNPOD_API_KEY`. If a GraphQL call
fails, that one call is retried against `https://api.runpod.io/v2`. The
old REST host is not used.

## What the pod runs

The container command is `bash -c` with one script argument, sent as JSON
`{"entrypoint":["/bin/bash","-c"],"cmd":["..."]}` so RunPod does not split it
on spaces. That script curls `runpod/bootstrap.sh` from this public repo,
starts it with `nohup`, then `exec /start.sh`. `/start.sh` stays the main
process, so sshd and the HTTP proxy stay up even if the job exits.
`bootstrap.sh` starts the watchdog, clones the repo with the caramel
submodule, and runs `pod_run.sh --full`.

`pod_run.sh` then:

1. Installs the same Python packages as `setup.sh` (torch 2.9.1 cu128,
   transformers 4.46.3, and the rest). It does **not** download the English
   `naklecha/caramel-1` weights. Training starts from scratch, and
   `model.py` exports the Italian checkpoint for `generate.py`.
2. Downloads MLS Italian, FLEURS Italian, and the DataHive sample.
3. Builds synthetic left/right dialogues (gap, no fake overlap) and holds
   the shortest DataHive conversation out for dev. The next shortest is
   used only to fit timing bands.
4. Encodes with Mimi (8 codebooks, 24 kHz) and writes `speed.json`
   (frames/s and how many times faster than realtime). Audio is loaded
   and resampled on a CPU worker pool, and tokens are flushed in batches
   so a crash can resume the same store. After tokenize, and after each
   training stage, `stores/` and that stage's `ckpt.pt` are copied into
   the pull directory.
5. Runs the three stages in `train/README.md`.
6. Exports the best checkpoint, generates four 8-second Italian self-play
   clips, and writes timing stats plus Italian ASR word clarity
   (faster-whisper small, mean log-probability) when that install works.
7. Serves `/workspace/pilot-out` on port 8080 until the launcher has copied
   it or ten minutes pass, then terminates.

If tokenization is still running with under 20 minutes left on the cap, it
stops adding clips so training and the samples still happen.

## Getting the files back

No new account and no bucket. The launcher copies `/workspace/pilot-out`
before it terminates. Direct `ip:port` and SSH are blocked by the
launcher's egress proxy, so the first try is the RunPod HTTP proxy:

`https://<pod-id>-8080.proxy.runpod.net/`

Each request sends `X-Pilot-Token`. The pod deletes `PULLED` at startup
and again just before it writes `DONE`. `/ack` writes `PULLED` only when
`DONE` or `FAILED` already exists and the ack is newer than that file.
A failure keeps the pod up for the full 10 minutes so the error can be
copied; an ack during that wait does not end it early. SSH/scp is only a
last resort. The folder is not world-readable without the token, and the
pod is deleted after the copy.

If the launcher dies, `bash runpod/supervise.sh` starts `runpod/watch.py`
again. A network or TLS timeout is retried. The watcher calls terminate
only after three readings in a row say the pod is gone, and only after
`myself { pods }` also does not list it. One empty status does not
terminate a healthy pod.

A later attempt can skip download and tokenize:

```bash
python runpod/launch.py --launch --resume-from runpod/retrieved/<old-pod-id>
```

That uploads `stores/` and `checkpoints/` through the pull server once the
new pod is answering, then the pod copies them into place.

If 12 minutes after create the container uptime is still 0 or less, or
that proxy URL has not answered, the launcher terminates the pod and
exits non-zero. A crash loop must not run until the 2.5 hour cap.

What comes back: `best.pt` (or the safetensors export), four sample wavs,
logs, `speed.json`, `eval.json`, and the attribution note. The optimizer
checkpoint stays on the pod and is deleted with it. Raw audio is not copied.

Files land in `runpod/retrieved/<pod-id>/`.

## Rehearsed on 27 Nov 2026

`python runpod/launch.py --dry-run` finished without sending an API key.
The nine public data URLs answered HTTP 200. `bash runpod/pod_run.sh --local-slice`
then ran on this CPU: about 40 seconds each of MLS Italian, FLEURS Italian test,
and the shortest DataHive conversation; four synthetic dialogues; real Mimi
codes (not placeholders); upstream `train.py` for a stop, a resume, and
`--save_best`; `model.py` export; and one prompted `generate.py` clip (3.0 s).
Timing stats ran on that clip. faster-whisper was not installed, so the ASR
step took the skip path. No pod was created.

## A free rehearsal on your machine

```bash
bash runpod/pod_run.sh --local-slice
```

That downloads about 40 seconds from each open source (not the 15 GB MLS
archive), builds a tiny synthetic dialogue, encodes it if Mimi loads on
CPU, and runs a few upstream `train.py` steps with a tiny `--cfg`. Upstream
does run on CPU. The tiny network only checks the wiring. The rented run
does not pass `--cfg`.

## What this run is not

It is a short pilot, not the English 11,000-hour recipe. Synthetic
dialogues do not contain real overlap or backchannels. All three stages
use the one GPU the pod bills for. A slow codec can still eat the cap,
which is why the speed log and the 20-minute reserve exist.

## Time and cost for this retry

The failed 2-GPU run spent about 50 minutes before tokenize finished
(download about 35, synthetic audio about 15) and about 24 minutes on
mono tokenize, which was still short of the full mono pool. About three
quarters of that tokenize time was CPU read and resample, not the GPU.

Scaled to one GPU and to 100 h mono / 80 h synthetic stereo, a cautious
estimate is:

- prep (install, download, synth, delete raw): about 30–40 minutes
- mono tokenize: about 20–30 minutes
- stereo tokenize (about 83 h, one GPU): about 25–40 minutes
- the three stages (1254 steps), export, four samples, eval: about 20–30 minutes
- pull of stores and checkpoints along the way: a few minutes

That is about **1.6–2.2 hours**, or about **$7–$10** at $4.39/h. The
$12 cap and the 2.5 hour wall ($10.98) both sit above that range. The
CPU worker pool should help the resample portion; this estimate does not
assume it makes tokenize several times faster. It is scaled from the
failed run, not a new measurement.
