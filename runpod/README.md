# RunPod pilot

One 2× H100 SXM pod, one Italian training pass, then the pod is deleted.
Nothing here rents a machine until you pass `--launch`.

The owner approved **$25** total. The scripts stop at **2.5 hours or $22**,
whichever comes first. At the price checked on 27 Nov 2026 ($4.39 per GPU-hour
on Secure Cloud) that is about **$21.95** if the run uses the whole 2.5 hours.
A two-hour run is about **$18**.

## What you need

- A RunPod API key in `RUNPOD_API_KEY`. The same account is enough. This
  does not create a new account.
- An SSH public key already saved on that RunPod account, if you want the
  launcher to copy files with `scp`. Without it, the launcher uses the pod's
  HTTP port and a one-time token instead.
- If this GitHub repo is private, a read token you already have in
  `GITHUB_TOKEN`. The caramel submodule is public.

Common Voice is not downloaded. It needs a Mozilla Data Collective account.

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
export GITHUB_TOKEN=...            # only if the repo is private
export CARAMEL_REPO_REF=main       # or the branch that contains runpod/
export PILOT_SSH_KEY=~/.ssh/id_ed25519   # optional, for scp
python runpod/launch.py --launch
```

`--launch` is the command that spends money. With no flag, the script only
does the dry run.

The pod is a Secure Cloud **2× NVIDIA H100 80GB HBM3** (the SXM 80 GB part).
The image is `runpod/pytorch:2.2.0-cu1281-torch291-ubuntu2404`: Ubuntu 24.04,
CUDA 12.8.1, PyTorch 2.9.1 built for CUDA 12.8. That matches
`third_party/caramel/setup.sh` (Python 3.12, torch 2.9.1, CUDA 12.8+).
The pod asks for a host CUDA version of at least 12.8.

Container disk is 120 GB. `volumeInGb` is 0 and there is no network volume.
`terminateAfter` is set to 2.5 hours on the RunPod side as well.

## How the money cap works

Quoted rate: **$4.39 × 2 GPUs = $8.78 per pod-hour** (878 cents).

The clock starts when the create call returns. Both the launcher and a
loop inside the pod compute the bill the same way, in whole cents, rounded up:

```text
cents = (elapsed_seconds * hourly_cents + 3599) / 3600
```

They terminate when either of these is true:

- elapsed time is **9000 seconds** (2.5 h), or
- the rounded bill reaches **2200 cents** ($22).

At $8.78/h the 2.5 hour mark is $21.95, so the **clock hits first**, about
17 seconds before the $22 mark. If the live `costPerHr` on the pod is higher,
the $22 mark hits first and the run is shorter. Example: at $20/h the dollar
cap fires at about 1 hour 6 minutes.

Before create, the launcher reads the Secure Cloud price from GraphQL. It
refuses to launch if that price is above **$4.90 per GPU-hour**. Two GPUs at
$4.90 for 2.5 h would be $24.50, still under the $25 approval, but a higher
quote needs a new yes.

While the pod is up the launcher prints a line about every 20 seconds:

```text
[cost] 1200s  $2.93  rate $8.78/h  status RUNNING  via graphql
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

`runpod/bootstrap.sh` is the container command. It starts sshd from the
`PUBLIC_KEY` RunPod injects, starts the watchdog, clones this repo with
the caramel submodule, then runs `pod_run.sh --full`.

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
   (frames/s and how many times faster than realtime). Two GPUs split the
   clip list when both are free.
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
before it terminates:

1. **scp** to the pod's public port 22, or to `ssh.runpod.io` with the
   pod host id, using the SSH key already on your RunPod account. This is
   the path to use. Direct SSH can copy files. The proxy hop is only a
   shell, so the script prefers the public port.
2. If scp does not connect, **HTTP**: the pod serves the same folder on
   port 8080. Each request must send `X-Pilot-Token`. The launcher prints
   the token once. The folder is not world-readable without it, and the
   pod is deleted after the copy.

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
dialogues do not contain real overlap or backchannels. Stage 1 uses one
GPU while the pod bills for two. Tokenizing uses both GPUs when it can;
a slow codec can still eat the cap, which is why the speed log and the
20-minute reserve exist.
