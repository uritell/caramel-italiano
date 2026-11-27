"""One-sided live turn of the caramel model.

Each call is one 80 ms frame (1920 samples at 24 kHz). The person is the
"other" speaker: their 8 Mimi codebooks are fed in. The model is the
speaker being modelled: Model.step samples its own 8 codebooks (forced is
-2) unless the person is talking. While they are, and for a short hangover
after, those 8 codes are forced to a silence frame and the played audio is
zeros. The person's codes still go in on the other side. Batch size is 1.

Acoustic codebooks are one frame behind codebook 0, the same delay as
generate.py. The first frame therefore returns no audio. The next frame
returns the previous one, aligned, and Mimi stream-decodes it.

The context is model.cfg.context frames (3000 = 240 s). When it fills,
the caches are cleared in place and the frame that did not fit starts a
new conversation. A CUDA graph is recorded when the device is CUDA; the
reset keeps the same tensors so the graph stays valid.
"""
from __future__ import annotations

import inspect
import math
import time
from dataclasses import dataclass

import numpy as np
import torch

SR = 24000
FRAME = 1920  # 80 ms at 24 kHz. Mimi emits one frame of 8 codes per chunk.
# Quieter than the old generate.py defaults (1.1 / 50), which talked over the person.
DEFAULT_TEMP = 0.75
DEFAULT_TOP_K = 25
DEFAULT_VAD_RMS = 0.02
DEFAULT_HANGOVER_MS = 500


@dataclass
class FrameResult:
    pcm: np.ndarray
    ms: float
    frame_index: int
    minutes_left: float
    reset: bool
    status: str


def _device_of(model) -> torch.device:
    emb = getattr(model, "emb", None)
    if emb is not None and hasattr(emb, "weight"):
        return emb.weight.device
    return torch.device("cpu")


def _delays(cfg) -> list[int]:
    raw = list(getattr(cfg, "delays", None) or ([0] + [1] * 7))
    if len(raw) < 8:
        raw = raw + [1] * (8 - len(raw))
    return [int(d) for d in raw[:8]]


def delay_frame(aligned: torch.Tensor, prev: torch.Tensor | None, delays: list[int], card: int) -> torch.Tensor:
    """Aligned 8 codes -> delayed 8 tokens for this frame. Delay 0 is current, delay 1 is the previous frame."""
    out = torch.empty(8, dtype=torch.long, device=aligned.device)
    for k, delay in enumerate(delays):
        if delay <= 0:
            out[k] = aligned[k]
        elif prev is None:
            out[k] = card
        else:
            out[k] = prev[k]
    return out


def aligned_from_delayed(prev: torch.Tensor | None, curr: torch.Tensor, delays: list[int], card: int) -> torch.Tensor | None:
    """Undo the delay. Codebook k of the frame that just finished sits at delayed index t-1+delay."""
    if prev is None:
        return None
    aligned = torch.empty(8, dtype=torch.long, device=curr.device)
    for k, delay in enumerate(delays):
        src = curr[k] if delay == 1 else prev[k] if delay == 0 else None
        if src is None:
            return None
        if int(src) < 0 or int(src) >= card:
            return None
        aligned[k] = src
    return aligned


class _CodecState:
    def __init__(self) -> None:
        self.enc_cache = None
        self.enc_pad = None
        self.dec_cache = None
        self.dec_pad = None

    def clear(self) -> None:
        self.enc_cache = self.enc_pad = self.dec_cache = self.dec_pad = None


def _fill(kwargs: dict, params, names: tuple[str, ...], value) -> None:
    for name in names:
        if name in params:
            kwargs[name] = value
            return


def encode_frame(mimi, wave: torch.Tensor, state: _CodecState) -> torch.Tensor:
    """Stream-encode [1, 1, 1920] to 8 codes. Uses padding_cache when this Mimi has it."""
    params = inspect.signature(mimi.encode).parameters
    kwargs = {}
    if "num_quantizers" in params:
        kwargs["num_quantizers"] = 8
    if "use_streaming" in params:
        kwargs["use_streaming"] = True
    if "padding_cache" in params:
        kwargs["padding_cache"] = state.enc_pad
    _fill(kwargs, params, ("encoder_cache", "encoder_past_key_values"), state.enc_cache)
    out = mimi.encode(wave, **kwargs)
    codes = out.audio_codes if hasattr(out, "audio_codes") else out[0]
    if hasattr(out, "padding_cache"):
        state.enc_pad = out.padding_cache
    state.enc_cache = getattr(out, "encoder_cache", None)
    if state.enc_cache is None:
        state.enc_cache = getattr(out, "encoder_past_key_values", None)
    if codes.dim() == 2:
        codes = codes[:, -1]
    else:
        codes = codes[0, :, -1]
    return codes.to(dtype=torch.long).view(8)


def decode_frame(mimi, codes: torch.Tensor, state: _CodecState) -> torch.Tensor:
    """Stream-decode 8 aligned codes to PCM. codes is [8]."""
    params = inspect.signature(mimi.decode).parameters
    kwargs = {}
    if "use_streaming" in params:
        kwargs["use_streaming"] = True
    if "padding_cache" in params:
        kwargs["padding_cache"] = state.dec_pad
    _fill(kwargs, params, ("decoder_cache", "decoder_past_key_values"), state.dec_cache)
    out = mimi.decode(codes.view(1, 8, 1), **kwargs)
    audio = out.audio_values if hasattr(out, "audio_values") else out[0]
    if hasattr(out, "padding_cache"):
        state.dec_pad = out.padding_cache
    state.dec_cache = getattr(out, "decoder_cache", None)
    if state.dec_cache is None:
        state.dec_cache = getattr(out, "decoder_past_key_values", None)
    return audio[0, 0] if audio.dim() == 3 else audio.reshape(-1)


def frame_rms(wave: torch.Tensor) -> float:
    x = wave.detach().float().reshape(-1)
    if x.numel() == 0:
        return 0.0
    return float(torch.sqrt((x * x).mean()))


def hangover_frames(hangover_ms: int) -> int:
    """500 ms is 7 frames, because each frame is 80 ms."""
    if hangover_ms <= 0:
        return 0
    return max(1, math.ceil(hangover_ms / 80.0))


class LiveLoop:
    def __init__(
        self,
        model,
        mimi,
        temp: float = DEFAULT_TEMP,
        top_k: int = DEFAULT_TOP_K,
        vad_rms: float = DEFAULT_VAD_RMS,
        hangover_ms: int = DEFAULT_HANGOVER_MS,
        gate: bool = True,
    ):
        self.model = model
        self.mimi = mimi
        self.temp = float(temp)
        self.top_k = max(1, int(top_k))
        self.vad_rms = float(vad_rms)
        self.hangover_ms = int(hangover_ms)
        self.hangover_n = hangover_frames(self.hangover_ms)
        self.gate_enabled = bool(gate)
        self._hang_left = 0
        self._silence = None
        self.cfg = model.cfg
        self.device = _device_of(model)
        self.card = int(self.cfg.card)
        self.context = int(self.cfg.context)
        self.delays = _delays(self.cfg)
        self.st = model.init_state(1, self.context)
        self.inp = torch.full((1, int(self.cfg.n_q)), self.card, dtype=torch.long, device=self.device)
        self.forced = torch.full((1, int(self.cfg.dep_q)), -2, dtype=torch.long, device=self.device)
        self.codec = _CodecState()
        self.graph = None
        self.out = None
        self._reset_flags()
        self._maybe_graph()

    def _reset_flags(self) -> None:
        dev = self.device
        self.prev_model = torch.full((8,), self.card, dtype=torch.long, device=dev)
        self.prev_human = torch.full((8,), self.card, dtype=torch.long, device=dev)
        self.prev_human_aligned = None
        self.prev_model_delayed = None
        self.frame_index = 0
        self._hang_left = 0
        self.codec.clear()

    def _zero_state(self) -> None:
        st = self.st
        pos = st.get("pos")
        if pos is not None:
            pos.zero_()
        for key in ("kc", "vc", "dkc", "dvc"):
            for tensor in st.get(key) or []:
                tensor.zero_()

    def _maybe_graph(self) -> None:
        if self.device.type != "cuda":
            return
        try:
            stream = torch.cuda.Stream()
            stream.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(stream):
                for _ in range(3):
                    self.model.step(self.st, self.inp, self.forced, self.temp, self.top_k)
            torch.cuda.current_stream().wait_stream(stream)
            self.graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(self.graph):
                self.out = self.model.step(self.st, self.inp, self.forced, self.temp, self.top_k)
        except Exception as exc:  # noqa: BLE001
            print(f"CUDA graph skipped: {exc}", flush=True)
            self.graph = None
        self._zero_state()

    def set_sampling(self, temp=None, top_k=None, gate=None, vad_rms=None, hangover_ms=None) -> dict:
        """Apply a live change from the page. A new temperature drops the CUDA graph."""
        if temp is not None and float(temp) != self.temp:
            self.temp = max(0.0, min(2.0, float(temp)))
            self.graph = None
        if top_k is not None and int(top_k) != self.top_k:
            self.top_k = max(1, min(200, int(top_k)))
            self.graph = None
        if gate is not None:
            self.gate_enabled = bool(gate)
            if not self.gate_enabled:
                self._hang_left = 0
        if vad_rms is not None:
            self.vad_rms = max(0.0, float(vad_rms))
        if hangover_ms is not None:
            self.hangover_ms = max(0, int(hangover_ms))
            self.hangover_n = hangover_frames(self.hangover_ms)
        return {
            "temp": self.temp,
            "top_k": self.top_k,
            "gate": self.gate_enabled,
            "vad_rms": self.vad_rms,
            "hangover_ms": self.hangover_ms,
        }

    def _silence_codes(self) -> torch.Tensor:
        """One frame of digital silence, encoded once on a throwaway codec state."""
        if self._silence is not None:
            return self._silence
        try:
            codes = encode_frame(self.mimi, torch.zeros(1, 1, FRAME, device=self.device), _CodecState())
            self._silence = codes.to(device=self.device, dtype=torch.long).view(8).clamp(0, self.card - 1)
        except Exception:  # noqa: BLE001
            self._silence = torch.zeros(8, dtype=torch.long, device=self.device)
        return self._silence

    def _gate_open(self, rms: float) -> bool:
        """True while the person is speaking and for the hangover after they stop."""
        if not self.gate_enabled:
            return False
        if rms >= self.vad_rms:
            self._hang_left = self.hangover_n
            return True
        if self._hang_left > 0:
            self._hang_left -= 1
            return True
        return False

    def _step(self) -> torch.Tensor:
        if self.graph is None:
            return self.model.step(self.st, self.inp, self.forced, self.temp, self.top_k)
        self.graph.replay()
        return self.out

    def reset(self) -> None:
        self._zero_state()
        self._reset_flags()

    def push(self, samples) -> FrameResult:
        """One 1920-sample frame. Returns the model's PCM for the frame that just became complete."""
        wave = _as_frame(samples, self.device)
        reset = False
        if self.frame_index >= self.context:
            self.reset()
            reset = True
        rms = frame_rms(wave)
        gated = self._gate_open(rms)
        aligned = encode_frame(self.mimi, wave, self.codec).to(self.device)
        human_delayed = delay_frame(aligned, self.prev_human_aligned, self.delays, self.card)
        # Previous delayed tokens: model on the left (the side step samples), person on the right.
        # The person's codes stay on the right even while the gate mutes the model.
        self.inp[0, :8] = self.prev_model
        self.inp[0, 8:16] = self.prev_human
        if gated:
            self.forced[0].copy_(self._silence_codes())
        else:
            self.forced.fill_(-2)
        if self.device.type == "cuda":
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        out = self._step()[0]
        if self.device.type == "cuda":
            torch.cuda.synchronize()
        ms = (time.perf_counter() - t0) * 1e3
        ready = aligned_from_delayed(self.prev_model_delayed, out, self.delays, self.card)
        pcm = np.zeros(0, dtype=np.float32)
        if ready is not None and gated:
            pcm = np.zeros(FRAME, dtype=np.float32)
        elif ready is not None:
            audio = decode_frame(self.mimi, ready, self.codec).detach().float().cpu().numpy().reshape(-1)
            pcm = np.ascontiguousarray(audio, dtype=np.float32)
        self.prev_model_delayed = out.detach().clone()
        self.prev_model = out.detach().clone()
        self.prev_human = human_delayed.detach().clone()
        self.prev_human_aligned = aligned.detach().clone()
        self.frame_index += 1
        left = max(0.0, (self.context - self.frame_index) * 0.08 / 60.0)
        if reset:
            status = "The 4 minute memory is full. Starting a fresh conversation."
        elif gated:
            status = "Waiting while you talk."
        else:
            status = "Listening."
        print(
            f"frame {self.frame_index} {ms:.2f} ms/frame "
            f"minutes_left={left:.2f} reset={int(reset)} gate={int(gated)}",
            flush=True,
        )
        return FrameResult(pcm, ms, self.frame_index, left, reset, status)


def _as_frame(samples, device: torch.device) -> torch.Tensor:
    if isinstance(samples, torch.Tensor):
        wave = samples.detach().float().reshape(-1)
    else:
        wave = torch.from_numpy(np.ascontiguousarray(samples, dtype=np.float32)).reshape(-1)
    if wave.numel() != FRAME:
        raise ValueError(f"expected {FRAME} samples, got {wave.numel()}")
    return wave.to(device).view(1, 1, FRAME)
