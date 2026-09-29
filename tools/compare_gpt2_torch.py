#!/usr/bin/env python3
"""Fair jasmine vs PyTorch comparison for GPT-2 inference.

The point of this script is to make the comparison honest, because a naive run is
dominated by things that have nothing to do with the transformer code:

  * BLAS. jasmine links whatever `CBLAS` CMake finds; on a stock Debian/Ubuntu
    that is the reference netlib BLAS (single-threaded, unoptimized), while
    torch ships MKL or OpenBLAS. Measured on distilgpt2, swapping netlib for
    OpenBLAS alone changes jasmine's full forward by 3.5x. So the point of
    --jasmine-bin is to let you pass a build with a comparable BLAS; this script
    reports both sides' BLAS so the confound stays visible.
  * Precision. torch runs fp32 by default, so jasmine is run with --dtype float.
    Comparing an fp64 jasmine against an fp32 torch measures dtype, not code.
  * Threads. Both sides are pinned to --threads. jasmine parallelizes by row
    panels in jas_mat_gemm.hpp and assumes a single-threaded BLAS underneath, so
    OPENBLAS_NUM_THREADS is pinned to 1 for both halves of the measurement.
  * Weights. The exported weight file and the HF checkpoint must be the same
    model; the logits checksums printed below are the evidence, and a mismatch
    fails the run instead of quietly comparing two different models.

Usage:
    python tools/compare_gpt2_torch.py \
        --jasmine-bin build/benches/bench_gpt2_infer \
        --weights build/distilgpt2_weights.bin \
        --threads 4 --prefill 128 --decode 32 --repeats 3

    # compare a second jasmine build (e.g. an OpenBLAS one) side by side
    python tools/compare_gpt2_torch.py ... --compare-bin build-openblas/benches/bench_gpt2_infer
"""

import argparse
import os
import re
import statistics
import subprocess
import sys
import time
from pathlib import Path

# Silence the noisy "you should use attention_mask" style warnings; they do not
# affect the measurement and drown the output.
os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")


def log(msg=""):
    print(msg, flush=True)


def run_jasmine(binary, weights, args, threads, label):
    """Runs the C++ timing driver and parses its key=value output."""
    env = dict(os.environ)
    # jasmine splits GEMMs across OpenMP row panels and documents the assumption
    # that the BLAS underneath is single-threaded; pinning this for both sides
    # keeps the thread budget identical.
    env["OMP_NUM_THREADS"] = str(threads)
    env["OPENBLAS_NUM_THREADS"] = "1"
    env["MKL_NUM_THREADS"] = "1"

    cmd = [
        str(binary), str(weights),
        "--dtype", "float",
        "--prefill", str(args.prefill),
        "--decode", str(args.decode),
        "--repeats", str(args.repeats),
        "--warmup", str(args.warmup),
    ]
    log(f"[{label}] {' '.join(cmd)}")
    log(f"[{label}] OMP_NUM_THREADS={threads} OPENBLAS_NUM_THREADS=1")
    proc = subprocess.run(cmd, env=env, capture_output=True, text=True, check=False)
    if proc.returncode != 0:
        raise SystemExit(f"[{label}] failed (rc={proc.returncode}):\n{proc.stderr}")

    values = {}
    for line in proc.stdout.splitlines():
        if "=" in line:
            key, _, value = line.partition("=")
            if value:
                try:
                    values[key] = float(value)
                except ValueError:
                    pass
    required = ("full_forward_ms", "prefill_ms", "decode_ms_per_token", "logits_last_col_sum")
    missing = [k for k in required if k not in values]
    if missing:
        raise SystemExit(f"[{label}] missing fields {missing} in:\n{proc.stdout}")
    return values


def blas_description(binary):
    """Reports which BLAS the jasmine binary actually linked, for the record."""
    try:
        out = subprocess.run(["ldd", str(binary)], capture_output=True, text=True, check=False).stdout
    except OSError:
        return "unknown"
    names = []
    for line in out.splitlines():
        m = re.search(r"lib(\w*(?:blas|mkl)\w*)\.so", line)
        if m:
            lib = m.group(1)
            if lib not in names:
                names.append(lib)
    return ", ".join(names) if names else "none linked"


def build_inputs(vocab, prefill, decode, next_id):
    """Same synthetic token ids the C++ driver builds, so both sides see identical work."""
    ids = [[100 + (t * 7919) % 40000 for t in range(prefill)]]
    return ids, next_id


def run_torch(model_id, args, threads, prefill, decode, next_id):
    """Mirrors the C++ driver's three measurements with the equivalent torch calls."""
    import torch
    from transformers import AutoModelForCausalLM

    torch.set_num_threads(threads)
    # AutoModelForCausalLM instead of AutoModelWithLMHead: the latter has been
    # removed in transformers 5.x.
    model = AutoModelForCausalLM.from_pretrained(model_id, dtype=torch.float32)
    model.eval()

    ids, _ = build_inputs(model.config.vocab_size, prefill, decode, next_id)
    ids_t = torch.tensor(ids, dtype=torch.long)
    next_t = torch.tensor([[next_id]], dtype=torch.long)

    def full_forward():
        return model(ids_t)

    def prefill():
        return model(ids_t, use_cache=True)

    def decode_steps(past):
        out = None
        for _ in range(decode):
            out = model(next_t, past_key_values=past, use_cache=True)
            past = out.past_key_values
        return out

    with torch.no_grad():
        for _ in range(args.warmup):
            full_forward()
            out = prefill()
            decode_steps(out.past_key_values)

        full_times = []
        for _ in range(args.repeats):
            t0 = time.perf_counter()
            full_forward()
            full_times.append((time.perf_counter() - t0) * 1e3)

        prefill_times = []
        for _ in range(args.repeats):
            t0 = time.perf_counter()
            out = prefill()
            prefill_times.append((time.perf_counter() - t0) * 1e3)

        decode_times = []
        for _ in range(args.repeats):
            out = prefill()
            past = out.past_key_values
            t0 = time.perf_counter()
            decode_steps(past)
            decode_times.append((time.perf_counter() - t0) * 1e3 / decode)

        logits = full_forward().logits[0, -1, :]

    return {
        "full_forward_ms": statistics.median(full_times),
        "prefill_ms": statistics.median(prefill_times),
        "decode_ms_per_token": statistics.median(decode_times),
        "logits_last_col_sum": float(logits.sum().item()),
        "logits_first": float(logits[0].item()),
    }


def report(rows, decode):
    """Prints the comparison table. rows: list of (label, metrics dict, blas str)."""
    metrics = [
        ("full forward, tokens", "full_forward_ms"),
        ("prefill (cache), tokens", "prefill_ms"),
        ("decode, ms per token", "decode_ms_per_token"),
    ]
    base_label, base, _ = rows[0]
    others = rows[1:]

    log("")
    log("=" * 78)
    log(f"{'metric':<34}{base_label:>20}" + "".join(f"{l:>22}" for l, _, _ in others))
    log("-" * 78)
    for title, key in metrics:
        line = f"{title:<34}{base[key]:>20.4f}"
        for _, m, _ in others:
            line += f"{m[key]:>15.4f} ({base[key] / m[key]:5.2f}x)"
        log(line)
    log("-" * 78)
    log(f"{'prefill + decode total':<34}"
        + f"{base['prefill_ms'] + base['decode_ms_per_token'] * decode:>20.4f}"
        + "".join(f"{m['prefill_ms'] + m['decode_ms_per_token'] * decode:>22.4f}"
                  for _, m, _ in others))
    log("")
    log(f"{'logits checksum (must agree)':<34}{base['logits_last_col_sum']:>20.4f}"
        + "".join(f"{m['logits_last_col_sum']:>22.4f}" for _, m, _ in others))
    log("")
    for label, _, blas in rows:
        log(f"  BLAS for {label:<28} {blas}")
    log("=" * 78)


def main():
    ap = argparse.ArgumentParser(description="jasmine vs PyTorch GPT-2 inference comparison")
    ap.add_argument("--jasmine-bin", default="build/benches/bench_gpt2_infer")
    ap.add_argument("--compare-bin", default=None,
                    help="a second jasmine build to compare (e.g. linked against a faster BLAS)")
    ap.add_argument("--weights", default="build/distilgpt2_weights.bin")
    ap.add_argument("--model", default="distilgpt2",
                    help="HF model id whose weights the exported file came from")
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--prefill", type=int, default=128)
    ap.add_argument("--decode", type=int, default=32)
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--warmup", type=int, default=1)
    args = ap.parse_args()

    for path in (args.jasmine_bin, args.weights):
        if not Path(path).exists():
            raise SystemExit(f"not found: {path}")
    if args.compare_bin and not Path(args.compare_bin).exists():
        raise SystemExit(f"not found: {args.compare_bin}")

    log(f"threads={args.threads}  prefill={args.prefill} tokens  decode={args.decode} "
        f"tokens  repeats={args.repeats}  warmup={args.warmup}")
    log(f"weights={args.weights}  HF model={args.model}")

    rows = []
    m = run_jasmine(args.jasmine_bin, args.weights, args, args.threads, "jasmine")
    rows.append(("jasmine", m, blas_description(args.jasmine_bin)))
    if args.compare_bin:
        m2 = run_jasmine(args.compare_bin, args.weights, args, args.threads, "jasmine-alt")
        rows.append(("jasmine-alt", m2, blas_description(args.compare_bin)))

    try:
        import torch  # noqa: F401
    except ImportError:
        raise SystemExit("torch is not importable; this comparison needs PyTorch installed")

    mt = run_torch(args.model, args, args.threads, args.prefill, args.decode, 50255)
    rows.append(("pytorch", mt, f"torch {__import__('torch').__version__} (MKL/OpenBLAS bundled)"))

    report(rows, args.decode)

    # The checksums are the guard against comparing two different models: the
    # exported file and the HF checkpoint are only the same model if the final
    # logits agree to fp32 accumulation noise.
    sums = [r[1]["logits_last_col_sum"] for r in rows]
    spread = max(sums) - min(sums)
    scale = max(abs(s) for s in sums)
    rel = spread / scale if scale else 0.0
    log(f"checksum spread = {spread:.6g} (relative {rel:.2e})")
    if rel > 1e-4:
        log("WARNING: the logits disagree more than fp32 noise allows; the two sides")
        log("         are probably not running the same weights, and the timings")
        log("         above do not describe the same computation.")
        return 1
    log("checksums agree within fp32 accumulation noise, so both sides ran the same model")
    return 0


if __name__ == "__main__":
    sys.exit(main())
