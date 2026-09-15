#!/usr/bin/env python3
"""一条命令把 jasmine 的 GPT-2 推理结果与 HuggingFace 对比。

这是判断「jasmine 的 GPT-2 能不能正常运行」的端到端验证入口：
同一份权重、同一个 prompt，分别跑 jasmine 和 HF，逐 token 比对生成结果；
`--logits` 再把同一段序列喂给两边做一次 forward，比对完整 logits。

用法：
    python tools/verify_gpt2.py --text "The capital of France is"
    python tools/verify_gpt2.py --model gpt2 --text "The capital of France is"
    python tools/verify_gpt2.py --text "Hello" --max-new 0 --logits   # 只比 logits
    python tools/verify_gpt2.py --sample --text "Once upon a time" --logits
    python tools/verify_gpt2.py --prompt-ids-file build/prompt_ids.txt

退出码：0 = 一致；1 = 不一致或出错（便于进 CI）。
"""

import argparse
import subprocess
import sys
from pathlib import Path

import numpy as np

from jasmine_weights import WeightReader


def log(msg=""):
    print(msg, flush=True)


class HF:
    """按需加载 HF 模型 / tokenizer（只加载一次）。"""

    def __init__(self, model_id):
        self.model_id = model_id
        self._model = None
        self._tok = None

    @property
    def tokenizer(self):
        if self._tok is None:
            from transformers import AutoTokenizer
            self._tok = AutoTokenizer.from_pretrained(self.model_id)
        return self._tok

    @property
    def model(self):
        if self._model is None:
            import torch
            from transformers import AutoModelForCausalLM
            m = AutoModelForCausalLM.from_pretrained(self.model_id, dtype=torch.float32)
            m.eval()
            self._model = m
        return self._model

    def generate(self, prompt_ids, opt):
        import torch
        ids_in = torch.tensor([prompt_ids])
        if opt.max_new <= 0:
            return list(prompt_ids)
        kwargs = dict(max_new_tokens=opt.max_new, pad_token_id=self.tokenizer.eos_token_id)
        if opt.greedy:
            kwargs["do_sample"] = False
        else:
            kwargs.update(do_sample=True, temperature=opt.temperature,
                          top_k=opt.top_k or 50)
            torch.manual_seed(opt.seed)
        with torch.no_grad():
            return self.model.generate(ids_in, **kwargs)[0].tolist()

    def forward_logits(self, ids):
        """对给定 id 序列做一次 forward，返回 [vocab, T]。"""
        import torch
        with torch.no_grad():
            out = self.model(torch.tensor([list(ids)]))
        return out.logits.float().numpy()[0].T.astype(np.float64)


def run_jasmine(binary, weights, prompt_file, out_file, opt):
    """调用 gpt2_generate；返回 (ids, logits 或 None)。"""
    cmd = [str(binary), str(weights), str(prompt_file),
           "--max-new", str(opt.max_new), "--eos", str(opt.eos),
           "--seed", str(opt.seed), "--out-ids", str(out_file), "--quiet"]
    cmd += ["--greedy"] if opt.greedy else [
        "--sample", "--temperature", str(opt.temperature), "--top-k", str(opt.top_k)]

    logits_file = out_file.with_suffix(".logits.bin")
    if opt.logits:
        cmd += ["--dump-logits", str(logits_file)]

    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        log(f"  jasmine FAILED (exit {proc.returncode})")
        log(proc.stdout)
        log(proc.stderr)
        raise SystemExit(1)

    ids = [int(x) for x in out_file.read_text().split()]
    out_file.unlink(missing_ok=True)

    logits = None
    if opt.logits:
        r = WeightReader(logits_file)
        logits = r.get("logits")
        logits_file.unlink(missing_ok=True)
    return ids, logits


def first_mismatch(a, b):
    for i, (x, y) in enumerate(zip(a, b)):
        if x != y:
            return i
    return None


def main():
    ap = argparse.ArgumentParser(
        description="Compare jasmine GPT-2 inference against HuggingFace")
    ap.add_argument("--model", default="distilgpt2", help="HF model id")
    ap.add_argument("--text", default="The capital of France is", help="prompt text")
    ap.add_argument("--prompt-ids-file", default=None,
                    help="use existing prompt ids instead of --text")
    ap.add_argument("--weights", default=None,
                    help="jasmine weight file (default: build/<model>_weights.bin)")
    ap.add_argument("--binary", default="build/examples/gpt2_generate")
    ap.add_argument("--max-new", type=int, default=20,
                    help="tokens to generate; 0 = only compare logits")
    ap.add_argument("--eos", type=int, default=50256, help="eos id; <0 disables")
    ap.add_argument("--greedy", dest="greedy", action="store_true", default=True)
    ap.add_argument("--sample", dest="greedy", action="store_false")
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--top-k", type=int, default=40)
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--logits", action="store_true",
                    help="also compare full-sequence logits (same ids on both sides)")
    ap.add_argument("--tol", type=float, default=1e-3, help="logits tolerance")
    ap.add_argument("--no-export", action="store_true",
                    help="do not auto-export weights when missing")
    opt = ap.parse_args()

    weights = Path(opt.weights or f"build/{opt.model.replace('/', '_')}_weights.bin")
    binary = Path(opt.binary)

    log(f"=== jasmine vs HuggingFace: {opt.model} ===")

    if not binary.exists():
        log(f"ERROR: binary not found: {binary}")
        log("       build it with: cmake --build build -j --target gpt2_generate")
        return 1

    # ---- [1] 权重 ----
    if not weights.exists():
        if opt.no_export:
            log(f"ERROR: weights not found: {weights} (and --no-export was given)")
            return 1
        log(f"[1/5] weights missing -> exporting to {weights} (may take a while) ...")
        proc = subprocess.run(
            [sys.executable, "tools/export_gpt2.py", "--model", opt.model,
             "--out", str(weights)],
            capture_output=True, text=True)
        if proc.returncode != 0:
            log("[1/5] export FAILED")
            log(proc.stdout)
            log(proc.stderr)
            return 1
    log(f"[1/5] weights: {weights}")

    hf = HF(opt.model)

    # ---- [2] prompt ----
    if opt.prompt_ids_file:
        prompt_ids = [int(x) for x in Path(opt.prompt_ids_file).read_text().split()]
        log(f"[2/5] prompt: {len(prompt_ids)} tokens (from {opt.prompt_ids_file})")
    else:
        prompt_ids = hf.tokenizer(opt.text)["input_ids"]
        log(f"[2/5] prompt: {opt.text!r} -> {len(prompt_ids)} tokens")

    Path("build").mkdir(exist_ok=True)
    prompt_file = Path("build") / "verify_prompt_ids.txt"
    prompt_file.write_text("\n".join(str(i) for i in prompt_ids) + "\n")

    # ---- [3] jasmine ----
    js_ids, js_logits = run_jasmine(binary, weights, prompt_file,
                                   Path("build") / "verify_out_ids.txt", opt)
    prompt_file.unlink(missing_ok=True)
    log(f"[3/5] jasmine: {len(js_ids) - len(prompt_ids)} new tokens (total {len(js_ids)})")

    # ---- [4] HF ----
    hf_ids = hf.generate(prompt_ids, opt)
    log(f"[4/5] HF     : {len(hf_ids) - len(prompt_ids)} new tokens (total {len(hf_ids)})")

    # ---- [5] 比对 ----
    log("[5/5] compare")
    ok = True

    if opt.greedy:
        if len(js_ids) != len(hf_ids):
            log(f"  token ids : LENGTH MISMATCH (jasmine {len(js_ids)} vs HF {len(hf_ids)})")
            ok = False
        mi = first_mismatch(js_ids, hf_ids)
        if mi is None:
            log(f"  token ids : MATCH ({len(js_ids)}/{len(js_ids)})")
        else:
            ok = False
            lo, hi = max(0, mi - 3), min(len(js_ids), mi + 4)
            log(f"  token ids : MISMATCH at full-sequence index {mi}")
            log(f"    jasmine[{lo}:{hi}] = {js_ids[lo:hi]}")
            log(f"    HF     [{lo}:{hi}] = {hf_ids[lo:hi]}")
    else:
        log("  token ids : N/A (采样模式两侧 RNG 不同，逐 token 比对无意义；"
            "请看 --logits 的数值比对)")

    if opt.logits:
        if js_logits is None:
            log("  logits    : SKIPPED (jasmine dump failed)")
            ok = False
        else:
            # 用 jasmine 的序列喂 HF，两侧输入完全相同，与 RNG 无关
            hf_logits = hf.forward_logits(js_ids)
            if js_logits.shape != hf_logits.shape:
                ok = False
                log(f"  logits    : SHAPE MISMATCH {js_logits.shape} vs {hf_logits.shape}")
            else:
                diff = np.abs(js_logits - hf_logits)
                m = float(diff.max())
                passed = m < opt.tol
                ok &= passed
                v, t = np.unravel_index(int(diff.argmax()), diff.shape)
                log(f"  logits    : max_abs_diff = {m:.3e} over {js_logits.shape[0]}x"
                    f"{js_logits.shape[1]} ({'PASS' if passed else 'FAIL'}, tol {opt.tol})")
                if not passed:
                    log(f"    worst at (vocab={v}, pos={t}): "
                        f"jasmine={js_logits[v, t]:.6f} HF={hf_logits[v, t]:.6f}")
                n_bad = int((js_logits.argmax(axis=0) != hf_logits.argmax(axis=0)).sum())
                ok &= (n_bad == 0)
                log(f"  argmax    : {'MATCH all positions' if n_bad == 0 else f'{n_bad} differ'}")

    log()
    try:
        log(f"  text: {hf.tokenizer.decode(js_ids)!r}")
    except Exception:
        pass
    log(f"RESULT: {'PASS' if ok else 'FAIL'}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
