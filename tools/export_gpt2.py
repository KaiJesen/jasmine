#!/usr/bin/env python3
"""把 HuggingFace 的 GPT-2 权重导出成 jasmine 的单文件权重格式。

用法：
    python tools/export_gpt2.py --model distilgpt2 --out build/gpt2_weights.bin
    python tools/export_gpt2.py --model gpt2 --out build/gpt2_weights.bin \
        --golden-prompt "Hello, my dog is cute"

导出文件格式（见 jas_weight_io.hpp）：
    JASMINE_WEIGHTS_V1\\n
    f32\\n
    <count>\\n
    <name> <rows> <cols> <byte_offset>\\n   x count
    \\n
    <二进制 float32，行优先>

所有 tensor 都转成 **jasmine 原生布局**：
  * Conv1D 权重 [in, out] -> 转置为 [out, in]（jasmine 的 weight_net_t 是 y = W x，W 为 [out, in]）
  * fused c_attn [n_embd, 3*n_embd] -> 转置后按行切成 q / k / v
  * LayerNorm 的 1-D [d] -> [d, 1] 列向量
  * wte [vocab, d_model] -> [d_model, vocab]（embedding 查表布局）
  * lm_head 与 wte 绑定，因此不单独导出 lm_head.weight；lm_head 无 bias
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

from jasmine_weights import WeightWriter


def col(vec):
    """1-D [d] -> 2-D [d, 1]（jasmine 的 gamma/beta/bias 布局）。"""
    return np.asarray(vec, dtype=np.float32).reshape(-1, 1)


def linear_from_conv1d(weight, bias):
    """HF GPT-2 的 Conv1D: weight [in, out] -> jasmine [out, in]；bias [out] -> [out, 1]。"""
    return np.asarray(weight, dtype=np.float32).T, col(bias)


def export(model_id, out_path, golden_prompt=None, revision=None):
    from transformers import AutoModelForCausalLM, AutoTokenizer

    print(f"[export] loading {model_id} ...", flush=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_id, dtype=torch.float32, revision=revision
    )
    model.eval()
    cfg = model.config

    d_model = int(cfg.n_embd)
    n_heads = int(cfg.n_head)
    n_layers = int(cfg.n_layer)
    vocab = int(cfg.vocab_size)
    n_pos = int(cfg.n_positions)
    d_ff = int(getattr(cfg, "n_inner", None) or 4 * d_model)

    sd = {k: v.detach().cpu().numpy() for k, v in model.state_dict().items()}

    w = WeightWriter()
    # 结构参数：让 C++ 侧无需解析 config.json 也能构造模型
    w.add_scalar("cfg.n_layers", n_layers)
    w.add_scalar("cfg.n_heads", n_heads)
    w.add_scalar("cfg.d_model", d_model)
    w.add_scalar("cfg.d_ff", d_ff)
    w.add_scalar("cfg.vocab", vocab)
    w.add_scalar("cfg.n_pos", n_pos)

    # 词嵌入 / 位置嵌入
    wte = sd["transformer.wte.weight"]                    # [vocab, d_model]
    assert wte.shape == (vocab, d_model), wte.shape
    w.add("wte.weight", wte.T)                            # -> [d_model, vocab]

    if "transformer.wpe.weight" in sd:
        wpe = sd["transformer.wpe.weight"]                # [n_pos, d_model]
        assert wpe.shape == (n_pos, d_model), wpe.shape
        w.add("wpe.weight", wpe.T)                        # -> [d_model, n_pos]

    for i in range(n_layers):
        p = f"transformer.h.{i}."
        prefix = f"h.{i}."

        w.add(prefix + "ln_1.weight", col(sd[p + "ln_1.weight"]))
        w.add(prefix + "ln_1.bias", col(sd[p + "ln_1.bias"]))

        if p + "attn.c_attn.weight" in sd:
            # fused QKV：Conv1D [d_model, 3*d_model] -> [3*d_model, d_model] -> 切 q/k/v
            qkv_w = np.asarray(sd[p + "attn.c_attn.weight"], dtype=np.float32)
            qkv_b = np.asarray(sd[p + "attn.c_attn.bias"], dtype=np.float32)
            assert qkv_w.shape == (d_model, 3 * d_model), qkv_w.shape
            assert qkv_b.shape == (3 * d_model,), qkv_b.shape
            qkv_w = qkv_w.T                                   # [3*d_model, d_model]
            for idx, tag in enumerate(("q", "k", "v")):
                s = slice(idx * d_model, (idx + 1) * d_model)
                w.add(prefix + f"attn.{tag}.weight", qkv_w[s, :])
                w.add(prefix + f"attn.{tag}.bias", col(qkv_b[s]))
        else:
            # 拆分式投影：Conv1D [d_model, d_model]
            for tag in ("q", "k", "v"):
                kw = sd[p + f"attn.{tag}_proj.weight"]
                kb = sd[p + f"attn.{tag}_proj.bias"]
                tw, tb = linear_from_conv1d(kw, kb)
                assert tw.shape == (d_model, d_model), tw.shape
                w.add(prefix + f"attn.{tag}.weight", tw)
                w.add(prefix + f"attn.{tag}.bias", tb)

        ow, ob = linear_from_conv1d(sd[p + "attn.c_proj.weight"], sd[p + "attn.c_proj.bias"])
        assert ow.shape == (d_model, d_model), ow.shape
        w.add(prefix + "attn.out.weight", ow)
        w.add(prefix + "attn.out.bias", ob)

        w.add(prefix + "ln_2.weight", col(sd[p + "ln_2.weight"]))
        w.add(prefix + "ln_2.bias", col(sd[p + "ln_2.bias"]))

        fw, fb = linear_from_conv1d(sd[p + "mlp.c_fc.weight"], sd[p + "mlp.c_fc.bias"])
        assert fw.shape == (d_ff, d_model), fw.shape
        w.add(prefix + "mlp.fc.weight", fw)
        w.add(prefix + "mlp.fc.bias", fb)

        pw, pb = linear_from_conv1d(sd[p + "mlp.c_proj.weight"], sd[p + "mlp.c_proj.bias"])
        assert pw.shape == (d_model, d_ff), pw.shape
        w.add(prefix + "mlp.proj.weight", pw)
        w.add(prefix + "mlp.proj.bias", pb)

    w.add("ln_f.weight", col(sd["transformer.ln_f.weight"]))
    w.add("ln_f.bias", col(sd["transformer.ln_f.bias"]))

    # 黄金值：固定 prompt 下 HF 的 logits，供 C++ 单测逐点比对
    if golden_prompt is not None:
        tokenizer = AutoTokenizer.from_pretrained(model_id, revision=revision)
        ids = tokenizer(golden_prompt, return_tensors="pt")["input_ids"]
        if ids.shape[1] > n_pos:
            raise SystemExit(f"golden prompt too long: {ids.shape[1]} > n_pos {n_pos}")
        with torch.no_grad():
            out = model(ids, output_hidden_states=True)
            logits = out.logits.float().cpu().numpy()            # [1, T, vocab]
        logits = logits[0].T                                      # [vocab, T]
        w.add("golden.input_ids", ids.numpy().astype(np.float32))
        w.add("golden.logits", logits)

        # 逐层隐藏状态：[嵌入输出, block0 输出, ...]（ln_f 之前），对应 forward_stages
        #
        # 注意 transformers >= 5 的一个坑：hidden_states[-1] 存的是**经过 ln_f 之后**的值，
        # 而 hidden_states[0..n_layers-1] 才是各 block 的真实 pre-ln_f 输出。
        # 因此最后一个 block 的输出必须手动跑一遍，否则拿到的黄金值无法与 jasmine 的
        # forward_stages 对齐（LayerNorm 会把差异掩盖成「看起来数值范围相似」）。
        stages = [h.float().cpu().numpy()[0] for h in out.hidden_states[:n_layers]]
        with torch.no_grad():
            last_in = torch.tensor(stages[-1]).unsqueeze(0)
            last = model.transformer.h[-1](last_in)
            last = last[0] if isinstance(last, (tuple, list)) else last
            last = last.float().cpu().numpy()[0]
        # 自检：假设成立则 ln_f(最后一个 block 输出) 应等于 HF 的 hidden_states[-1]
        with torch.no_grad():
            check = model.transformer.ln_f(torch.tensor(last).unsqueeze(0))[0].numpy()
        ref_last = out.hidden_states[-1].float().cpu().numpy()[0]
        if np.abs(check - ref_last).max() > 1e-3:
            print("[export] WARNING: ln_f(last block) != hidden_states[-1]; "
                  "transformers version may have changed hidden_states semantics")
        stages.append(last)

        for i, h in enumerate(stages):
            w.add(f"golden.hidden.{i}", h.T)   # [d_model, T]
        print(f"[export] golden prompt -> {ids.shape[1]} tokens, "
              f"logits {logits.shape}, {len(stages)} hidden stages, prompt={golden_prompt!r}")

    path = w.write(out_path)
    size_mb = path.stat().st_size / (1024 * 1024)
    print(f"[export] wrote {len(w.entries)} tensors, {size_mb:.1f} MiB -> {path}")

    meta = {
        "model": model_id,
        "config": {
            "n_layers": n_layers, "n_heads": n_heads, "d_model": d_model,
            "d_ff": d_ff, "vocab": vocab, "n_pos": n_pos,
        },
        "tensors": [n for n, _, _, _ in w.entries],
    }
    if golden_prompt is not None:
        meta["golden_prompt"] = golden_prompt
    meta_path = path.with_suffix(path.suffix + ".json")
    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(f"[export] manifest -> {meta_path}")


def main():
    ap = argparse.ArgumentParser(description="Export HuggingFace GPT-2 weights for jasmine")
    ap.add_argument("--model", default="distilgpt2", help="HF model id (distilgpt2 / gpt2 / ...)")
    ap.add_argument("--out", default="build/gpt2_weights.bin", help="output weight file")
    ap.add_argument("--revision", default=None, help="optional HF revision")
    ap.add_argument("--golden-prompt", default=None,
                    help="if set, also dump HF logits for this prompt (for the alignment test)")
    args = ap.parse_args()
    export(args.model, args.out, golden_prompt=args.golden_prompt, revision=args.revision)


if __name__ == "__main__":
    sys.exit(main())
