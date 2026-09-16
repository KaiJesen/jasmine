#!/usr/bin/env python3
"""把 HuggingFace 的 LLaMA 系权重（TinyLlama / LLaMA / Mistral / Qwen2 …）导出成 jasmine 格式。

用法：
    python tools/export_llama.py --model TinyLlama/TinyLlama-1.1B-Chat-v1.0 \\
        --out build/tinyllama_weights.bin --golden-prompt "The capital of France is"

导出文件格式与 GPT-2 完全一致（见 jas_weight_io.hpp）：
    JASMINE_WEIGHTS_V1\\n
    f32\\n
    <count>\\n
    <name> <rows> <cols> <byte_offset>\\n   x count
    \\n
    <二进制 float32，行优先>

布局换算（与 export_gpt2.py 的差别是**少了很多**，这正是 LLaMA 更好对齐的原因）：
  * 线性层**不需要转置**。GPT-2 用 nn.Conv1D，权重是 [in, out]，所以必须转置；
    LLaMA 用 nn.Linear，权重本来就是 [out, in]，与 jasmine 的 weight_net_t（y = W x）一致。
  * RMSNorm 的 1-D [d] -> [d, 1] 列向量（没有 beta）。
  * embed_tokens [vocab, d_model] -> [d_model, vocab]（embedding 查表布局）。
  * **没有 bias**：attention_bias / mlp_bias 均为 false，导出时断言其确实不存在。
  * K/V 投影本来就是 [n_kv_heads*d_head, d_model]，无需拆分（GQA 天然分离，不像 GPT-2 的 fused c_attn）。
  * lm_head：TinyLlama 的 tie_word_embeddings=false，因此单独导出；绑定的变体则跳过。

⚠️ RoPE 基频：jas_RoPE_t.hpp 目前把 10000 硬编码。TinyLlama 的 rope_theta 正是 10000.0，
所以可用；LLaMA-3.x 等用 500000 的模型会被这里拒绝（宁可报错，也不要出一个「看着像但其实错了」
的 logits —— 这类错误极难发现）。
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

from jasmine_weights import WeightWriter

# jas_RoPE_t.hpp 里硬编码的基频
SUPPORTED_ROPE_THETA = 10000.0


def col(vec):
    """1-D [d] -> 2-D [d, 1]（jasmine 的 gamma / bias 布局）。"""
    return np.asarray(vec, dtype=np.float32).reshape(-1, 1)


def check_model_supported(cfg):
    """在动手导出前先把关：任何一条不满足都会让 C++ 侧的 logits 静默地对不上。"""
    problems = []

    rope_theta = float(getattr(cfg, "rope_theta", 10000.0) or 10000.0)
    if abs(rope_theta - SUPPORTED_ROPE_THETA) > 1e-6:
        problems.append(
            f"rope_theta={rope_theta} but jas_RoPE_t.hpp hard-codes {SUPPORTED_ROPE_THETA}; "
            "extend mat_RoPE_t before exporting this model"
        )

    if getattr(cfg, "attention_bias", False):
        problems.append("attention_bias=true, but jas_llama_t.hpp zeroes all attention biases")
    if getattr(cfg, "mlp_bias", False):
        problems.append("mlp_bias=true, but jas_llama_t.hpp zeroes all mlp biases")

    # 注意：新版 transformers 会把 rope_scaling 规范化成 {'rope_type': 'default', ...}，
    # 这**不是**真的做了缩放（TinyLlama 就是这种），只有非 default 的才是。
    rope_scaling = getattr(cfg, "rope_scaling", None)
    if rope_scaling:
        rope_type = str(rope_scaling.get("rope_type", rope_scaling.get("type", "default")))
        if rope_type != "default":
            problems.append(f"rope_scaling type={rope_type!r} is not implemented in jasmine")

    act = getattr(cfg, "hidden_act", None)
    if act not in (None, "silu"):
        problems.append(f"hidden_act={act!r}, but llama_ffn_branch_t uses silu (SwiGLU)")

    if problems:
        raise SystemExit("[export] refusing to export:\n  - " + "\n  - ".join(problems))


def capture_stages(model, ids):
    """用 forward hook 抓取 [embed 输出, block0..blockN-1 输出]（全部为 pre-final-norm）。

    为什么不直接用 `output_hidden_states=True` 的返回值：transformers 各版本会
    在末尾**追加一个 final-norm 之后的张量**，于是 hidden_states[-1] 并不是最后一层的
    真实输出（GPT-2 踩过这个坑，LLaMA 在 v5 上同样如此 —— 这里实测过：
    hs[-1] 的每行 RMS 约等于 1，而残差流的 RMS 是 0.02 量级，且与最后一层的 hook 输出
    相差 3.09）。拿错层次的张量当黄金值会让「逐层对齐」这一层保护完全失效。
    hook 与版本无关，因此这里不再依赖 hidden_states 的语义。
    """
    embed_cap = []
    layer_caps = [[] for _ in range(len(model.model.layers))]
    handles = []

    def make_hook(sink):
        def hook(module, inputs, output):
            h = output[0] if isinstance(output, (tuple, list)) else output
            sink.append(h.detach())
        return hook

    handles.append(model.model.embed_tokens.register_forward_hook(make_hook(embed_cap)))
    for i, layer in enumerate(model.model.layers):
        handles.append(layer.register_forward_hook(make_hook(layer_caps[i])))

    with torch.no_grad():
        out = model(ids, output_hidden_states=True)

    for h in handles:
        h.remove()

    stages = [embed_cap[0]] + [c[0] for c in layer_caps]
    return stages, out


def dump_golden(w, model, tokenizer, tag, ids):
    """把 HF 在给定 token 序列上的 logits 与逐层 hidden 写进权重文件，供 C++ 单测比对。"""
    stages, out = capture_stages(model, ids)
    logits = out.logits.float().cpu().numpy()[0].T          # [vocab, T]

    w.add(f"golden.{tag}.input_ids", ids.numpy().astype(np.float32))
    w.add(f"golden.{tag}.logits", logits)

    hs = [h.float().cpu().numpy()[0] for h in stages]

    # 自检：若本版本确实在 hidden_states 末尾追加了 final-norm 之后的值，
    # 则 norm(我们的末层输出) 应当等于它；否则说明两端语义已经一致。
    try:
        ref_last = out.hidden_states[-1].float().cpu().numpy()[0]
        with torch.no_grad():
            check = model.model.norm(stages[-1]).float().cpu().numpy()[0]
        gap = float(np.abs(check - ref_last).max())
        note = ("appends a post-final-norm tensor" if gap < 5e-3
                else "does NOT append a post-final-norm tensor")
        print(f"[export] hidden_states[-1] check: gap={gap:.3e} -> transformers {note}; "
              f"using hook-captured pre-norm stages either way")
    except Exception as e:                                  # noqa: BLE001
        print(f"[export] hidden_states self-check skipped: {e}")

    for i, h in enumerate(hs):
        w.add(f"golden.{tag}.hidden.{i}", h.T)      # [d_model, T]
    print(f"[export] golden[{tag}]: {ids.shape[1]} tokens, logits {logits.shape}, "
          f"{len(hs)} hidden stages (embed + {len(hs) - 1} blocks)")
    return ids.shape[1]


def as_batch_ids(ids):
    """把 tokenizer 的返回值统一成 [1, T] 的 long 张量。

    transformers v5 里 `apply_chat_template(..., return_tensors='pt')` 返回的是
    BatchEncoding（含 attention_mask），直接取 .shape 会 AttributeError；
    而不同版本又可能直接返回张量或 list，所以这里统一兜一层。
    """
    if hasattr(ids, "input_ids"):
        ids = ids.input_ids
    if hasattr(ids, "shape"):                 # 已经是张量
        return ids if ids.dim() == 2 else ids.unsqueeze(0)
    return torch.tensor([list(ids)], dtype=torch.long)


def export(model_id, out_path, golden_prompt=None, golden_chat=None, revision=None):
    from transformers import AutoModelForCausalLM, AutoTokenizer

    print(f"[export] loading {model_id} (float32 for a precise reference) ...", flush=True)
    # float32 而非原生 bf16：黄金值要尽量精确，bf16 只有 8 位尾数，
    # 拿它当基准会把参考实现自身的舍入误差算到我们头上。
    model = AutoModelForCausalLM.from_pretrained(model_id, dtype=torch.float32, revision=revision)
    model.eval()
    cfg = model.config
    check_model_supported(cfg)

    d_model = int(cfg.hidden_size)
    n_heads = int(cfg.num_attention_heads)
    n_kv_heads = int(getattr(cfg, "num_key_value_heads", n_heads) or n_heads)
    n_layers = int(cfg.num_hidden_layers)
    vocab = int(cfg.vocab_size)
    n_pos = int(cfg.max_position_embeddings)
    d_ff = int(cfg.intermediate_size)
    rms_eps = float(getattr(cfg, "rms_norm_eps", 1e-5))
    rope_theta = float(getattr(cfg, "rope_theta", 10000.0) or 10000.0)
    tied = bool(getattr(cfg, "tie_word_embeddings", False))

    if d_model % n_heads != 0:
        raise SystemExit(f"[export] hidden_size {d_model} not divisible by heads {n_heads}")
    if n_heads % n_kv_heads != 0:
        raise SystemExit(f"[export] heads {n_heads} not divisible by kv heads {n_kv_heads}")
    d_head = d_model // n_heads

    print(f"[export] layers={n_layers} heads={n_heads} kv_heads={n_kv_heads} "
          f"d_model={d_model} d_head={d_head} d_ff={d_ff} vocab={vocab} n_pos={n_pos} "
          f"rms_eps={rms_eps} rope_theta={rope_theta} tied={tied}", flush=True)

    sd = {k: v.detach().cpu().numpy() for k, v in model.state_dict().items()}

    # bias 必须一个都不存在（check_model_supported 已看过 config，这里再核对实际张量）
    bias_keys = [k for k in sd if k.endswith(".bias")]
    if bias_keys:
        raise SystemExit("[export] unexpected bias tensors: " + ", ".join(sorted(bias_keys)[:8]))

    w = WeightWriter()
    w.add_scalar("cfg.n_layers", n_layers)
    w.add_scalar("cfg.n_heads", n_heads)
    w.add_scalar("cfg.n_kv_heads", n_kv_heads)
    w.add_scalar("cfg.d_model", d_model)
    w.add_scalar("cfg.d_ff", d_ff)
    w.add_scalar("cfg.vocab", vocab)
    w.add_scalar("cfg.n_pos", n_pos)
    w.add_scalar("cfg.rms_eps", rms_eps)
    w.add_scalar("cfg.rope_theta", rope_theta)
    w.add_scalar("cfg.tie_word_embeddings", 1.0 if tied else 0.0)

    # 词嵌入：[vocab, d_model] -> [d_model, vocab]（唯一的转置）
    wte = sd["model.embed_tokens.weight"]
    assert wte.shape == (vocab, d_model), wte.shape
    w.add("wte.weight", wte.T)

    def linear(name, expect_shape):
        """LLaMA 的 nn.Linear 权重本就是 [out, in]，与 jasmine 一致，直接透传。"""
        arr = np.asarray(sd[name], dtype=np.float32)
        assert arr.shape == expect_shape, f"{name}: {arr.shape} != {expect_shape}"
        return arr

    for i in range(n_layers):
        p = f"model.layers.{i}."
        prefix = f"h.{i}."

        w.add(prefix + "ln_1.weight", col(sd[p + "input_layernorm.weight"]))
        w.add(prefix + "ln_2.weight", col(sd[p + "post_attention_layernorm.weight"]))

        w.add(prefix + "attn.q.weight", linear(p + "self_attn.q_proj.weight", (d_model, d_model)))
        # GQA：K/V 是 [n_kv_heads*d_head, d_model]，本来就只有这么宽，不需要切
        w.add(prefix + "attn.k.weight", linear(p + "self_attn.k_proj.weight", (n_kv_heads * d_head, d_model)))
        w.add(prefix + "attn.v.weight", linear(p + "self_attn.v_proj.weight", (n_kv_heads * d_head, d_model)))
        w.add(prefix + "attn.out.weight", linear(p + "self_attn.o_proj.weight", (d_model, d_model)))

        # SwiGLU 三个矩阵：gate/up 是 d_model->d_ff，down 是 d_ff->d_model
        w.add(prefix + "mlp.gate.weight", linear(p + "mlp.gate_proj.weight", (d_ff, d_model)))
        w.add(prefix + "mlp.up.weight", linear(p + "mlp.up_proj.weight", (d_ff, d_model)))
        w.add(prefix + "mlp.down.weight", linear(p + "mlp.down_proj.weight", (d_model, d_ff)))

    w.add("ln_f.weight", col(sd["model.norm.weight"]))
    if not tied:
        if "lm_head.weight" not in sd:
            raise SystemExit("[export] tie_word_embeddings=false but lm_head.weight is missing")
        w.add("lm_head.weight", linear("lm_head.weight", (vocab, d_model)))

    tokenizer = AutoTokenizer.from_pretrained(model_id, revision=revision)

    if golden_prompt is not None:
        ids = as_batch_ids(tokenizer(golden_prompt, return_tensors="pt")["input_ids"])
        if ids.shape[1] > n_pos:
            raise SystemExit(f"golden prompt too long: {ids.shape[1]} > n_pos {n_pos}")
        dump_golden(w, model, tokenizer, "raw", ids)
        print(f"[export] golden raw prompt={golden_prompt!r}")

    if golden_chat is not None:
        msgs = [{"role": "user", "content": golden_chat}]
        ids = as_batch_ids(tokenizer.apply_chat_template(
            msgs, add_generation_prompt=True, return_tensors="pt"))
        if ids.shape[1] > n_pos:
            raise SystemExit(f"golden chat too long: {ids.shape[1]} > n_pos {n_pos}")
        dump_golden(w, model, tokenizer, "chat", ids)
        print(f"[export] golden chat (apply_chat_template) user={golden_chat!r}")

    path = w.write(out_path)
    size_mb = path.stat().st_size / (1024 * 1024)
    print(f"[export] wrote {len(w.entries)} tensors, {size_mb:.1f} MiB -> {path}")

    meta = {
        "model": model_id,
        "config": {
            "n_layers": n_layers, "n_heads": n_heads, "n_kv_heads": n_kv_heads,
            "d_model": d_model, "d_head": d_head, "d_ff": d_ff,
            "vocab": vocab, "n_pos": n_pos, "rms_eps": rms_eps,
            "rope_theta": rope_theta, "tie_word_embeddings": tied,
        },
        "tensors": [n for n, _, _, _ in w.entries],
    }
    if golden_prompt is not None:
        meta["golden_prompt"] = golden_prompt
    if golden_chat is not None:
        meta["golden_chat"] = golden_chat
    meta_path = path.with_suffix(path.suffix + ".json")
    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(f"[export] manifest -> {meta_path}")


def main():
    ap = argparse.ArgumentParser(description="Export HuggingFace LLaMA-family weights for jasmine")
    ap.add_argument("--model", default="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
                    help="HF model id (must be a LLaMA-architecture chat/instruct model)")
    ap.add_argument("--out", default="build/tinyllama_weights.bin", help="output weight file")
    ap.add_argument("--revision", default=None, help="optional HF revision")
    ap.add_argument("--golden-prompt", default=None,
                    help="raw text prompt: also dump HF logits/hiddens for it (alignment test)")
    ap.add_argument("--golden-chat", default=None,
                    help="user message: dump HF logits for apply_chat_template(...) — "
                         "the exact path examples/llama_chat takes")
    args = ap.parse_args()
    export(args.model, args.out, golden_prompt=args.golden_prompt,
           golden_chat=args.golden_chat, revision=args.revision)


if __name__ == "__main__":
    sys.exit(main())
