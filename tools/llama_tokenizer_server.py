#!/usr/bin/env python3
"""常驻的 LLaMA 系 tokenizer 服务，供 examples/llama_chat 交互式调用。

与 tools/gpt2_tokenizer_server.py 的区别：
  * 多一个 `M` 操作：用 `tokenizer.apply_chat_template` 把**整段对话**渲染成 token id。
  * GPT-2 是 byte-level BPE，随便怎么拼都一样；LLaMA 用的是 SentencePiece，
    `enc(a) + enc(b) != enc(a + b)`（SPM 会给每段文本前面加一个 ▁ 空格）。
    实测把 `<|user|>\\n` 和用户文本分开编码再拼接，会得到 1724("What") 而不是
    5618(" What")，后面还会多/少一个 29871(▁) —— 也就是说**手写聊天模板一定会错**。
    所以这里坚持用 apply_chat_template 作为唯一事实来源。

协议（stdin -> stdout，单行；文本一律 base64 以免任何转义/编码问题）：
    E <base64(text)>                -> ok <id> <id> ...
    D <id> <id> ...                 -> ok <base64(text)>
    M <n> <b64 role> <b64 content> …-> ok <id> <id> ...    (apply_chat_template + generation prompt)
    V                               -> ok <vocab> <n_pos> <eos_id>
    任何失败                        -> err <base64(message)>

用法（通常由 llama_chat 自动拉起）：
    python tools/llama_tokenizer_server.py --model TinyLlama/TinyLlama-1.1B-Chat-v1.0
"""

import argparse
import base64
import sys


def b64e(s: str) -> str:
    return base64.b64encode(s.encode("utf-8")).decode("ascii")


def b64d(s: str) -> str:
    return base64.b64decode(s.encode("ascii")).decode("utf-8", "replace")


def main():
    ap = argparse.ArgumentParser(description="persistent LLaMA-family tokenizer server")
    ap.add_argument("--model", default="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
                    help="HF model id for the tokenizer")
    ap.add_argument("--revision", default=None)
    args = ap.parse_args()

    from transformers import AutoTokenizer

    try:
        tok = AutoTokenizer.from_pretrained(args.model, revision=args.revision)
    except Exception as e:                                  # noqa: BLE001
        print(f"err {b64e(f'failed to load tokenizer {args.model}: {e}')}", flush=True)
        return 1

    vocab = tok.vocab_size
    n_pos = getattr(tok, "model_max_length", 2048)
    if n_pos > 10**6:          # 有些 tokenizer 用 sentinel 表示「无限」
        n_pos = 2048
    eos = tok.eos_token_id if tok.eos_token_id is not None else -1
    if not getattr(tok, "chat_template", None):
        print(f"err {b64e(f'{args.model} has no chat_template; llama_chat needs one')}",
              flush=True)
        return 1

    # 就绪握手：vocab / 上下文上限 / eos
    print(f"ready {vocab} {n_pos} {eos}", flush=True)

    for line in sys.stdin:
        line = line.rstrip("\n")
        if not line:
            continue
        try:
            op, _, rest = line.partition(" ")
            if op == "E":
                ids = tok(b64d(rest), add_special_tokens=False)["input_ids"]
                print("ok " + " ".join(str(i) for i in ids), flush=True)
            elif op == "D":
                ids = [int(x) for x in rest.split()] if rest.strip() else []
                print("ok " + b64e(tok.decode(ids)), flush=True)
            elif op == "M":
                fields = rest.split()
                n = int(fields[0])
                toks = fields[1:]
                if len(toks) != 2 * n:
                    raise ValueError(f"M expects {2 * n} fields after the count, got {len(toks)}")
                messages = [
                    {"role": b64d(toks[2 * i]), "content": b64d(toks[2 * i + 1])}
                    for i in range(n)
                ]
                ids = tok.apply_chat_template(messages, add_generation_prompt=True)
                if not isinstance(ids, list):          # 某些版本返回 BatchEncoding
                    ids = ids["input_ids"]
                print("ok " + " ".join(str(i) for i in ids), flush=True)
            elif op == "V":
                print(f"ok {vocab} {n_pos} {eos}", flush=True)
            elif op == "Q":
                break
            else:
                print(f"err {b64e(f'unknown op {op!r}')}", flush=True)
        except Exception as e:                              # noqa: BLE001
            print(f"err {b64e(str(e))}", flush=True)

    return 0


if __name__ == "__main__":
    sys.exit(main())
