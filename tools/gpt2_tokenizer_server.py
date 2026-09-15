#!/usr/bin/env python3
"""常驻的 GPT-2 tokenizer 服务，供 examples/gpt2_chat 交互式调用。

为什么需要它：`transformers` 的 tokenizer 每次冷启动约 2 秒，而交互式对话每轮至少要
encode/decode 各一次，反复起进程会让响应慢到不可用。这里把 tokenizer 加载一次，
之后用行协议常驻服务。

协议（stdin -> stdout，均为单行；文本一律 base64 以保证任意字节安全）：
    E <base64(text)>   -> ok <id> <id> ...
    D <id> <id> ...    -> ok <base64(text)>
    V                  -> ok <vocab_size> <n_positions>
    任何失败           -> err <base64(message)>

用法（通常由 gpt2_chat 自动拉起，不必手动运行）：
    python tools/gpt2_tokenizer_server.py --model distilgpt2
"""

import argparse
import base64
import sys


def b64e(s: str) -> str:
    return base64.b64encode(s.encode("utf-8")).decode("ascii")


def b64d(s: str) -> str:
    return base64.b64decode(s.encode("ascii")).decode("utf-8", "replace")


def main():
    ap = argparse.ArgumentParser(description="persistent GPT-2 tokenizer server")
    ap.add_argument("--model", default="distilgpt2", help="HF model id for the tokenizer")
    ap.add_argument("--revision", default=None)
    args = ap.parse_args()

    from transformers import AutoTokenizer

    try:
        tok = AutoTokenizer.from_pretrained(args.model, revision=args.revision)
    except Exception as e:                                  # noqa: BLE001
        print(f"err {b64e(f'failed to load tokenizer {args.model}: {e}')}", flush=True)
        return 1

    vocab = tok.vocab_size
    n_pos = getattr(tok, "model_max_length", 1024)
    if n_pos > 10**6:          # 有些 tokenizer 用 sentinel 表示「无限」
        n_pos = 1024

    # 就绪握手：让 C++ 侧确认 tokenizer 可用，并拿到词表大小
    print(f"ready {vocab} {n_pos}", flush=True)

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
            elif op == "V":
                print(f"ok {vocab} {n_pos}", flush=True)
            elif op == "Q":
                break
            else:
                print(f"err {b64e(f'unknown op {op!r}')}", flush=True)
        except Exception as e:                              # noqa: BLE001
            print(f"err {b64e(str(e))}", flush=True)

    return 0


if __name__ == "__main__":
    sys.exit(main())
