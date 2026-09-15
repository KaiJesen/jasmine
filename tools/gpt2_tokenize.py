#!/usr/bin/env python3
"""GPT-2 文本 <-> token id 互转（词表由 Python 侧处理，C++ 不依赖 tokenizers）。

用法：
    # 文本 -> ids（写到文件，每行一个 id；缺省打印到 stdout）
    python tools/gpt2_tokenize.py encode --model distilgpt2 \
        --text "Hello, my dog is cute" --out build/prompt_ids.txt

    # ids -> 文本
    python tools/gpt2_tokenize.py decode --model distilgpt2 --ids-file build/out_ids.txt
    python tools/gpt2_tokenize.py decode --model distilgpt2 --ids 15496,11,50256
"""

import argparse
import sys


def load_tokenizer(model_id, revision=None):
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(model_id, revision=revision)
    return tok


def read_ids_file(path):
    ids = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            ids.extend(int(x) for x in line.replace(",", " ").split())
    return ids


def write_ids_file(path, ids):
    with open(path, "w", encoding="utf-8") as f:
        for i in ids:
            f.write(f"{i}\n")


def main():
    ap = argparse.ArgumentParser(description="GPT-2 tokenize / detokenize helper")
    sub = ap.add_subparsers(dest="cmd", required=True)

    def common(p):
        p.add_argument("--model", default="distilgpt2")
        p.add_argument("--revision", default=None)

    p_enc = sub.add_parser("encode", help="text -> token ids")
    common(p_enc)
    p_enc.add_argument("--text", required=True)
    p_enc.add_argument("--out", default=None, help="write one id per line; default stdout")

    p_dec = sub.add_parser("decode", help="token ids -> text")
    common(p_dec)
    p_dec.add_argument("--ids", default=None, help="comma/space separated ids")
    p_dec.add_argument("--ids-file", default=None, help="file with one id per line")

    args = ap.parse_args()

    if args.cmd == "encode":
        tok = load_tokenizer(args.model, args.revision)
        ids = tok(args.text)["input_ids"]
        if args.out:
            write_ids_file(args.out, ids)
            print(f"[tokenize] {len(ids)} tokens -> {args.out}", file=sys.stderr)
        else:
            print(" ".join(str(i) for i in ids))
        return 0

    if args.cmd == "decode":
        if args.ids_file:
            ids = read_ids_file(args.ids_file)
        elif args.ids is not None:
            ids = [int(x) for x in args.ids.replace(",", " ").split()]
        else:
            raise SystemExit("decode requires --ids or --ids-file")
        tok = load_tokenizer(args.model, args.revision)
        print(tok.decode(ids))
        return 0

    return 1


if __name__ == "__main__":
    sys.exit(main())
