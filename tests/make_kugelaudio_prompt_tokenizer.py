#!/usr/bin/env -S uv run --script
# /// script
# dependencies = ["gguf"]
# ///
from pathlib import Path
import argparse
import gguf


def make_byte_to_unicode():
    out = [None] * 256
    used = [False] * 256
    def add(b):
        out[b] = b
        used[b] = True
    for b in range(33, 127):
        add(b)
    for b in range(161, 173):
        add(b)
    for b in range(174, 256):
        add(b)
    n = 0
    for b in range(256):
        if not used[b]:
            out[b] = 256 + n
            n += 1
    return out


def cp_to_utf8(cp: int) -> str:
    return chr(cp)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    byte_to_unicode = make_byte_to_unicode()
    tokens = [cp_to_utf8(cp) for cp in byte_to_unicode]
    token_types = [1] * len(tokens)

    w = gguf.GGUFWriter(args.out, arch="vibevoice")
    w.add_array("tokenizer.tokens", tokens)
    w.add_array("tokenizer.token_type", token_types)
    w.add_array("tokenizer.merges", [])
    w.add_array("tokenizer.special_tokens_ids", [151652, 151653, 151654, 151655])
    w.add_array("tokenizer.special_tokens_text", [
        "<|vision_start|>",
        "<|vision_end|>",
        "<|vision_pad|>",
        "<|image_pad|>",
    ])
    w.add_uint32("tokenizer.bos_id", 0)
    w.add_uint32("tokenizer.eos_id", 0)
    w.add_uint32("tokenizer.pad_id", 151655)
    w.write_header_to_file()
    w.write_kv_data_to_file()
    w.write_tensors_to_file()
    w.close()
    print(Path(args.out))


if __name__ == "__main__":
    main()
