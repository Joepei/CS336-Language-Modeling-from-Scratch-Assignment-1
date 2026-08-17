"""
Encode training and validation datasets into numpy uint16 token ID arrays.

Usage (from cs336_basics/ directory):
    uv run encode_datasets.py

Output files written to ../data/:
    owt_train_tokens.bin      - OWT train encoded with OWT tokenizer
    owt_valid_tokens.bin      - OWT validation encoded with OWT tokenizer
    ts_train_tokens.bin       - TinyStories train encoded with TinyStories tokenizer
    ts_valid_tokens.bin       - TinyStories validation encoded with TinyStories tokenizer

Load back with:
    import numpy as np
    tokens = np.fromfile('path/to/file.bin', dtype=np.uint16)
"""

import numpy as np
import time
import os
import sys
sys.path.insert(0, os.path.dirname(__file__))

from cs336_basics import Tokenizer

SPECIAL_TOKENS = ["<|endoftext|>"]
DATA_DIR = os.path.join(os.path.dirname(__file__), '..', 'data')
VOCAB_DIR = os.path.dirname(__file__)

CHUNK_SIZE = 1_000_000  # tokens per write chunk


def encode_file_to_bin(tokenizer, input_path, output_path):
    """Stream-encode a text file and write token IDs as uint16 binary."""
    print(f"Encoding {os.path.basename(input_path)} -> {os.path.basename(output_path)}")
    t0 = time.time()
    total_tokens = 0

    def line_iter(path):
        with open(path, encoding='utf-8') as f:
            yield from f

    with open(output_path, 'wb') as out:
        buf = []
        for token_id in tokenizer.encode_iterable(line_iter(input_path)):
            buf.append(token_id)
            if len(buf) >= CHUNK_SIZE:
                np.array(buf, dtype=np.uint16).tofile(out)
                total_tokens += len(buf)
                buf = []
        if buf:
            np.array(buf, dtype=np.uint16).tofile(out)
            total_tokens += len(buf)

    elapsed = time.time() - t0
    size_mb = os.path.getsize(output_path) / 1e6
    print(f"  {total_tokens:,} tokens  |  {size_mb:.1f} MB  |  {elapsed:.1f}s  ({total_tokens/elapsed:,.0f} tok/s)\n")


def main():
    owt_tokenizer = Tokenizer.from_files(
        os.path.join(VOCAB_DIR, 'owt_vocab.json'),
        os.path.join(VOCAB_DIR, 'owt_merges.pkl'),
        SPECIAL_TOKENS,
    )
    ts_tokenizer = Tokenizer.from_files(
        os.path.join(VOCAB_DIR, 'tiny_stories_vocab.json'),
        os.path.join(VOCAB_DIR, 'tiny_stories_merges.pkl'),
        SPECIAL_TOKENS,
    )

    tasks = [
        (owt_tokenizer, 'owt_train.txt',            'owt_train_tokens.bin'),
        (owt_tokenizer, 'owt_valid.txt',            'owt_valid_tokens.bin'),
        (ts_tokenizer,  'tinystories_train.txt',    'ts_train_tokens.bin'),
        (ts_tokenizer,  'tinystories_validation.txt', 'ts_valid_tokens.bin'),
    ]

    for tokenizer, in_file, out_file in tasks:
        in_path  = os.path.join(DATA_DIR, in_file)
        out_path = os.path.join(DATA_DIR, out_file)
        if not os.path.exists(in_path):
            print(f"Skipping {in_file} (not found)")
            continue
        encode_file_to_bin(tokenizer, in_path, out_path)

    print("Done.")


if __name__ == '__main__':
    main()
