"""Dumps the tokens per byte statistics for the given tokenizer.

Sample command:
python toylib_projects/tinystories/tokenizer/bytes_per_token.py \
  --tokenizer gpt2 \
  --output-path toylib_projects/tinystories/data/bpt_gpt2.npy
"""

import argparse
from transformers import AutoTokenizer
import numpy as np


def parse_command_line_args():
    parser = argparse.ArgumentParser(
        description="Run tokens per byte analysis for a given tokenizer"
    )

    parser.add_argument(
        "--tokenizer",
        type=str,
        required=True,
        default="gpt2",
        help="HF tokenizer name (e.g., 'gpt2')",
    )
    parser.add_argument(
        "--output-path",
        type=str,
        required=True,
        help="Path to save bytes per token (.npy file)",
    )
    return parser.parse_args()


def compute_bytes_per_token(tokenizer_name: str) -> np.ndarray:
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    special_token_ids = [
        tokenizer.bos_token_id,
        tokenizer.eos_token_id,
        tokenizer.unk_token_id,
    ]

    bpt = []
    for token_id in range(len(tokenizer)):
        if token_id in special_token_ids:
            # for special tokens, use -1 to mark as invalid
            bpt.append(-1)
        else:
            # Decode the token to its actual string representation
            decoded = tokenizer.decode([token_id])
            bpt.append(len(decoded.encode("utf-8")))
    return np.array(bpt)


def main():
    args = parse_command_line_args()
    bpt = compute_bytes_per_token(args.tokenizer)
    np.save(args.output_path, bpt)
    print("Average bytes per token:", np.mean(bpt))
    print("Total number of tokens:", len(bpt))
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)
    print(f"(Should be the same as the vocab size: {tokenizer.vocab_size})")


if __name__ == "__main__":
    main()
