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


def main():
    args = parse_command_line_args()
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)
    special_token_ids = [
        tokenizer.bos_token_id,
        tokenizer.eos_token_id,
        tokenizer.unk_token_id,
    ]

    bytes_per_token = []
    for token_id in range(len(tokenizer)):
        if token_id in special_token_ids:
            # for special tokens, use -1 to mark as invalid
            bytes_per_token.append(-1)
        else:
            # Decode the token to its actual string representation
            decoded = tokenizer.decode([token_id])
            bytes_per_token.append(len(decoded.encode("utf-8")))
    np.save(args.output_path, np.array(bytes_per_token))
    print("Average bytes per token:", np.mean(bytes_per_token))
    print("Total number of tokens:", len(bytes_per_token))
    print(f"(Should be the same as the vocab size: {tokenizer.vocab_size})")


if __name__ == "__main__":
    main()
