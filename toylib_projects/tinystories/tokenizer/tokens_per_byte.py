"""Dumps the tokens per byte statistics for the given tokenizer."""

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

    bytes_per_token = []
    for token_id in range(len(tokenizer)):
        # Decode the token to its actual string representation
        decoded = tokenizer.decode([token_id])
        bytes_per_token.append(len(decoded.encode("utf-8")))
    np.save(args.output_path, np.array(bytes_per_token))
    print("Average bytes per token:", np.mean(bytes_per_token))
    print("Total number of tokens:", len(bytes_per_token))
    print(f"(Should be the same as the vocab size: {tokenizer.vocab_size})")


if __name__ == "__main__":
    main()
