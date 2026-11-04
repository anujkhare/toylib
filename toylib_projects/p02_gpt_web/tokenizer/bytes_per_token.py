"""Dumps the tokens per byte statistics for the given tokenizer.

Sample command:
python toylib_projects/p02_gpt_web/tokenizer/bytes_per_token.py \
  --tokenizer gpt2 \
  --output-path toylib_projects/p02_gpt_web/data/bpt_gpt2.npy
"""

import argparse
import numpy as np
from transformers import AutoTokenizer


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


def token_bytes(tokenizer, token_id: int) -> int:
    """Count a token's bytes in the text.

    Decoding one token at a time would not work: tokens that hold part of a UTF-8
    character come back as the three-byte replacement character. GPT-2 stores each
    byte as one printable character, so we count those instead.
    """
    return len(tokenizer.convert_ids_to_tokens(token_id))


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
            bpt.append(token_bytes(tokenizer, token_id))
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
