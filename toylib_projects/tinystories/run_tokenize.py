"""Script to process a text file with delimited examples, tokenize each example.

Example usage:

python toylib_projects/tinystories/run_tokenize.py \
    --input-path=toylib_projects/tinystories/data/tinystories_sample.txt \
    --output-path=toylib_projects/tinystories/data/tokenized.npy \
    --tokenizer=gpt2 \
    --delimiter="<|endoftext|>"

# Full TinyStories v2
python toylib_projects/tinystories/run_tokenize.py \
    --input-path=toylib_projects/tinystories/data/TinyStoriesV2-GPT4-valid.txt \
    --output-path=toylib_projects/tinystories/data/TinyStoriesV2-GPT4-valid-tokenized.npy \
    --tokenizer=gpt2 \
    --delimiter="<|endoftext|>"

python toylib_projects/tinystories/run_tokenize.py \
    --input-path=toylib_projects/tinystories/data/TinyStoriesV2-GPT4-train.txt \
    --output-path=toylib_projects/tinystories/data/TinyStoriesV2-GPT4-train-tokenized.npy \
    --tokenizer=gpt2 \
    --delimiter="<|endoftext|>"

"""

import argparse
import numpy as np

from transformers import AutoTokenizer
from typing import Any


def process_text_file(
    input_path: str,
    output_path: str,
    tokenizer: Any,
    delimiter: str = "\n",
    encoding: str = "utf-8",
    max_sequence_length: int = 1024,
) -> None:
    """
    Process a text file with delimited examples, tokenize each example,
    and save as a numpy array.

    Args:
        input_path: path to input text file
        output_path: path to save the numpy array (.npy file)
        tokenizer_version: Version/type of tokenizer to use
        delimiter: Delimiter separating examples (default: newline)
        encoding: Text file encoding (default: utf-8)
    """
    # Read the text file
    with open(input_path, "r", encoding=encoding) as f:
        content = f.read()

    # Split into examples
    examples = content.split(delimiter)

    # Filter out empty examples
    examples = [ex.strip() for ex in examples if ex.strip()]

    # Tokenize each example
    tokenized_examples = [
        np.array(
            tokenizer(example, truncation=True, max_length=max_sequence_length)[
                "input_ids"
            ],
            dtype=np.uint16,
        )
        for example in examples
    ]


    # Concatenate all tokenized examples into a single array
    token_array = np.array(tokenized_examples, dtype=object)

    # Save to disk
    np.save(output_path, token_array, allow_pickle=True)
    print(f"Saved {token_array.shape[0]} tokenized examples to {output_path}")

    # Calculate the lengths of each example and save
    lengths = [len(ex) for ex in tokenized_examples]
    np.save(
        output_path.replace(".npy", "_lengths.npy"), np.array(lengths, dtype=np.uint16)
    )
    print(
        f"Example lengths: min {min(lengths)}, max {max(lengths)}, avg {sum(lengths) / len(lengths):.2f}"
    )


def parse_command_line_args():
    parser = argparse.ArgumentParser(
        description="Process text file with delimited examples and tokenize them"
    )

    parser.add_argument(
        "--input-path", type=str, required=True, help="Path to input text file"
    )

    parser.add_argument(
        "--output-path",
        type=str,
        required=True,
        help="Path to save the tokenized numpy array (.npy file)",
    )

    parser.add_argument(
        "--tokenizer",
        type=str,
        required=True,
        default="gpt2",
        help="Tokenizer version from transformers to use (e.g., 'gpt2')",
    )

    parser.add_argument(
        "--max-sequence-length",
        type=int,
        required=False,
        default=1024,
        help="Maximum sequence length for tokenization (default: 1024)",
    )

    parser.add_argument(
        "--delimiter",
        type=str,
        default="\n",
        help="Delimiter separating examples (default: newline)",
    )

    parser.add_argument(
        "--encoding",
        type=str,
        default="utf-8",
        help="Text file encoding (default: utf-8)",
    )

    return parser.parse_args()


def main():
    args = parse_command_line_args()
    tokenizer = AutoTokenizer.from_pretrained("gpt2")

    process_text_file(
        input_path=args.input_path,
        output_path=args.output_path,
        delimiter=args.delimiter,
        tokenizer=tokenizer,
        encoding=args.encoding,
        max_sequence_length=args.max_sequence_length,
    )


if __name__ == "__main__":
    main()
