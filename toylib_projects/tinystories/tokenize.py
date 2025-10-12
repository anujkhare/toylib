import argparse
import numpy as np

import transformers


def process_text_file(
    input_path: str,
    output_path: str,
    tokenizer: transformers.PreTrainedTokenizer,
    delimiter: str = "\n",
    encoding: str = "utf-8",
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
        np.array(tokenizer(example)["input_ids"], dtype=np.uint16)
        for example in examples
    ]

    # Concatenate all tokenized examples into a single array
    token_array = np.array(tokenized_examples, dtype=object)

    # Save to disk
    np.save(output_path, token_array)
    print(f"Saved {token_array.shape[0]} tokenized examples to {output_path}")


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
        "--tokenizer-version",
        type=str,
        required=True,
        default="gpt2",
        help="Tokenizer version from transformers to use (e.g., 'gpt2')",
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

    process_text_file(
        input_path=args.input_path,
        output_path=args.output_path,
        tokenizer_version=args.tokenizer_version,
        delimiter=args.delimiter,
        encoding=args.encoding,
    )


if __name__ == "__main__":
    main()
