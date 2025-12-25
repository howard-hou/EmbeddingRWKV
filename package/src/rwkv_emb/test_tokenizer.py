"""
Basic tokenizer validation for RWKV vocab.

This script mirrors the layout of ``test.py`` and focuses on round-trip
correctness for the TRIE_TOKENIZER implementation using a few common
strings.
"""

import os
import sys

from reference.utils import TRIE_TOKENIZER

# Locate vocab file relative to this script
VOCAB_PATH = os.path.join(os.path.dirname(__file__), "reference", "rwkv_vocab_v20230424.txt")


def run_test(name: str, text: str, tokenizer: TRIE_TOKENIZER) -> None:
    """Encode and decode a string, asserting round-trip stability."""
    print(f"\n--- {name} ---")
    print(f"Input text: {repr(text)}")

    tokens = tokenizer.encode(text)
    decoded = tokenizer.decode(tokens)

    print(f"Tokens ({len(tokens)}): {tokens[:20]}{' ...' if len(tokens) > 20 else ''}")
    print(f"Decoded text: {repr(decoded)}")

    if decoded == text:
        print("Result: SUCCESS")
    else:
        print("Result: FAILURE (decoded text does not match input)")
        sys.exit(1)


def main() -> None:
    if not os.path.isfile(VOCAB_PATH):
        print(f"Vocabulary file not found: {VOCAB_PATH}")
        sys.exit(1)

    tokenizer = TRIE_TOKENIZER(VOCAB_PATH)

    print("\n=== Tokenizer Round-trip Tests ===")

    run_test("ASCII text", "The quick brown fox jumps over the lazy dog.", tokenizer)
    run_test("Mixed language", "RWKV 分词器测试 🦆🚀", tokenizer)
    run_test("Repeated tokens", "haha haha haha!!!", tokenizer)

    print("\nAll tokenizer tests passed.")


if __name__ == "__main__":
    main()
