

from typing import List, Dict


def safe_import_transformers():
    try:
        from transformers import BertTokenizer
        return BertTokenizer
    except Exception as exc:  # pragma: no cover
        print("Could not import 'transformers'. Please install it:")
        print("    pip install transformers")
        raise exc


def print_section(title: str) -> None:
    print("\n" + "=" * 80)
    print(title)
    print("=" * 80)


def explain_tokenization(tokenizer, text: str) -> None:
    print_section("1) Tokenize a simple sentence")
    print(f"Input text: {text}")

    # Basic tokenization
    tokens: List[str] = tokenizer.tokenize(text)
    print(f"Tokens: {tokens}")

    # Convert tokens to ids (numbers that BERT understands)
    input_ids: List[int] = tokenizer.convert_tokens_to_ids(tokens)
    print(f"Token IDs: {input_ids}")

    # Add special tokens [CLS] at the beginning and [SEP] at the end
    print_section("2) Adding special tokens [CLS] and [SEP]")
    encoded_with_specials: Dict[str, List[int]] = tokenizer.encode_plus(
        text,
        add_special_tokens=True,    # adds [CLS] and [SEP]
        return_attention_mask=True, # mask tells BERT which positions are real text vs padding
        return_tensors=None         # keep as plain Python lists for easy printing
    )
    print("Encoded keys:", list(encoded_with_specials.keys()))
    print("input_ids:", encoded_with_specials["input_ids"])  # includes [CLS] and [SEP]
    print("attention_mask:", encoded_with_specials["attention_mask"])  # 1 for real tokens

    # Decode back to text (special tokens removed by default)
    decoded_text: str = tokenizer.decode(encoded_with_specials["input_ids"], skip_special_tokens=True)
    print("Decoded back (without special tokens):", decoded_text)


def show_subword_behavior(tokenizer) -> None:
    print_section("3) Subword tokenization (WordPiece)")

    tricky_words = [
        "unbelievable",
        "playground",
        "xylophonic",  # unusual word that might get split a lot
    ]

    for word in tricky_words:
        sub_tokens = tokenizer.tokenize(word)
        print(f"Word: {word:>12} -> Sub-tokens: {sub_tokens}")
        sub_token_ids = tokenizer.convert_tokens_to_ids(sub_tokens)
        print(f"               -> IDs: {sub_token_ids}")

    print("\nNote: '##' prefix means the piece continues a previous subword.")


def demonstrate_padding_and_truncation(tokenizer) -> None:
    print_section("4) Batching: padding and truncation")

    sentences = [
        "BERT is great!",
        "Tokenizers turn text into numbers.",
        "This sentence is a bit longer than the others to show truncation.",
    ]

    encoded = tokenizer(
        sentences,
        add_special_tokens=True,
        padding="max_length",   # pad every sequence to the same length
        truncation=True,         # cut off sequences that are too long
        max_length=12,           # small on purpose to show truncation
        return_attention_mask=True,
        return_tensors=None,
    )

    print("input_ids (each row is one sentence):")
    for row in encoded["input_ids"]:
        print(row)

    print("\nattention_mask (1 = real token, 0 = padding):")
    for row in encoded["attention_mask"]:
        print(row)


def main() -> None:
    BertTokenizer = safe_import_transformers()

    # 'bert-base-uncased' is the classic English BERT.
    # "uncased" means it lowercases text (Hello -> hello).
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

    explain_tokenization(tokenizer, "Hello, how are you doing today?")
    show_subword_behavior(tokenizer)
    demonstrate_padding_and_truncation(tokenizer)

    print_section("5) Recap")
    print("- Tokenizers split text into tokens and map them to IDs.")
    print("- BERT uses special tokens: [CLS] at start, [SEP] at end.")
    print("- Unknown or rare words are broken into subwords using WordPiece.")
    print("- In batches, we pad to equal lengths and use attention masks to ignore padding.")


if __name__ == "__main__":
    main()
