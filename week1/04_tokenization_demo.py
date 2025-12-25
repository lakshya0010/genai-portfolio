from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")


texts = [
    "I love machine learning.",
    "Machine learning is transforming the world.",
    "Large language models work on tokens, not raw text.",
    "This is a longer sentence meant to show how token counts increase with length."
]

for text in texts:
    tokens = tokenizer.tokenize(text)
    token_ids = tokenizer.convert_tokens_to_ids(tokens)

    print("TEXT:", text)
    print("TOKENS:", tokens)
    print("TOKEN COUNT:", len(tokens))
    print("-" * 50)