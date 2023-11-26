
import tokenizers


def basic_tokenizer() -> tokenizers.Tokenizer:
    """Creates tokenizer."""

    tokenizer = tokenizers.Tokenizer(
        tokenizers.models.WordLevel(
            {
                token: i
                for i, token in enumerate(
                    list("LAGVSERTIDPKQNFYMHWCXBUZO") +
                    list("?<>._-")
                )
            },
            unk_token="?",
        )
    )

    tokenizer.pre_tokenizer = tokenizers.pre_tokenizers.Split(
        tokenizers.Regex("[A-Z]"), behavior="removed", invert=True)

    tokenizer.add_special_tokens(list("?<>._-"))
    tokenizer.post_processor = tokenizers.processors.TemplateProcessing(
        single="$A",  # < $A >
        pair=None,
        special_tokens=[("<", 26), (">", 27)],
    )
    return tokenizer

