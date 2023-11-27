
import tokenizers


MOGWAI_VOCAB = list("ARNDCQEGHILKMFPSTWYV")


def basic_tokenizer() -> tokenizers.Tokenizer:
    """Creates tokenizer.

    Uses mogwai vocab
    https://github.com/songlab-cal/mogwai/blob/main/mogwai/vocab.py
    """

    tokenizer = tokenizers.Tokenizer(
        tokenizers.models.WordLevel(
            {
                token: i
                for i, token in enumerate(
                    MOGWAI_VOCAB + list("?<>._-")
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
    tokenizer.decoder = tokenizers.decoders.Metaspace(add_prefix_space=False)
    return tokenizer

