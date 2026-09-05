from sumy.parsers.plaintext import PlaintextParser
from sumy.nlp.tokenizers import Tokenizer
from sumy.summarizers.lsa import LsaSummarizer


def extractive_summary(text, length):
    """
    Generate an extractive summary using LSA.
    """

    parser = PlaintextParser.from_string(text, Tokenizer("english"))
    summarizer = LsaSummarizer()

    if length == "Short":
        sentences = 3
    elif length == "Medium":
        sentences = 5
    else:
        sentences = 8

    summary = summarizer(parser.document, sentences)

    return " ".join(str(sentence) for sentence in summary)