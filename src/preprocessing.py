"""
Canonical text preprocessing for the BANKING77 intent classifier.

This is the single source of truth for turning a raw customer query into the
form the models were trained on. Notebook 01 uses it to build the processed
datasets, and the inference layer (Notebook 05 and the Streamlit app) imports
the same function, so training and serving cannot drift apart.

The pipeline is intentionally conservative:

1. Lowercase.
2. Remove non-alphabetic characters (this also strips digits).
3. Normalize whitespace.
4. Tokenize with NLTK, then rejoin.

Stopwords are deliberately kept: intent often hinges on negation and function
words ("do not", "cannot", "wasn't"), so removing them would destroy signal.

Lemmatization is deliberately not applied. It was evaluated and left out
because BANKING77 queries are short and already close to their base forms.

Note on step 4: tokenization is not redundant after step 2. NLTK splits
"cannot" into "can not", which affects the unigram/bigram vocabulary.
"""

import re

import nltk
from nltk.tokenize import word_tokenize


def _ensure_tokenizer_data():
    """Make sure the NLTK tokenizer models are available.

    Notebook 01 downloads these explicitly, but the Streamlit app runs on
    machines that have never executed a notebook (a fresh deployment, for
    example). Without this, the first query raises LookupError. The check is
    cheap and a no-op once the data is present.
    """

    for resource in ("punkt_tab", "punkt"):

        try:
            nltk.data.find(f"tokenizers/{resource}")

        except LookupError:

            try:
                nltk.download(resource, quiet=True)

            except Exception:
                # Offline with the data already cached elsewhere, or a
                # transient failure; word_tokenize will raise a clear error
                # if the resource is genuinely unusable.
                pass


_ensure_tokenizer_data()


def preprocess_text(text):
    """Normalize a raw customer query into model-ready text."""

    # Convert to lowercase
    text = str(text).lower()

    # Remove non-alphabetic characters
    text = re.sub(r"[^a-z\s]", " ", text)

    # Normalize whitespace
    text = re.sub(r"\s+", " ", text).strip()

    # Tokenize
    tokens = word_tokenize(text)

    # Reconstruct text
    return " ".join(tokens)


# Backwards-compatible alias: the inference code referred to this name.
preprocess_query = preprocess_text
