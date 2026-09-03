import os
import sys

import joblib
import numpy as np
import streamlit as st

from sentence_transformers import SentenceTransformer


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="Banking Intent Classifier",
    page_icon="🏦",
    layout="centered"
)


# ============================================================
# PATHS
# ============================================================

BASE_DIR = os.path.dirname(
    os.path.abspath(__file__)
)

PROJECT_DIR = os.path.dirname(
    BASE_DIR
)

MODEL_DIR = os.path.join(
    PROJECT_DIR,
    "models"
)

SRC_DIR = os.path.join(
    PROJECT_DIR,
    "src"
)


# ============================================================
# PREPROCESSING
# ============================================================

# Imported from src/preprocessing.py so the app applies exactly the same
# preprocessing the models were trained on.

sys.path.insert(0, SRC_DIR)

from preprocessing import preprocess_query


# ============================================================
# LOAD MODEL CONFIG
# ============================================================

config = joblib.load(
    os.path.join(
        MODEL_DIR,
        "final_config.pkl"
    )
)

MODEL_NAME = config["model_name"]
PROBABILITY_THRESHOLD = config.get(
    "probability_threshold",
    0.60
)

SVM_MARGIN_THRESHOLD = config.get(
    "svm_margin_threshold",
    0.10
)


# ============================================================
# LOAD CLASSIFIER
# ============================================================

@st.cache_resource
def load_classifier():

    if MODEL_NAME == "Embeddings + MLP":

        return joblib.load(
            os.path.join(
                MODEL_DIR,
                "semantic_mlp.pkl"
            )
        )

    elif MODEL_NAME == "Embeddings + Linear SVM":

        return joblib.load(
            os.path.join(
                MODEL_DIR,
                "semantic_svm.pkl"
            )
        )

    elif MODEL_NAME == "Embeddings + Logistic Regression":

        return joblib.load(
            os.path.join(
                MODEL_DIR,
                "semantic_logistic_regression.pkl"
            )
        )

    elif MODEL_NAME == "TF-IDF + Linear SVM":

        return joblib.load(
            os.path.join(
                MODEL_DIR,
                "linear_svm.pkl"
            )
        )

    elif MODEL_NAME == "TF-IDF + Logistic Regression":

        return joblib.load(
            os.path.join(
                MODEL_DIR,
                "logistic_regression.pkl"
            )
        )

    else:

        raise ValueError(
            f"Unknown model: {MODEL_NAME}"
        )


classifier = load_classifier()


# ============================================================
# LOAD REPRESENTATION MODEL
# ============================================================

@st.cache_resource
def load_embedding_model():

    return SentenceTransformer(
        config["embedding_model"]
    )


if "Embeddings" in MODEL_NAME:

    embedding_model = load_embedding_model()

else:

    vectorizer = joblib.load(
        os.path.join(
            MODEL_DIR,
            "tfidf_vectorizer.pkl"
        )
    )


# ============================================================
# FEATURE CREATION
# ============================================================

def create_features(text):

    processed_text = preprocess_query(text)

    if "Embeddings" in MODEL_NAME:

        return embedding_model.encode(
            [processed_text]
        )

    return vectorizer.transform(
        [processed_text]
    )


# ============================================================
# PREDICTION
# ============================================================

def predict_intent(text):

    features = create_features(text)

    # --------------------------------------------------------
    # Probability-based model
    # --------------------------------------------------------

    if hasattr(classifier, "predict_proba"):

        probabilities = classifier.predict_proba(
            features
        )[0]

        sorted_indices = np.argsort(
            probabilities
        )[::-1]

        top_index = sorted_indices[0]
        second_index = sorted_indices[1]

        top_score = probabilities[top_index]
        second_score = probabilities[second_index]

        top_intent = classifier.classes_[top_index]
        second_intent = classifier.classes_[second_index]

        margin = (
            top_score -
            second_score
        )

        if top_score < PROBABILITY_THRESHOLD:

            status = "Low confidence"

        else:

            status = "High confidence"

        return {
            "intent": top_intent,
            "score": top_score,
            "score_type": "probability",
            "second_intent": second_intent,
            "second_score": second_score,
            "margin": margin,
            "status": status
        }

    # --------------------------------------------------------
    # SVM
    # --------------------------------------------------------

    scores = classifier.decision_function(
        features
    )

    sorted_indices = np.argsort(
        scores[0]
    )[::-1]

    top_index = sorted_indices[0]
    second_index = sorted_indices[1]

    top_score = scores[0][top_index]
    second_score = scores[0][second_index]

    top_intent = classifier.classes_[top_index]
    second_intent = classifier.classes_[second_index]

    margin = (
        top_score -
        second_score
    )

    if margin < SVM_MARGIN_THRESHOLD:

        status = "Ambiguous prediction"

    else:

        status = "Clear prediction"

    return {
        "intent": top_intent,
        "score": top_score,
        "score_type": "decision_score",
        "second_intent": second_intent,
        "second_score": second_score,
        "margin": margin,
        "status": status
    }


# ============================================================
# UI
# ============================================================

st.title("🏦 Banking Intent Classifier")

st.write(
    "Enter a customer query to predict the most likely "
    "BANKING77 intent."
)


st.divider()


# ============================================================
# EXAMPLE QUERIES
# ============================================================

st.subheader("Try an example")

examples = [
    "My card hasn't arrived yet",
    "I forgot my PIN",
    "Why was I charged an extra fee?",
    "How do I transfer money?",
    "I need help with something"
]

selected_example = st.selectbox(
    "Example query",
    [""] + examples
)


# ============================================================
# QUERY INPUT
# ============================================================

query = st.text_area(
    "Customer query",
    value=selected_example,
    placeholder="Example: Where is my new card?",
    height=100
)


predict_button = st.button(
    "Predict Intent",
    type="primary",
    use_container_width=True
)


# ============================================================
# PREDICTION OUTPUT
# ============================================================

if predict_button:

    # Guard on the preprocessed text, not the raw text: input made only of
    # digits or punctuation ("12345", "!!!") is non-empty raw but reduces to
    # an empty string, which would otherwise produce an arbitrary intent.
    if not preprocess_query(query):

        st.warning(
            "Please enter a customer query containing words."
        )

    else:

        result = predict_intent(query)

        st.divider()

        st.subheader("Prediction")

        st.write(
            f"**Intent:** `{result['intent']}`"
        )

        # ----------------------------------------------------
        # Probability model
        # ----------------------------------------------------

        if result["score_type"] == "probability":

            st.metric(
                "Prediction Probability",
                f"{result['score']:.2%}"
            )

            if result["status"] == "High confidence":

                st.success(
                    "The model is reasonably confident "
                    "in this prediction."
                )

            else:

                st.warning(
                    "The model is not very confident "
                    "in this prediction."
                )

        # ----------------------------------------------------
        # SVM
        # ----------------------------------------------------

        else:

            st.metric(
                "Decision Score",
                f"{result['score']:.4f}"
            )

            st.metric(
                "Top-2 Margin",
                f"{result['margin']:.4f}"
            )

            if result["status"] == "Ambiguous prediction":

                st.warning(
                    "The model is uncertain between "
                    "multiple intents."
                )

                st.write(
                    f"Possible alternative: "
                    f"`{result['second_intent']}`"
                )

            else:

                st.success(
                    "The model has a clear preference "
                    "for this intent."
                )


# ============================================================
# MODEL INFORMATION
# ============================================================

with st.expander("Model information"):

    st.write(
        f"**Model:** {MODEL_NAME}"
    )

    if config["embedding_model"]:

        st.write(
            f"**Embedding model:** "
            f"{config['embedding_model']}"
        )

    st.write(
        "**Dataset:** BANKING77"
    )

    st.write(
        "**Number of intents:** 77"
    )