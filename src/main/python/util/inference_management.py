import json
import logging
import os

import torch

os.environ["TRANSFORMERS_CACHE"] = "/app/src/main/python/models"
from transformers import pipeline

logging.getLogger().setLevel("DEBUG")

logging.debug(
    "{} Inference Management - Visible CUDA Capable GPUs.".format(
        torch.cuda.device_count()
    )
)


class ZeroComprehensionEngine(object):
    """
    This class will instantiate two models which will run on the most
    efficient hardware available; GPU or CPU. Two models will be loaded to
    memory; one for zero-shot inference which works by posing each candidate
    label as a "hypothesis" and the sequence which we want to classify as the
    "premise". This will softmax the scores for entailment vs. contradiction
    for each candidate label independently.
    """

    def __init__(self):

        try:
            # With GPU acceleration
            self.sentiment_classifier = pipeline(
                "sentiment-analysis",
                model="/app/src/main/python/models/bert-base-multilingual-uncased-sentiment",
                device=0,
            )
            self.zero_classifier = pipeline(
                "zero-shot-classification",
                model="/app/src/main/python/models/xlm-roberta-large-xnli",
                device=0,
            )
            logging.info("Using GPU acceleration, this will be more efficient.")
        except:
            # Without GPU acceleration
            self.sentiment_classifier = pipeline(
                "sentiment-analysis",
                model="/app/src/main/python/models/bert-base-multilingual-uncased-sentiment",
            )
            self.zero_classifier = pipeline(
                "zero-shot-classification",
                model="/app/src/main/python/models/xlm-roberta-large-xnli",
            )
            logging.info("Using CPU only, this will be less efficient.")

    def free_comprehend(self, comment_text, candidate_labels):
        zero_classifier_response = {
            "premise": self.zero_classifier(
                comment_text, candidate_labels, multi_class=True
            )
        }
        logging.debug(zero_classifier_response)
        sentiment_classifier_response = self.sentiment_classifier(comment_text)
        response = dict(
            zip(zero_classifier_response["premise"]["labels"], zero_classifier_response["premise"]["scores"],)
        )
        response["input_str"] = zero_classifier_response["premise"]["sequence"]
        response["sentiment_label"] = sentiment_classifier_response[0]["label"]
        response["sentiment_confidence"] = sentiment_classifier_response[0]["score"]
        return response
