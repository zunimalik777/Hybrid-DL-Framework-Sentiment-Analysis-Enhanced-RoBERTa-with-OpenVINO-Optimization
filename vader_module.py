"""
Objective 3: VADER (Valence Aware Dictionary and sEntiment Reasoner)

A rule/lexicon-based sentiment baseline used to benchmark the learned
models (GRU, RoBERTa) against a zero-training reference point.
"""

from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from tqdm.auto import tqdm

from . import config


class VaderBaseline:
    def __init__(self, pos_threshold=config.VADER_POS_THRESHOLD, neg_threshold=config.VADER_NEG_THRESHOLD):
        self.analyzer = SentimentIntensityAnalyzer()
        self.pos_threshold = pos_threshold
        self.neg_threshold = neg_threshold

    def score(self, text: str) -> float:
        return self.analyzer.polarity_scores(text)["compound"]

    def predict_label(self, text: str) -> int:
        """Binary prediction: 1 = positive, 0 = negative.
        Neutral scores (between thresholds) are broken toward negative,
        matching the binary IMDb / Sentiment140 label scheme."""
        compound = self.score(text)
        return 1 if compound >= self.pos_threshold else 0

    def predict_batch(self, texts):
        return [self.predict_label(t) for t in tqdm(texts, desc="VADER inference")]
