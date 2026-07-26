"""
Objective 4: Word2Vec

Trains a domain-specific Word2Vec (skip-gram) model on the combined
corpus and builds the vocabulary + embedding matrix consumed by the
Two-Layer GRU (Objective 2).
"""

import numpy as np
from gensim.models import Word2Vec

from . import config
from .data_utils import tokenize_simple

PAD_TOKEN = "<pad>"
UNK_TOKEN = "<unk>"


def train_word2vec(texts, vector_size=config.W2V_VECTOR_SIZE, window=config.W2V_WINDOW,
                    min_count=config.W2V_MIN_COUNT, epochs=config.W2V_EPOCHS,
                    workers=config.W2V_WORKERS) -> Word2Vec:
    tokenized = [tokenize_simple(t) for t in texts]
    model = Word2Vec(
        sentences=tokenized,
        vector_size=vector_size,
        window=window,
        min_count=min_count,
        workers=workers,
        sg=1,          # skip-gram
        epochs=epochs,
        seed=config.SEED,
    )
    return model


def build_vocab(w2v_model: Word2Vec):
    """word -> index, reserving 0 for PAD and 1 for UNK."""
    vocab = {PAD_TOKEN: 0, UNK_TOKEN: 1}
    for i, word in enumerate(w2v_model.wv.index_to_key):
        vocab[word] = i + 2
    return vocab


def build_embedding_matrix(w2v_model: Word2Vec, vocab: dict) -> np.ndarray:
    dim = w2v_model.vector_size
    matrix = np.zeros((len(vocab), dim), dtype=np.float32)
    for word, idx in vocab.items():
        if word in w2v_model.wv:
            matrix[idx] = w2v_model.wv[word]
        else:
            matrix[idx] = np.random.normal(scale=0.1, size=(dim,))
    return matrix


def texts_to_sequences(texts, vocab, max_len=config.MAX_SEQ_LEN):
    sequences = []
    for t in texts:
        tokens = tokenize_simple(t)[:max_len]
        ids = [vocab.get(tok, vocab[UNK_TOKEN]) for tok in tokens]
        if len(ids) < max_len:
            ids = ids + [vocab[PAD_TOKEN]] * (max_len - len(ids))
        sequences.append(ids)
    return np.array(sequences, dtype=np.int64)
