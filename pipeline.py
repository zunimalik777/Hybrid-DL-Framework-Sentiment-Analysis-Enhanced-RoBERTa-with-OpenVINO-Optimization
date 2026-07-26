"""
End-to-end orchestration of all six objectives:

    1. RoBERTa fine-tuning & evaluation
    2. Two-Layer GRU (+ Word2Vec) training & evaluation
    3. VADER baseline evaluation
    4. Word2Vec embedding training (feeds step 2)
    5. ONNX export (RoBERTa + GRU)
    6. OpenVINO FP32 / FP16 conversion + NNCF INT8 quantization,
       each re-evaluated so accuracy/latency/size trade-offs are visible.

This module is imported from the Colab notebook; each function can also
be run independently.
"""

import os
import numpy as np
import pandas as pd

from . import config
from . import data_utils
from . import vader_module
from . import word2vec_module
from . import gru_model
from . import roberta_model
from . import metrics
from . import export_onnx
from . import openvino_utils


# --------------------------------------------------------------------------
# 1. Data
# --------------------------------------------------------------------------
def prepare_data():
    config.set_seed()
    datasets = data_utils.load_all_datasets()
    splits = {}
    for name, df in datasets.items():
        train_df, val_df, test_df = data_utils.split_dataset(df)
        splits[name] = {"train": train_df, "val": val_df, "test": test_df}
        print(f"[{name}] train={len(train_df)} val={len(val_df)} test={len(test_df)}")
    return datasets, splits


# --------------------------------------------------------------------------
# 3. VADER baseline (no training required)
# --------------------------------------------------------------------------
def run_vader_baseline(test_df, dataset_label=""):
    vader = vader_module.VaderBaseline()
    preds = vader.predict_batch(test_df["text"].tolist())
    labels = test_df["label"].to_numpy()
    result = metrics.compute_metrics(labels, preds)
    print(f"[VADER | {dataset_label}] {result}")
    return result, np.array(preds), labels


# --------------------------------------------------------------------------
# 4 + 2. Word2Vec + Two-Layer GRU
# --------------------------------------------------------------------------
def run_word2vec_gru(train_df, val_df, test_df, dataset_label=""):
    w2v_model = word2vec_module.train_word2vec(train_df["text"].tolist())
    vocab = word2vec_module.build_vocab(w2v_model)
    embedding_matrix = word2vec_module.build_embedding_matrix(w2v_model, vocab)

    train_seq = word2vec_module.texts_to_sequences(train_df["text"], vocab)
    val_seq = word2vec_module.texts_to_sequences(val_df["text"], vocab)
    test_seq = word2vec_module.texts_to_sequences(test_df["text"], vocab)

    train_loader, val_loader, test_loader = gru_model.make_dataloaders(
        train_seq, train_df["label"].to_numpy(),
        val_seq, val_df["label"].to_numpy(),
        test_seq, test_df["label"].to_numpy(),
    )

    model = gru_model.TwoLayerGRUClassifier(embedding_matrix)
    model, history = gru_model.train_gru(model, train_loader, val_loader)

    preds, labels = gru_model.predict_gru(model, test_loader)
    result = metrics.compute_metrics(labels, preds)
    print(f"[GRU | {dataset_label}] {result}")

    artifacts = {
        "model": model,
        "w2v_model": w2v_model,
        "vocab": vocab,
        "embedding_matrix": embedding_matrix,
        "test_seq": test_seq,
        "history": history,
    }
    return result, preds, labels, artifacts


# --------------------------------------------------------------------------
# 1. RoBERTa
# --------------------------------------------------------------------------
def run_roberta(train_df, val_df, test_df, dataset_label=""):
    tokenizer, model, trainer = roberta_model.train_roberta(train_df, val_df)
    preds, labels = roberta_model.evaluate_roberta(model, tokenizer, test_df)
    result = metrics.compute_metrics(labels, preds)
    print(f"[RoBERTa | {dataset_label}] {result}")
    return result, preds, labels, {"model": model, "tokenizer": tokenizer}


# --------------------------------------------------------------------------
# 5 + 6. ONNX -> OpenVINO FP32 / FP16 -> NNCF INT8, for one model
# --------------------------------------------------------------------------
def optimize_gru_pipeline(gru_artifacts, test_df, model_name="gru_sentiment"):
    model = gru_artifacts["model"]
    vocab = gru_artifacts["vocab"]
    test_seq = gru_artifacts["test_seq"]
    labels = test_df["label"].to_numpy()

    onnx_path = export_onnx.export_gru_to_onnx(model)

    fp32_xml = openvino_utils.convert_onnx_to_openvino(
        onnx_path, config.OV_FP32_DIR, model_name, precision="FP32"
    )
    fp16_xml = openvino_utils.convert_onnx_to_openvino(
        onnx_path, config.OV_FP16_DIR, model_name, precision="FP16"
    )

    calib_data = openvino_utils.build_gru_calibration_data(test_seq)
    int8_xml = openvino_utils.quantize_int8_nncf(
        fp32_xml, calib_data, input_name="input_ids",
        output_dir=config.OV_INT8_DIR, model_name=model_name,
    )

    variants = {"ONNX (FP32)": onnx_path, "OpenVINO FP32": fp32_xml,
                "OpenVINO FP16": fp16_xml, "OpenVINO INT8 (NNCF)": int8_xml}

    results = {}
    for variant_name, path in variants.items():
        preds = _predict_gru_variant(variant_name, path, test_seq)
        m = metrics.compute_metrics(labels, preds)
        size_mb = metrics.model_size_mb(path if not path.endswith(".xml") else path)
        m["size_mb"] = size_mb
        results[variant_name] = m
        print(f"[GRU | {variant_name}] {m}")

    return results


def _predict_gru_variant(variant_name, path, test_seq):
    preds = []
    if variant_name.startswith("ONNX"):
        session = export_onnx.get_onnx_session(path)
        for i in range(len(test_seq)):
            out = session.run(None, {"input_ids": test_seq[i:i + 1].astype(np.int64)})[0]
            preds.append(int(np.argmax(out, axis=1)[0]))
    else:
        compiled = openvino_utils.compile_model(path)
        for i in range(len(test_seq)):
            out = openvino_utils.run_openvino_inference(
                compiled, {"input_ids": test_seq[i:i + 1].astype(np.int64)}
            )
            logits = list(out.values())[0]
            preds.append(int(np.argmax(logits, axis=1)[0]))
    return np.array(preds)


def optimize_roberta_pipeline(roberta_artifacts, test_df, model_name="roberta_sentiment",
                               max_len=config.MAX_SEQ_LEN):
    model = roberta_artifacts["model"]
    tokenizer = roberta_artifacts["tokenizer"]
    labels = test_df["label"].to_numpy()
    texts = test_df["text"].tolist()

    onnx_path = export_onnx.export_roberta_to_onnx(model, tokenizer)

    fp32_xml = openvino_utils.convert_onnx_to_openvino(
        onnx_path, config.OV_FP32_DIR, model_name, precision="FP32"
    )
    fp16_xml = openvino_utils.convert_onnx_to_openvino(
        onnx_path, config.OV_FP16_DIR, model_name, precision="FP16"
    )

    calib_data = openvino_utils.build_roberta_calibration_data(texts, tokenizer)
    int8_xml = openvino_utils.quantize_int8_nncf(
        fp32_xml, calib_data, input_name=None,
        output_dir=config.OV_INT8_DIR, model_name=model_name,
    )

    variants = {"ONNX (FP32)": onnx_path, "OpenVINO FP32": fp32_xml,
                "OpenVINO FP16": fp16_xml, "OpenVINO INT8 (NNCF)": int8_xml}

    results = {}
    for variant_name, path in variants.items():
        preds = _predict_roberta_variant(variant_name, path, texts, tokenizer, max_len)
        m = metrics.compute_metrics(labels, preds)
        m["size_mb"] = metrics.model_size_mb(path)
        results[variant_name] = m
        print(f"[RoBERTa | {variant_name}] {m}")

    return results


def _predict_roberta_variant(variant_name, path, texts, tokenizer, max_len):
    preds = []
    if variant_name.startswith("ONNX"):
        session = export_onnx.get_onnx_session(path)
        for text in texts:
            enc = tokenizer(text, return_tensors="np", truncation=True, max_length=max_len, padding="max_length")
            out = session.run(None, {
                "input_ids": enc["input_ids"].astype(np.int64),
                "attention_mask": enc["attention_mask"].astype(np.int64),
            })[0]
            preds.append(int(np.argmax(out, axis=1)[0]))
    else:
        compiled = openvino_utils.compile_model(path)
        for text in texts:
            enc = tokenizer(text, return_tensors="np", truncation=True, max_length=max_len, padding="max_length")
            out = openvino_utils.run_openvino_inference(compiled, {
                "input_ids": enc["input_ids"].astype(np.int64),
                "attention_mask": enc["attention_mask"].astype(np.int64),
            })
            logits = list(out.values())[0]
            preds.append(int(np.argmax(logits, axis=1)[0]))
    return np.array(preds)


# --------------------------------------------------------------------------
# Full run
# --------------------------------------------------------------------------
def run_full_pipeline(dataset_name="imdb"):
    """Runs objectives 1-6 sequentially on the requested dataset
    ("imdb", "sentiment140", or "combined") and returns a consolidated
    results table."""
    datasets, splits = prepare_data()
    split = splits[dataset_name]
    train_df, val_df, test_df = split["train"], split["val"], split["test"]

    all_results = {}

    vader_res, *_ = run_vader_baseline(test_df, dataset_name)
    all_results["VADER (baseline)"] = vader_res

    gru_res, _, _, gru_artifacts = run_word2vec_gru(train_df, val_df, test_df, dataset_name)
    all_results["Word2Vec + Two-Layer GRU"] = gru_res

    roberta_res, _, _, roberta_artifacts = run_roberta(train_df, val_df, test_df, dataset_name)
    all_results["RoBERTa (fine-tuned)"] = roberta_res

    gru_opt_results = optimize_gru_pipeline(gru_artifacts, test_df)
    for k, v in gru_opt_results.items():
        all_results[f"GRU - {k}"] = v

    roberta_opt_results = optimize_roberta_pipeline(roberta_artifacts, test_df)
    for k, v in roberta_opt_results.items():
        all_results[f"RoBERTa - {k}"] = v

    table = metrics.results_table(all_results)
    out_csv = os.path.join(config.RESULTS_DIR, f"results_{dataset_name}.csv")
    table.to_csv(out_csv)
    print(f"Saved consolidated results -> {out_csv}")
    return table
