"""
Objective 5: ONNX Export

Exports both the RoBERTa transformer and the Two-Layer GRU to the ONNX
format, which is the common entry point for the OpenVINO (FP32/FP16)
and NNCF INT8 conversions in `openvino_utils.py`.
"""

import os
import torch
import onnx
import onnxruntime as ort

from . import config


def export_roberta_to_onnx(model, tokenizer, save_path=None, max_len=config.MAX_SEQ_LEN):
    save_path = save_path or os.path.join(config.ONNX_DIR, "roberta_sentiment.onnx")
    model.eval()
    model.to("cpu")

    dummy = tokenizer(
        "This movie was absolutely fantastic and I loved every minute of it.",
        return_tensors="pt",
        truncation=True,
        max_length=max_len,
        padding="max_length",
    )

    torch.onnx.export(
        model,
        (dummy["input_ids"], dummy["attention_mask"]),
        save_path,
        input_names=["input_ids", "attention_mask"],
        output_names=["logits"],
        dynamic_axes={
            "input_ids": {0: "batch_size", 1: "sequence"},
            "attention_mask": {0: "batch_size", 1: "sequence"},
            "logits": {0: "batch_size"},
        },
        opset_version=config.ONNX_OPSET,
        do_constant_folding=True,
    )
    _validate_onnx(save_path)
    print(f"[ONNX] RoBERTa exported -> {save_path}")
    return save_path


def export_gru_to_onnx(model, save_path=None, max_len=config.MAX_SEQ_LEN):
    save_path = save_path or os.path.join(config.ONNX_DIR, "gru_sentiment.onnx")
    model.eval()
    model.to("cpu")

    dummy_input = torch.randint(low=0, high=1000, size=(1, max_len), dtype=torch.long)

    torch.onnx.export(
        model,
        dummy_input,
        save_path,
        input_names=["input_ids"],
        output_names=["logits"],
        dynamic_axes={"input_ids": {0: "batch_size"}, "logits": {0: "batch_size"}},
        opset_version=config.ONNX_OPSET,
        do_constant_folding=True,
    )
    _validate_onnx(save_path)
    print(f"[ONNX] GRU exported -> {save_path}")
    return save_path


def _validate_onnx(path):
    onnx_model = onnx.load(path)
    onnx.checker.check_model(onnx_model)


def run_onnx_inference(onnx_path, feed_dict):
    """Runs a single inference with ONNX Runtime; returns logits (numpy)."""
    session = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
    outputs = session.run(None, feed_dict)
    return outputs[0]


def get_onnx_session(onnx_path):
    return ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
