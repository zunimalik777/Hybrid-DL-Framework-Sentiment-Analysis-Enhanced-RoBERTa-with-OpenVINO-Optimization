"""
Objective 6: OpenVINO Optimization (FP32 / FP16) + NNCF INT8 Quantization

Pipeline:
    ONNX  --(ov.convert_model)-->  OpenVINO IR (FP32)
    OpenVINO IR (FP32)  --(save_model compress_to_fp16=True)-->  OpenVINO IR (FP16)
    OpenVINO IR (FP32)  --(nncf.quantize, post-training static quantization)-->  INT8 IR
"""

import os
import numpy as np
import openvino as ov

from . import config


# --------------------------------------------------------------------------
# ONNX -> OpenVINO IR (FP32 / FP16)
# --------------------------------------------------------------------------
def convert_onnx_to_openvino(onnx_path: str, output_dir: str, model_name: str,
                              precision: str = "FP32"):
    """Converts an ONNX model to OpenVINO IR at the requested precision.

    precision: "FP32" or "FP16"
    """
    os.makedirs(output_dir, exist_ok=True)
    ov_model = ov.convert_model(onnx_path)

    xml_path = os.path.join(output_dir, f"{model_name}_{precision.lower()}.xml")
    compress_to_fp16 = precision.upper() == "FP16"
    ov.save_model(ov_model, xml_path, compress_to_fp16=compress_to_fp16)
    print(f"[OpenVINO] {precision} IR saved -> {xml_path}")
    return xml_path


def compile_model(xml_path: str, device: str = "CPU"):
    core = ov.Core()
    model = core.read_model(xml_path)
    compiled = core.compile_model(model, device)
    return compiled


def run_openvino_inference(compiled_model, feed_dict: dict):
    """feed_dict: {input_name: np.ndarray}. Returns dict of output arrays."""
    result = compiled_model(feed_dict)
    outputs = {}
    for output in compiled_model.outputs:
        outputs[output.get_any_name()] = result[output]
    return outputs


# --------------------------------------------------------------------------
# NNCF INT8 Post-Training Static Quantization
# --------------------------------------------------------------------------
def quantize_int8_nncf(fp32_xml_path: str, calibration_samples: list, input_name: str,
                        output_dir: str, model_name: str, subset_size=config.NNCF_SUBSET_SIZE):
    """Quantizes an OpenVINO FP32 IR model to INT8 using NNCF post-training
    quantization with a representative calibration dataset.

    `calibration_samples`: list of np.ndarray inputs (already preprocessed /
    tokenized, matching the model's expected input shape).
    `input_name`: name of the model input the calibration samples map to
    (e.g. "input_ids" for the GRU, or a dict for multi-input RoBERTa).
    """
    import nncf

    core = ov.Core()
    fp32_model = core.read_model(fp32_xml_path)

    def transform_fn(data_item):
        # data_item is a single element yielded from calibration_samples;
        # for single-input models (GRU) it's an array, for multi-input
        # models (RoBERTa) it's already a dict of {name: array}.
        if isinstance(data_item, dict):
            return data_item
        return {input_name: data_item}

    calibration_dataset = nncf.Dataset(calibration_samples, transform_fn)

    quantized_model = nncf.quantize(
        fp32_model,
        calibration_dataset,
        subset_size=min(subset_size, len(calibration_samples)),
        preset=nncf.QuantizationPreset.MIXED,
    )

    os.makedirs(output_dir, exist_ok=True)
    xml_path = os.path.join(output_dir, f"{model_name}_int8.xml")
    ov.save_model(quantized_model, xml_path)
    print(f"[NNCF] INT8 IR saved -> {xml_path}")
    return xml_path


# --------------------------------------------------------------------------
# Convenience: build calibration data for each model type
# --------------------------------------------------------------------------
def build_gru_calibration_data(sequences: np.ndarray, n_samples=config.NNCF_CALIBRATION_SAMPLES):
    n_samples = min(n_samples, len(sequences))
    idx = np.random.RandomState(config.SEED).choice(len(sequences), n_samples, replace=False)
    return [sequences[i:i + 1].astype(np.int64) for i in idx]


def build_roberta_calibration_data(texts, tokenizer, n_samples=config.NNCF_CALIBRATION_SAMPLES,
                                    max_len=config.MAX_SEQ_LEN):
    n_samples = min(n_samples, len(texts))
    idx = np.random.RandomState(config.SEED).choice(len(texts), n_samples, replace=False)
    samples = []
    for i in idx:
        enc = tokenizer(
            texts[i], return_tensors="np", truncation=True, max_length=max_len, padding="max_length"
        )
        samples.append({
            "input_ids": enc["input_ids"].astype(np.int64),
            "attention_mask": enc["attention_mask"].astype(np.int64),
        })
    return samples
