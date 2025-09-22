import sys
import onnx
import torch
import numpy as np

from pyinfinitensor.onnx import OnnxStub, backend
import onnxruntime
from transformers import AutoTokenizer, AutoModel
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

# ==== 加载模型与分词器 ====
MODEL_PATH = "/home/zenghua/BGE/bge_sim_512.onnx"
onnx_model = onnx.load(MODEL_PATH)
model = OnnxStub(onnx_model, backend.cuda_runtime())
session = onnxruntime.InferenceSession(MODEL_PATH)
tokenizer = AutoTokenizer.from_pretrained("/home/zenghua/.cache/modelscope/hub/models/BAAI/bge-large-zh/")

app = FastAPI()


@app.post("/infer")
async def infer(request: Request):
    """
    接收一段中文文本，先encode，再送入OnnxStub模型推理
    """
    try:
        body = await request.json()
        text = body.get("text", "")
        if not text:
            return JSONResponse({"error": "Missing 'text' field"}, status_code=400)

        encoded = tokenizer(
            text,
            return_tensors="np",  # 返回 numpy 数组（>=transformers 4.46 支持）
            padding="max_length",
            truncation=True,
            max_length=512,
        )

        ort_inputs = {
            "input_ids": encoded["input_ids"].astype(np.int64),
            "attention_mask": encoded["attention_mask"].astype(np.int64),
        }
        # 输入拷贝到 InfiniTensor 张量
        for name, itensor in model.inputs.items():
            if name not in ort_inputs:
                raise RuntimeError(f"Input {name} not found")
            itensor.copyin_numpy(ort_inputs[name])

        # 推理
        raw_outputs = session.run(None, ort_inputs)
        model.run()

        # 拷贝输出结果
        outputs_if = [output.copyout_numpy() for output in model.outputs.values()]
        # output_names = [o.name for o in session.get_outputs()]
        # outputs = {name: out.shape for name, out in zip(output_names, raw_outputs)}
        # 对比
        compare_results = []
        all_pass = True
        for i, (ort_out, if_out) in enumerate(zip(raw_outputs, outputs_if)):
            ort_out = np.array(ort_out)
            if_out = np.array(if_out)

            same_shape = ort_out.shape == if_out.shape
            max_abs_err = float(np.max(np.abs(ort_out - if_out)))
            mean_abs_err = float(np.mean(np.abs(ort_out - if_out)))
            preview_ort = ort_out.flatten()[:5].tolist()
            preview_if = if_out.flatten()[:5].tolist()

            try:
                np.testing.assert_allclose(ort_out, if_out, rtol=1.0, atol=1e-5)
                passed = True
            except AssertionError:
                passed = False
                all_pass = False

            compare_results.append(
                {
                    "index": i,
                    "shape_ort": list(ort_out.shape),
                    "shape_if": list(if_out.shape),
                    "shape_match": same_shape,
                    "max_abs_err": max_abs_err,
                    "mean_abs_err": mean_abs_err,
                    "preview_ort": preview_ort,
                    "preview_if": preview_if,
                    "passed": passed,
                }
            )

        return JSONResponse({"all_passed": all_pass, "details": compare_results})

    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)
