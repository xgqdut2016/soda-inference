# Soda Inference Service

A high-performance inference service for embedding and reranking models, supporting both PyTorch and ONNX backends.

## 🚀 Quick Start

### Prerequisites

- Python 3.8+
- Conda environment (recommended)
- Required model files (see model setup below)

### Environment Setup

```bash
# Clone the repository
git clone <repository-url>
cd soda-inference

# Create and activate conda environment
conda create -n infinitensor-env python=3.8
conda activate infinitensor-env

# Install dependencies
pip install -r requirements.txt
```

## 📋 Test Pipeline

Follow this step-by-step pipeline to test the inference services:

### Step 1: Launch Services

#### 1.1 Launch Embedding Service

Choose one of the following embedding service configurations:

**BGE-M3 PyTorch (Port 10991):**
```bash
SERVICE_PORT=10991 MODEL_TYPE=embed MODEL_CLS=BgeM3Infer MODEL_KWARGS='{"model_path": "/home/zenghua/repos/soda-inference/bge-m3","use_fp16":false}' python src/start_server.py
```

**BGE-M3 Infinitensor (Port 10992):**
```bash
SERVICE_PORT=10992 MODEL_TYPE=embed MODEL_CLS=InfinitensorInfer MODEL_KWARGS='{"model_path": "/home/zenghua/BGE/bge_sim_512.onnx","tokenizer_path": "/home/zenghua/repos/soda-inference/bge-m3"}' python src/start_server.py
```

**BGE-M3 ONNX (Port 10993):**
```bash
SERVICE_PORT=10993 MODEL_TYPE=embed MODEL_CLS=BgeM3OnnxInfer MODEL_KWARGS='{"tokenizer_path":"/home/zenghua/repos/soda-inference/bge_m3_onnx/tokenizer","model_onnx_path":"/home/zenghua/repos/soda-inference/bge_m3_onnx/onnx_model/bge_m3_fp16_dense_sparse_optimized.onnx"}' python src/start_server.py
```

#### 1.2 Launch Reranking Service

Choose one of the following reranking service configurations:

**BGE Reranker PyTorch (Port 10993):**
```bash
SERVICE_PORT=10993 MODEL_TYPE=rerank MODEL_CLS=BgeRerankerInfer MODEL_KWARGS='{"model_path":"/home/zenghua/repos/soda-inference/bge-reranker-v2-m3","use_fp16":false}' python src/start_server.py
```

**BGE Reranker Infinitensor (Port 10994):**
```bash
SERVICE_PORT=10994 MODEL_TYPE=rerank MODEL_CLS=InfinitensorRerankerInfer MODEL_KWARGS='{"model_path":"/home/zenghua/BGE-reranker-512/bge_reranker_O1_sim_512.onnx","tokenizer_path":"/home/zenghua/repos/soda-inference/bge-reranker-v2-m3"}' python src/start_server.py
```

**BGE Reranker ONNX (Port 10996):**
```bash
SERVICE_PORT=10996 MODEL_TYPE=rerank MODEL_CLS=OptimumOnnxReranker MODEL_KWARGS='{"model_path":"/home/zenghua/repos/soda-inference/rerank-bge_v2_m3-onnx_cuda_o4"}' python src/start_server.py
```

### Step 2: Test Individual Services

#### 2.1 Test Embedding Service

Test the embedding service using `test/test_embed.py`:

```bash
# Test single embedding service
python test/test_embed.py

# Test multiple services (modify URLs in the script)
# The script will test both port 10991 and 10992 by default
```

**Expected Output:**
```json
{
  "dense_embed": [0.1, 0.2, ...],
  "sparse_embed": {"1": 0.5, "2": 0.62, ...}
}
```

#### 2.2 Test Reranking Service

Test the reranking service using `test/test_rerank.py`:

```bash
# Test reranking service
python test/test_rerank.py

# Modify the URL in the script to test different ports
# Default tests port 10994
```

**Expected Output:**
```json
[0.8, 0.3, 0.9, ...]  # Reranking scores for each document
```

### Step 3: End-to-End Testing

Run the comprehensive E2E test using `test/test_e2e.py`:

```bash
# Run E2E test (requires both embedding and reranking services)
python test/test_e2e.py
```

**What it does:**
- Tests embedding service on port 10992
- Tests reranking service on port 10993
- Processes 8 query-document pairs
- Generates `embedding.pkl` and `reranker.pkl` files
- Validates the complete pipeline

**Expected Output:**
```
Processing query-document pairs...
Generated embedding.pkl and reranker.pkl files
```

## 🔧 Service Configuration

### Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `SERVICE_PORT` | Port for the service | 9991 |
| `MODEL_TYPE` | Type of model (`embed` or `rerank`) | - |
| `MODEL_CLS` | Model class to use | - |
| `MODEL_KWARGS` | JSON string with model parameters | `{}` |
| `BATCH_SIZE` | Batch size for processing | 12 |
| `QUEUE_SIZE` | Queue size for requests | 1024 |
| `REST_TIMEOUT` | REST API timeout | 15.0 |

### Available Model Classes

#### Embedding Models
- `BgeM3Infer` - BGE-M3 PyTorch implementation
- `InfinitensorInfer` - Infinitensor ONNX implementation
- `BgeM3OnnxInfer` - BGE-M3 ONNX implementation
- `FastembedDense` - FastEmbed dense embeddings
- `FastembedSparse` - FastEmbed sparse embeddings

#### Reranking Models
- `BgeRerankerInfer` - BGE Reranker PyTorch implementation
- `InfinitensorRerankerInfer` - Infinitensor Reranker ONNX implementation
- `OptimumOnnxReranker` - Optimum ONNX Reranker
- `BceRerankerInfer` - BCE Reranker PyTorch implementation

## 📊 API Endpoints

### Embedding Service

**Endpoint:** `POST /embed`

**Request:**
```json
{
  "model": "default",
  "text": "your text here",
  "return_dense": true,
  "return_sparse": true
}
```

**Response:**
```json
{
  "dense_embed": [0.1, 0.2, ...],
  "sparse_embed": {"1": 0.5, "2": 0.62, ...}
}
```

### Reranking Service

**Endpoint:** `POST /rerank`

**Request:**
```json
{
  "model": "default",
  "query": "your query",
  "texts": ["document1", "document2", ...]
}
```

**Response:**
```json
[0.8, 0.3, 0.9, ...]
```

### Evaluate Generated Results

Compare generated pickle files with reference data:

```bash
# Evaluate all results
python eval_generated_pkl.py

# Evaluate only embeddings
python eval_generated_pkl.py --embedding-only

# Evaluate only reranker
python eval_generated_pkl.py --reranker-only

# Generate report
python eval_generated_pkl.py --output evaluation_report.txt
```
