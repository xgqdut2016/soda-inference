#!/usr/bin/env python3
"""
Benchmark script to evaluate latency per request for different inference modules.

This script benchmarks:
1. BgeM3Infer - BGE-M3 embedding model
2. InfinitensorInfer - InfiniTensor ONNX embedding model
3. BgeRerankerInfer - BGE reranker model
4. InfinitensorRerankerInfer - InfiniTensor ONNX reranker model

Usage:
    python benchmark_latency.py [--config config.json] [--output results.json] [--warmup 5] [--iterations 100]
"""

import sys
import json
import time
import argparse
import statistics
from pathlib import Path
from typing import Dict, List, Any, Tuple, Optional
import numpy as np
from loguru import logger
import traceback

# Add src to path for imports
sys.path.append('src')

from modules.bge_m3_infer import BgeM3Infer
from modules.infinitensor_infer import InfinitensorInfer
from modules.bge_reranker_infer import BgeRerankerInfer
from modules.infinitensor_reranker_infer import InfinitensorRerankerInfer

from datasets import load_dataset
import re

# ===== 定义 HTML 清理函数 =====
def clean_html(text):
    text = re.sub(r"<[^>]+>", "", text)  # 去除所有 HTML 标签
    text = text.replace("\n", "").strip()
    return text

class LatencyBenchmark:
    """Benchmark class for measuring inference latency."""

    def __init__(self, config: Dict[str, Any], dataset_path):
        """Initialize benchmark with configuration."""
        self.config = config
        self.results = {}
        # dataset_path = "/home/xiaogq/verify_models/mteb_code/T2Retrieval"
        dataset = load_dataset(dataset_path)
        # ===== 提取 queries 和 corpus 文本 =====
        queries = [item["text"].strip() for item in dataset["queries"]]
        corpus = [clean_html(item["text"]) for item in dataset["corpus"]]
        num_samples = 16
        # Test data
        # self.test_queries = [
        #     '我最低选多少课',
        #     '离东南门最近的健身房在哪？',
        #     '军训期间的4学分算在大一本科生26学分的范围内吗',
        #     'GPA计算原则',
        #     '限制选择的26学分包括军训的4学分吗？',
        #     '这26学分是一个学期限制的吗？',
        #     '军事技能考核总分多少分能得到两学分',
        #     '军训时的4学分会算在大一学期的26学分限制中吗？即第一学期还可以选多少学分的课？',
        #     'What is the weather like today?',
        #     'How to implement a neural network?',
        #     'Machine learning best practices',
        #     'Python programming tips',
        #     'Data science workflow',
        #     'Deep learning architectures',
        #     'Natural language processing techniques',
        #     'Computer vision applications'
        # ]

        # self.test_documents = [
        #     '最低选课数目是4门',
        #     '东南门最近的健身房是xx健身房',
        #     '军训期间的4学分算在大一本科生26学分的范围内',
        #     'GPA计算原则是加权平均',
        #     '限制选择的26学分包括军训的4学分',
        #     '26学分是一个学期限制的',
        #     '军事技能考核总分100分能得到两学分',
        #     '军训时的4学分会算在大一学期的26学分限制中',
        #     'Today is sunny with a temperature of 25 degrees',
        #     'Neural networks are computational models inspired by biological neural networks',
        #     'Machine learning involves training algorithms on data to make predictions',
        #     'Python is a versatile programming language with many libraries',
        #     'Data science combines statistics, programming, and domain expertise',
        #     'Deep learning uses neural networks with multiple layers',
        #     'NLP focuses on interactions between computers and human language',
        #     'Computer vision enables machines to interpret visual information'
        # ]
        self.test_queries = queries[:num_samples]
        self.test_documents = corpus[:num_samples]
        # Create query-document pairs for reranking
        self.test_pairs = list(zip(self.test_queries, self.test_documents))

    def measure_embedding_latency(self, model_class, model_name: str, model_config: Dict[str, Any]) -> Dict[str, Any]:
        """Measure latency for embedding models."""
        logger.info(f"Benchmarking {model_name}...")

        try:
            # Initialize model
            model = model_class(**model_config)

            # Warmup runs
            warmup_runs = self.config.get('warmup_runs', 5)
            logger.info(f"Running {warmup_runs} warmup iterations...")
            for _ in range(warmup_runs):
                model.encode(self.test_queries[:2])  # Use subset for warmup

            # Actual benchmark
            iterations = self.config.get('iterations', 100)
            latencies = []

            logger.info(f"Running {iterations} benchmark iterations...")
            for i in range(iterations):
                start_time = time.perf_counter()
                results = model.encode(self.test_queries)
                end_time = time.perf_counter()

                latency = (end_time - start_time) * 1000  # Convert to milliseconds
                latencies.append(latency)

                if (i + 1) % 20 == 0:
                    logger.info(f"Completed {i + 1}/{iterations} iterations")

            # Calculate statistics
            stats = self._calculate_stats(latencies)
            stats['model_name'] = model_name
            stats['model_class'] = model_class.__name__
            stats['total_queries'] = len(self.test_queries)
            stats['iterations'] = iterations

            logger.info(f"✅ {model_name} benchmark completed")
            return stats

        except Exception as e:
            logger.error(f"❌ Error benchmarking {model_name}: {e}")
            logger.error(traceback.format_exc())
            return {
                'model_name': model_name,
                'model_class': model_class.__name__,
                'error': str(e),
                'status': 'failed'
            }

    def measure_reranker_latency(self, model_class, model_name: str, model_config: Dict[str, Any]) -> Dict[str, Any]:
        """Measure latency for reranker models."""
        logger.info(f"Benchmarking {model_name}...")

        try:
            # Initialize model
            model = model_class(**model_config)

            # Warmup runs
            warmup_runs = self.config.get('warmup_runs', 5)
            logger.info(f"Running {warmup_runs} warmup iterations...")
            for _ in range(warmup_runs):
                model.rerank(self.test_pairs[:2])  # Use subset for warmup

            # Actual benchmark
            iterations = self.config.get('iterations', 100)
            latencies = []

            logger.info(f"Running {iterations} benchmark iterations...")
            for i in range(iterations):
                start_time = time.perf_counter()
                results = model.rerank(self.test_pairs)
                end_time = time.perf_counter()

                latency = (end_time - start_time) * 1000  # Convert to milliseconds
                latencies.append(latency)

                if (i + 1) % 20 == 0:
                    logger.info(f"Completed {i + 1}/{iterations} iterations")

            # Calculate statistics
            stats = self._calculate_stats(latencies)
            stats['model_name'] = model_name
            stats['model_class'] = model_class.__name__
            stats['total_pairs'] = len(self.test_pairs)
            stats['iterations'] = iterations

            logger.info(f"✅ {model_name} benchmark completed")
            return stats

        except Exception as e:
            logger.error(f"❌ Error benchmarking {model_name}: {e}")
            logger.error(traceback.format_exc())
            return {
                'model_name': model_name,
                'model_class': model_class.__name__,
                'error': str(e),
                'status': 'failed'
            }

    def _calculate_stats(self, latencies: List[float]) -> Dict[str, Any]:
        """Calculate latency statistics."""
        if not latencies:
            return {'error': 'No latency data'}

        return {
            'mean_ms': float(statistics.mean(latencies)),
            'median_ms': float(statistics.median(latencies)),
            'std_ms': float(statistics.stdev(latencies) if len(latencies) > 1 else 0),
            'min_ms': float(min(latencies)),
            'max_ms': float(max(latencies)),
            'p95_ms': float(np.percentile(latencies, 95)),
            'p99_ms': float(np.percentile(latencies, 99)),
            'qps': 1000.0 / statistics.mean(latencies),  # Queries per second
            'latencies': latencies  # Include raw data for detailed analysis
        }

    def run_benchmark(self) -> Dict[str, Any]:
        """Run complete benchmark suite."""
        logger.info("🚀 Starting latency benchmark...")

        results = {
            'config': self.config,
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
            'embedding_models': {},
            'reranker_models': {}
        }

        # Benchmark embedding models
        embedding_configs = self.config.get('embedding_models', {})
        for model_name, model_config in embedding_configs.items():
            if model_name == 'BgeM3Infer':
                stats = self.measure_embedding_latency(BgeM3Infer, model_name, model_config)
                results['embedding_models'][model_name] = stats
            elif model_name in ['InfinitensorInfer', 'InfinitensorInferPerRequest']:
                stats = self.measure_embedding_latency(InfinitensorInfer, model_name, model_config)
                results['embedding_models'][model_name] = stats

        # Benchmark reranker models
        reranker_configs = self.config.get('reranker_models', {})
        for model_name, model_config in reranker_configs.items():
            if model_name == 'BgeRerankerInfer':
                stats = self.measure_reranker_latency(BgeRerankerInfer, model_name, model_config)
                results['reranker_models'][model_name] = stats
            elif model_name in ['InfinitensorRerankerInfer', 'InfinitensorRerankerInferPerRequest']:
                stats = self.measure_reranker_latency(InfinitensorRerankerInfer, model_name, model_config)
                results['reranker_models'][model_name] = stats

        logger.info("✅ Benchmark completed!")
        return results

    def print_summary(self, results: Dict[str, Any]):
        """Print benchmark summary."""
        print("\n" + "="*80)
        print("LATENCY BENCHMARK RESULTS")
        print("="*80)
        print(f"Timestamp: {results['timestamp']}")
        print(f"Warmup runs: {results['config'].get('warmup_runs', 5)}")
        print(f"Benchmark iterations: {results['config'].get('iterations', 100)}")

        # Embedding models summary
        if results['embedding_models']:
            print("\n📊 EMBEDDING MODELS")
            print("-" * 50)
            for model_name, stats in results['embedding_models'].items():
                if 'error' in stats:
                    print(f"❌ {model_name}: {stats['error']}")
                else:
                    print(f"✅ {model_name}:")
                    print(f"   Mean latency: {stats['mean_ms']:.2f} ms")
                    print(f"   Median latency: {stats['median_ms']:.2f} ms")
                    print(f"   P95 latency: {stats['p95_ms']:.2f} ms")
                    print(f"   P99 latency: {stats['p99_ms']:.2f} ms")
                    print(f"   QPS: {stats['qps']:.2f}")
                    print(f"   Std dev: {stats['std_ms']:.2f} ms")

        # Reranker models summary
        if results['reranker_models']:
            print("\n📊 RERANKER MODELS")
            print("-" * 50)
            for model_name, stats in results['reranker_models'].items():
                if 'error' in stats:
                    print(f"❌ {model_name}: {stats['error']}")
                else:
                    print(f"✅ {model_name}:")
                    print(f"   Mean latency: {stats['mean_ms']:.2f} ms")
                    print(f"   Median latency: {stats['median_ms']:.2f} ms")
                    print(f"   P95 latency: {stats['p95_ms']:.2f} ms")
                    print(f"   P99 latency: {stats['p99_ms']:.2f} ms")
                    print(f"   QPS: {stats['qps']:.2f}")
                    print(f"   Std dev: {stats['std_ms']:.2f} ms")

        # Performance comparison
        print("\n🏆 PERFORMANCE COMPARISON")
        print("-" * 50)

        # Compare embedding models
        embedding_models = [(name, stats) for name, stats in results['embedding_models'].items()
                           if 'error' not in stats]
        if len(embedding_models) > 1:
            embedding_models.sort(key=lambda x: x[1]['mean_ms'])
            print("Embedding Models (sorted by latency):")
            for i, (name, stats) in enumerate(embedding_models, 1):
                print(f"  {i}. {name}: {stats['mean_ms']:.2f} ms (QPS: {stats['qps']:.2f})")

        # Compare reranker models
        reranker_models = [(name, stats) for name, stats in results['reranker_models'].items()
                          if 'error' not in stats]
        if len(reranker_models) > 1:
            reranker_models.sort(key=lambda x: x[1]['mean_ms'])
            print("Reranker Models (sorted by latency):")
            for i, (name, stats) in enumerate(reranker_models, 1):
                print(f"  {i}. {name}: {stats['mean_ms']:.2f} ms (QPS: {stats['qps']:.2f})")

        print("="*80)


def load_config(config_path: str) -> Dict[str, Any]:
    """Load configuration from JSON file."""
    try:
        with open(config_path, 'r') as f:
            return json.load(f)
    except Exception as e:
        logger.error(f"Failed to load config from {config_path}: {e}")
        return {}


def create_default_config() -> Dict[str, Any]:
    """Create default configuration."""
    return {
        "warmup_runs": 5,
        "iterations": 100,
        "embedding_models": {
            "BgeM3Infer": {
                "model_path": "/home/zenghua/repos/soda-inference/bge-m3",
                "use_fp16": False,
                "device": "cuda",
                "batch_size": 12,
                "max_length": 8192
            },
            "InfinitensorInfer": {
                "model_path": "/home/zenghua/BGE/bge_sim_512.onnx",
                "tokenizer_path": "/home/zenghua/repos/soda-inference/bge-m3",
                "device": "cuda",
                "batch_size": 12,
                "max_length": 512
            }
        },
        "reranker_models": {
            "BgeRerankerInfer": {
                "model_path": "/home/zenghua/repos/soda-inference/bge-reranker-v2-m3",
                "use_fp16": False,
                "device": "cuda",
                "batch_size": 256,
                "max_length": 512
            },
            "InfinitensorRerankerInfer": {
                "onnx_model_path": "/home/zenghua/BGE-reranker-512/bge_reranker_O1_sim_512.onnx",
                "model_path": "/home/zenghua/repos/soda-inference/bge-reranker-v2-m3",
                "device": "cuda",
                "batch_size": 256,
                "max_length": 512
            }
        }
    }


def main():
    parser = argparse.ArgumentParser(description='Benchmark inference latency for different models')
    parser.add_argument('--config', help='Path to configuration JSON file')
    parser.add_argument('--output', help='Output file for results (JSON format)')
    parser.add_argument('--warmup', type=int, default=5, help='Number of warmup runs (default: 5)')
    parser.add_argument('--iterations', type=int, default=100, help='Number of benchmark iterations (default: 100)')
    parser.add_argument('--verbose', '-v', action='store_true', help='Enable verbose logging')
    parser.add_argument('--embedding-only', action='store_true', help='Benchmark only embedding models')
    parser.add_argument('--reranker-only', action='store_true', help='Benchmark only reranker models')
    parser.add_argument('--dataset_path', type=str, help='dataset path')

    args = parser.parse_args()

    # Configure logging
    if args.verbose:
        logger.remove()
        logger.add(sys.stderr, level="DEBUG")
    else:
        logger.remove()
        logger.add(sys.stderr, level="INFO")

    # Load or create configuration
    if args.config:
        config = load_config(args.config)
    else:
        config = create_default_config()
        logger.info("Using default configuration. Use --config to specify custom config.")

    # Override config with command line arguments
    config['warmup_runs'] = args.warmup
    config['iterations'] = args.iterations

    # Filter models based on command line arguments
    if args.embedding_only:
        config['reranker_models'] = {}
    elif args.reranker_only:
        config['embedding_models'] = {}

    # Run benchmark
    dataset_path = args.dataset_path
    benchmark = LatencyBenchmark(config, dataset_path)
    results = benchmark.run_benchmark()

    # Print summary
    benchmark.print_summary(results)

    # Save results
    if args.output:
        with open(args.output, 'w') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        logger.info(f"Results saved to {args.output}")
    else:
        # Save to default file
        timestamp = time.strftime('%Y%m%d_%H%M%S')
        default_output = f"benchmark_results_{timestamp}.json"
        with open(default_output, 'w') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        logger.info(f"Results saved to {default_output}")


if __name__ == '__main__':
    main()
