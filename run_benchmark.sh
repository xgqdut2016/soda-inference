#!/bin/bash

# Benchmark script for evaluating inference latency
# This script provides easy commands to run different benchmark scenarios

echo "🚀 Soda Inference Latency Benchmark"
echo "=================================="

# Check if benchmark script exists
if [ ! -f "benchmark_latency.py" ]; then
    echo "❌ benchmark_latency.py not found!"
    exit 1
fi

# Function to run benchmark with different options
run_benchmark() {
    local description="$1"
    local args="$2"

    echo ""
    echo "📊 $description"
    echo "Command: python benchmark_latency.py $args"
    echo "----------------------------------------"

    python benchmark_latency.py $args
}

# Default benchmark (all models)
echo "Running default benchmark (all models)..."
run_benchmark "Full Benchmark - All Models" "--config benchmark_config.json --iterations 50"

# # Embedding models only
echo ""
echo "Running embedding models benchmark..."
# run_benchmark "Embedding Models Only" "--config benchmark_config.json --embedding-only --iterations 50"

# # Reranker models only
# echo ""
# echo "Running reranker models benchmark..."
# run_benchmark "Reranker Models Only" "--config benchmark_config.json --reranker-only --iterations 50"

# Quick benchmark (fewer iterations)
echo ""
echo "Running quick benchmark..."
# run_benchmark "Quick Benchmark" "--config benchmark_config_queue.json --iterations 10 --reranker-only --warmup 2"

echo ""
echo "✅ All benchmarks completed!"
echo "Check the generated benchmark_results_*.json files for detailed results."
