#!/usr/bin/env python3
"""
Evaluation script to compare generated pickle files from test_e2e.py with evaluation data.

This script compares:
1. embedding.pkl - BGE-M3 embeddings (dense and sparse)
2. reranker.pkl - SodaRerank scores

Usage:
    python eval_generated_pkl.py [--generated-dir .] [--eval-dir eval] [--output report.txt]
"""

import sys
import pickle
import argparse
import numpy as np
from pathlib import Path
from typing import Dict, List, Any, Tuple
import json
from loguru import logger

def load_pickle_file(file_path: Path) -> Any:
    """Load pickle file and return its contents."""
    try:
        with open(file_path, 'rb') as f:
            return pickle.load(f)
    except Exception as e:
        logger.error(f"Failed to load {file_path}: {e}")
        return None

def compare_dense_embeddings(generated: List[float], reference: List[float]) -> Dict[str, float]:
    """Compare dense embeddings using various metrics."""
    gen_array = np.array(generated)
    ref_array = np.array(reference)

    # Cosine similarity
    cos_sim = np.dot(gen_array, ref_array) / (np.linalg.norm(gen_array) * np.linalg.norm(ref_array))

    # Euclidean distance
    euclidean_dist = np.linalg.norm(gen_array - ref_array)

    # L2 norm difference
    l2_norm_diff = abs(np.linalg.norm(gen_array) - np.linalg.norm(ref_array))

    # Mean squared error
    mse = np.mean((gen_array - ref_array) ** 2)

    # Mean absolute error
    mae = np.mean(np.abs(gen_array - ref_array))

    return {
        'cosine_similarity': float(cos_sim),
        'euclidean_distance': float(euclidean_dist),
        'l2_norm_difference': float(l2_norm_diff),
        'mse': float(mse),
        'mae': float(mae)
    }

def compare_sparse_embeddings(generated: Dict[int, float], reference: Dict[int, float]) -> Dict[str, float]:
    """Compare sparse embeddings."""
    gen_keys = set(generated.keys())
    ref_keys = set(reference.keys())

    # Jaccard similarity of keys
    intersection = gen_keys.intersection(ref_keys)
    union = gen_keys.union(ref_keys)
    jaccard_similarity = len(intersection) / len(union) if union else 0.0

    # Compare values for common keys
    common_keys = intersection
    if common_keys:
        gen_values = [generated[k] for k in common_keys]
        ref_values = [reference[k] for k in common_keys]

        value_mse = np.mean([(g - r) ** 2 for g, r in zip(gen_values, ref_values)])
        value_mae = np.mean([abs(g - r) for g, r in zip(gen_values, ref_values)])
    else:
        value_mse = float('inf')
        value_mae = float('inf')

    return {
        'jaccard_similarity': jaccard_similarity,
        'common_keys_count': len(common_keys),
        'generated_keys_count': len(gen_keys),
        'reference_keys_count': len(ref_keys),
        'value_mse': float(value_mse),
        'value_mae': float(value_mae)
    }

def compare_embeddings(generated_data: List[Dict], reference_data: List[Dict]) -> Dict[str, Any]:
    """Compare embedding data between generated and reference."""
    if len(generated_data) != len(reference_data):
        logger.warning(f"Length mismatch: generated={len(generated_data)}, reference={len(reference_data)}")

    results = {
        'total_samples': min(len(generated_data), len(reference_data)),
        'dense_metrics': [],
        'sparse_metrics': [],
        'summary': {}
    }

    for i, (gen_item, ref_item) in enumerate(zip(generated_data, reference_data)):
        logger.info(f"Comparing embedding sample {i+1}/{results['total_samples']}")

        # Compare dense embeddings
        if 'dense_embed' in gen_item and 'dense_embed' in ref_item:
            dense_metrics = compare_dense_embeddings(gen_item['dense_embed'], ref_item['dense_embed'])
            results['dense_metrics'].append(dense_metrics)
        else:
            logger.warning(f"Missing dense_embed in sample {i}")

        # Compare sparse embeddings
        if 'sparse_embed' in gen_item and 'sparse_embed' in ref_item:
            sparse_metrics = compare_sparse_embeddings(gen_item['sparse_embed'], ref_item['sparse_embed'])
            results['sparse_metrics'].append(sparse_metrics)
        else:
            logger.warning(f"Missing sparse_embed in sample {i}")

    # Calculate summary statistics
    if results['dense_metrics']:
        dense_summary = {}
        for metric in ['cosine_similarity', 'euclidean_distance', 'l2_norm_difference', 'mse', 'mae']:
            values = [m[metric] for m in results['dense_metrics']]
            dense_summary[metric] = {
                'mean': float(np.mean(values)),
                'std': float(np.std(values)),
                'min': float(np.min(values)),
                'max': float(np.max(values))
            }
        results['summary']['dense'] = dense_summary

    if results['sparse_metrics']:
        sparse_summary = {}
        for metric in ['jaccard_similarity', 'value_mse', 'value_mae']:
            values = [m[metric] for m in results['sparse_metrics']]
            sparse_summary[metric] = {
                'mean': float(np.mean(values)),
                'std': float(np.std(values)),
                'min': float(np.min(values)),
                'max': float(np.max(values))
            }
        results['summary']['sparse'] = sparse_summary

    return results

def compare_reranker_scores(generated_data: List[List[float]], reference_data: List[List[float]]) -> Dict[str, Any]:
    """Compare reranker scores between generated and reference."""
    if len(generated_data) != len(reference_data):
        logger.warning(f"Length mismatch: generated={len(generated_data)}, reference={len(reference_data)}")

    results = {
        'total_samples': min(len(generated_data), len(reference_data)),
        'sample_metrics': [],
        'summary': {}
    }

    for i, (gen_scores, ref_scores) in enumerate(zip(generated_data, reference_data)):
        logger.info(f"Comparing reranker sample {i+1}/{results['total_samples']}")

        if len(gen_scores) != len(ref_scores):
            logger.warning(f"Score length mismatch in sample {i}: gen={len(gen_scores)}, ref={len(ref_scores)}")

        # Compare scores
        min_len = min(len(gen_scores), len(ref_scores))
        if min_len > 0:
            gen_array = np.array(gen_scores[:min_len])
            ref_array = np.array(ref_scores[:min_len])

            mse = np.mean((gen_array - ref_array) ** 2)
            mae = np.mean(np.abs(gen_array - ref_array))
            correlation = np.corrcoef(gen_array, ref_array)[0, 1] if len(gen_array) > 1 else 1.0

            results['sample_metrics'].append({
                'mse': float(mse),
                'mae': float(mae),
                'correlation': float(correlation),
                'score_count': min_len
            })

    # Calculate summary statistics
    if results['sample_metrics']:
        summary = {}
        for metric in ['mse', 'mae', 'correlation']:
            values = [m[metric] for m in results['sample_metrics']]
            summary[metric] = {
                'mean': float(np.mean(values)),
                'std': float(np.std(values)),
                'min': float(np.min(values)),
                'max': float(np.max(values))
            }
        results['summary'] = summary

    return results

def print_summary_report(embedding_results: Dict, reranker_results: Dict, output_file: str = None):
    """Print a comprehensive summary report."""
    report_lines = []

    def add_line(text: str):
        report_lines.append(text)
        print(text)

    add_line("=" * 80)
    add_line("EVALUATION REPORT: Generated vs Reference Pickle Files")
    add_line("=" * 80)

    # Embedding results
    if embedding_results is not None:
        add_line("\n📊 EMBEDDING COMPARISON RESULTS")
        add_line("-" * 50)
        add_line(f"Total samples compared: {embedding_results['total_samples']}")

        if 'dense' in embedding_results['summary']:
            add_line("\n🔹 Dense Embeddings:")
            dense_summary = embedding_results['summary']['dense']
            add_line(f"  Cosine Similarity: {dense_summary['cosine_similarity']['mean']:.4f} ± {dense_summary['cosine_similarity']['std']:.4f}")
            add_line(f"  Euclidean Distance: {dense_summary['euclidean_distance']['mean']:.4f} ± {dense_summary['euclidean_distance']['std']:.4f}")
            add_line(f"  MSE: {dense_summary['mse']['mean']:.6f} ± {dense_summary['mse']['std']:.6f}")
            add_line(f"  MAE: {dense_summary['mae']['mean']:.6f} ± {dense_summary['mae']['std']:.6f}")

        if 'sparse' in embedding_results['summary']:
            add_line("\n🔹 Sparse Embeddings:")
            sparse_summary = embedding_results['summary']['sparse']
            add_line(f"  Jaccard Similarity: {sparse_summary['jaccard_similarity']['mean']:.4f} ± {sparse_summary['jaccard_similarity']['std']:.4f}")
            add_line(f"  Value MSE: {sparse_summary['value_mse']['mean']:.6f} ± {sparse_summary['value_mse']['std']:.6f}")
            add_line(f"  Value MAE: {sparse_summary['value_mae']['mean']:.6f} ± {sparse_summary['value_mae']['std']:.6f}")

    # Reranker results
    if reranker_results is not None:
        add_line("\n📊 RERANKER COMPARISON RESULTS")
        add_line("-" * 50)
        add_line(f"Total samples compared: {reranker_results['total_samples']}")

        if reranker_results['summary']:
            add_line("\n🔹 Reranker Scores:")
            rerank_summary = reranker_results['summary']
            add_line(f"  MSE: {rerank_summary['mse']['mean']:.6f} ± {rerank_summary['mse']['std']:.6f}")
            add_line(f"  MAE: {rerank_summary['mae']['mean']:.6f} ± {rerank_summary['mae']['std']:.6f}")
            add_line(f"  Correlation: {rerank_summary['correlation']['mean']:.4f} ± {rerank_summary['correlation']['std']:.4f}")

    # Overall assessment
    add_line("\n🎯 OVERALL ASSESSMENT")
    add_line("-" * 50)

    # Check if results are good
    embedding_good = True
    reranker_good = True

    if embedding_results is not None and 'dense' in embedding_results['summary']:
        cos_sim_mean = embedding_results['summary']['dense']['cosine_similarity']['mean']
        if cos_sim_mean < 0.9:  # Threshold for good similarity
            embedding_good = False
    elif embedding_results is None:
        embedding_good = None  # Not evaluated

    if reranker_results is not None and reranker_results['summary']:
        corr_mean = reranker_results['summary']['correlation']['mean']
        if corr_mean < 0.8:  # Threshold for good correlation
            reranker_good = False
    elif reranker_results is None:
        reranker_good = None  # Not evaluated

    # Determine overall assessment
    if embedding_good is True and reranker_good is True:
        add_line(f"✅ Results look GOOD - Generated files closely match reference data (Cosine Similarity: {cos_sim_mean:.4f}, Reranker Correlation: {corr_mean:.4f})")
    elif embedding_good is True and reranker_good is None:
        add_line(f"✅ Embedding results look GOOD - Generated embeddings closely match reference data (Cosine Similarity: {cos_sim_mean:.4f})")
    elif embedding_good is None and reranker_good is True:
        add_line(f"✅ Reranker results look GOOD - Generated reranker scores closely match reference data (Correlation: {corr_mean:.4f})")
    elif embedding_good is False and reranker_good is None:
        add_line(f"❌ Embedding results need ATTENTION - Significant differences detected in embeddings (Cosine Similarity: {cos_sim_mean:.4f} < 0.9)")
    elif embedding_good is None and reranker_good is False:
        add_line(f"❌ Reranker results need ATTENTION - Significant differences detected in reranker scores (Correlation: {corr_mean:.4f} < 0.8)")
    elif embedding_good is False and reranker_good is False:
        add_line(f"❌ Results need ATTENTION - Significant differences detected in both components (Cosine Similarity: {cos_sim_mean:.4f} < 0.9, Correlation: {corr_mean:.4f} < 0.8)")
    else:
        add_line(f"⚠️  Results are MIXED - Some components match well, others need attention (Cosine Similarity: {cos_sim_mean:.4f}, Correlation: {corr_mean:.4f})")

    add_line("=" * 80)

    # Save to file if requested
    if output_file:
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write('\n'.join(report_lines))
        logger.info(f"Report saved to {output_file}")

def main():
    parser = argparse.ArgumentParser(description='Evaluate generated pickle files against reference data')
    parser.add_argument('--generated-dir', default='.', help='Directory containing generated pickle files (default: current directory)')
    parser.add_argument('--eval-dir', default='eval', help='Directory containing reference pickle files (default: eval)')
    parser.add_argument('--output', help='Output file for detailed report (optional)')
    parser.add_argument('--verbose', '-v', action='store_true', help='Enable verbose logging')
    parser.add_argument('--embedding-only', action='store_true', help='Evaluate only embeddings (skip reranker)')
    parser.add_argument('--reranker-only', action='store_true', help='Evaluate only reranker scores (skip embeddings)')

    args = parser.parse_args()

    # Validate mutually exclusive arguments
    if args.embedding_only and args.reranker_only:
        logger.error("Cannot specify both --embedding-only and --reranker-only")
        sys.exit(1)

    if args.verbose:
        logger.remove()
        logger.add(sys.stderr, level="DEBUG")

    generated_dir = Path(args.generated_dir)
    eval_dir = Path(args.eval_dir)

    # Check if directories exist
    if not generated_dir.exists():
        logger.error(f"Generated directory does not exist: {generated_dir}")
        sys.exit(1)

    if not eval_dir.exists():
        logger.error(f"Evaluation directory does not exist: {eval_dir}")
        sys.exit(1)

    # Define file paths
    embedding_gen_path = generated_dir / "embedding.pkl"
    embedding_eval_path = eval_dir / "embedding.pkl"
    reranker_gen_path = generated_dir / "reranker.pkl"
    reranker_eval_path = eval_dir / "reranker.pkl"

    # Check if files exist based on evaluation mode
    missing_files = []

    # Always check embedding files unless reranker-only mode
    if not args.reranker_only:
        for name, path in [("generated embedding", embedding_gen_path),
                           ("reference embedding", embedding_eval_path)]:
            if not path.exists():
                missing_files.append(f"{name}: {path}")

    # Always check reranker files unless embedding-only mode
    if not args.embedding_only:
        for name, path in [("generated reranker", reranker_gen_path),
                           ("reference reranker", reranker_eval_path)]:
            if not path.exists():
                missing_files.append(f"{name}: {path}")

    if missing_files:
        logger.error("Missing required files:")
        for file in missing_files:
            logger.error(f"  - {file}")
        sys.exit(1)

    logger.info("Loading pickle files...")

    # Load data based on evaluation mode
    embedding_results = None
    reranker_results = None

    if not args.reranker_only:
        logger.info("Loading embedding data...")
        generated_embeddings = load_pickle_file(embedding_gen_path)
        reference_embeddings = load_pickle_file(embedding_eval_path)

        if generated_embeddings is None or reference_embeddings is None:
            logger.error("Failed to load embedding pickle files")
            sys.exit(1)

        logger.info("Comparing embeddings...")
        embedding_results = compare_embeddings(generated_embeddings, reference_embeddings)

    if not args.embedding_only:
        logger.info("Loading reranker data...")
        generated_reranker = load_pickle_file(reranker_gen_path)
        reference_reranker = load_pickle_file(reranker_eval_path)

        if generated_reranker is None or reference_reranker is None:
            logger.error("Failed to load reranker pickle files")
            sys.exit(1)

        logger.info("Comparing reranker scores...")
        reranker_results = compare_reranker_scores(generated_reranker, reference_reranker)

    # Print and save report
    print_summary_report(embedding_results, reranker_results, args.output)

    # Save detailed results as JSON if output file is specified
    if args.output:
        detailed_results = {}
        if embedding_results is not None:
            detailed_results['embedding_comparison'] = embedding_results
        if reranker_results is not None:
            detailed_results['reranker_comparison'] = reranker_results

        json_output = args.output.replace('.txt', '.json') if args.output.endswith('.txt') else args.output + '.json'
        with open(json_output, 'w', encoding='utf-8') as f:
            json.dump(detailed_results, f, indent=2, ensure_ascii=False)
        logger.info(f"Detailed results saved to {json_output}")

if __name__ == '__main__':
    main()
