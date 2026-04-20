"""
Dynamic bucketing algorithm for protein sequences
"""

import numpy as np
import logging
from typing import List, Tuple, Dict
from collections import Counter

logger = logging.getLogger(__name__)
import math


class DynamicBucketing:
    """
    Dynamic bucketing algorithm that automatically determines optimal bucket boundaries
    based on sequence length distribution
    """
    
    @staticmethod
    def analyze_length_distribution(lengths: List[int]) -> Dict:
        """
        Analyze sequence length distribution
        
        Args:
            lengths (List[int]): List of sequence lengths
            
        Returns:
            Dict: Distribution statistics
        """
        lengths = np.array(lengths)
        
        stats = {
            'min_length': int(np.min(lengths)),
            'max_length': int(np.max(lengths)),
            'mean_length': float(np.mean(lengths)),
            'median_length': float(np.median(lengths)),
            'std_length': float(np.std(lengths)),
            'percentiles': {
                '25': float(np.percentile(lengths, 25)),
                '50': float(np.percentile(lengths, 50)),
                '75': float(np.percentile(lengths, 75)),
                '90': float(np.percentile(lengths, 90)),
                '95': float(np.percentile(lengths, 95)),
                '99': float(np.percentile(lengths, 99))
            }
        }
        
        return stats
    
    @staticmethod
    def calculate_optimal_bucket_count(num_proteins: int, target_options: List[int] = [4, 8, 16, 32]) -> int:
        """
        Calculate optimal number of buckets based on protein count
        
        Args:
            num_proteins (int): Total number of proteins
            target_options (List[int]): Possible bucket counts to choose from
            
        Returns:
            int: Optimal bucket count
        """
        # Heuristic: aim for 100-500 proteins per bucket on average
        ideal_proteins_per_bucket = 250
        ideal_buckets = max(1, round(num_proteins / ideal_proteins_per_bucket))
        
        # Find closest option
        best_option = min(target_options, key=lambda x: abs(x - ideal_buckets))
        
        # Ensure minimum bucket count
        if num_proteins < 100:
            return 4
        elif num_proteins < 1000:
            return min(8, best_option)
        else:
            return best_option
    
    @staticmethod
    def create_balanced_buckets(lengths: List[int], num_buckets: int) -> List[int]:
        """
        Create balanced bucket boundaries using quantile-based approach
        
        Args:
            lengths (List[int]): List of sequence lengths
            num_buckets (int): Number of buckets to create
            
        Returns:
            List[int]: Bucket boundaries
        """
        if not lengths:
            return [1024]  # Default fallback
            
        lengths = np.array(sorted(lengths))
        n = len(lengths)
        
        if num_buckets >= n:
            # More buckets than proteins, use unique lengths
            return sorted(list(set(lengths)))
        
        # Calculate percentiles for balanced distribution
        percentiles = np.linspace(0, 100, num_buckets + 1)[1:]  # Skip 0%, include 100%
        boundaries = []
        
        for p in percentiles:
            if p == 100:
                # Last bucket includes maximum length
                boundaries.append(int(np.max(lengths)))
            else:
                # Use percentile
                boundary = int(np.percentile(lengths, p))
                boundaries.append(boundary)
        
        # Remove duplicates and sort
        boundaries = sorted(list(set(boundaries)))
        
        # Ensure minimum spacing between boundaries
        min_spacing = 8  # At least 8 amino acids difference
        cleaned_boundaries = [boundaries[0]]
        
        for boundary in boundaries[1:]:
            if boundary - cleaned_boundaries[-1] >= min_spacing:
                cleaned_boundaries.append(boundary)
        
        return cleaned_boundaries
    
    @staticmethod
    def create_adaptive_buckets(lengths: List[int], num_buckets: int) -> List[int]:
        """
        Create adaptive bucket boundaries using k-means clustering approach
        
        Args:
            lengths (List[int]): List of sequence lengths
            num_buckets (int): Number of buckets to create
            
        Returns:
            List[int]: Bucket boundaries
        """
        if not lengths:
            return [1024]
            
        lengths = np.array(sorted(lengths))
        n = len(lengths)
        
        if num_buckets >= n:
            return sorted(list(set(lengths)))
        
        # Use simple k-means-like approach to find natural clusters
        # Start with equal-sized groups
        group_size = n // num_buckets
        boundaries = []
        
        for i in range(1, num_buckets):
            idx = i * group_size
            if idx < n:
                # Find a good boundary point (avoid splitting similar lengths)
                start_idx = max(0, idx - 5)
                end_idx = min(n, idx + 5)
                
                # Find the position with largest gap
                best_idx = idx
                max_gap = 0
                
                for j in range(start_idx, end_idx - 1):
                    gap = lengths[j + 1] - lengths[j]
                    if gap > max_gap:
                        max_gap = gap
                        best_idx = j + 1
                
                boundaries.append(int(lengths[best_idx]))
        
        # Add maximum length as final boundary
        boundaries.append(int(np.max(lengths)))
        
        # Remove duplicates and sort
        boundaries = sorted(list(set(boundaries)))
        
        return boundaries
    
    @staticmethod
    def optimize_buckets(lengths: List[int], 
                        bucket_options: List[int] = [4, 8, 16, 32],
                        method: str = 'balanced') -> Tuple[int, List[int]]:
        """
        Optimize bucket configuration
        
        Args:
            lengths (List[int]): List of sequence lengths
            bucket_options (List[int]): Possible bucket counts
            method (str): 'balanced' or 'adaptive'
            
        Returns:
            Tuple[int, List[int]]: (optimal_bucket_count, boundaries)
        """
        if not lengths:
            return 4, [1024]
        
        num_proteins = len(lengths)
        optimal_count = DynamicBucketing.calculate_optimal_bucket_count(num_proteins, bucket_options)
        
        if method == 'adaptive':
            boundaries = DynamicBucketing.create_adaptive_buckets(lengths, optimal_count)
        else:  # balanced
            boundaries = DynamicBucketing.create_balanced_buckets(lengths, optimal_count)
        
        return optimal_count, boundaries
    
    @staticmethod
    def evaluate_bucketing(lengths: List[int], boundaries: List[int]) -> Dict:
        """
        Evaluate the quality of a bucketing scheme
        
        Args:
            lengths (List[int]): List of sequence lengths
            boundaries (List[int]): Bucket boundaries
            
        Returns:
            Dict: Evaluation metrics
        """
        if not lengths or not boundaries:
            return {}
        
        # Assign proteins to buckets
        bucket_assignments = []
        bucket_counts = [0] * len(boundaries)
        
        for length in lengths:
            assigned = False
            for i, boundary in enumerate(boundaries):
                if length <= boundary:
                    bucket_assignments.append(i)
                    bucket_counts[i] += 1
                    assigned = True
                    break
            
            if not assigned:
                # Assign to last bucket
                bucket_assignments.append(len(boundaries) - 1)
                bucket_counts[-1] += 1
        
        # Calculate metrics
        total_proteins = len(lengths)
        bucket_sizes = np.array(bucket_counts)
        
        # Balance metric (lower is better)
        mean_size = total_proteins / len(boundaries)
        balance_score = np.std(bucket_sizes) / mean_size if mean_size > 0 else float('inf')
        
        # Utilization metric (higher is better)
        non_empty_buckets = np.sum(bucket_sizes > 0)
        utilization_score = non_empty_buckets / len(boundaries)
        
        # Efficiency metric (considers padding waste)
        efficiency_scores = []
        for i, boundary in enumerate(boundaries):
            bucket_proteins = [l for j, l in enumerate(lengths) if bucket_assignments[j] == i]
            if bucket_proteins:
                avg_length = np.mean(bucket_proteins)
                efficiency = avg_length / boundary
                efficiency_scores.append(efficiency)
        
        avg_efficiency = np.mean(efficiency_scores) if efficiency_scores else 0
        
        return {
            'bucket_counts': bucket_counts,
            'balance_score': float(balance_score),
            'utilization_score': float(utilization_score),
            'efficiency_score': float(avg_efficiency),
            'non_empty_buckets': int(non_empty_buckets),
            'total_buckets': len(boundaries)
        }


def create_dynamic_buckets(lengths: List[int], 
                          bucket_options: List[int] = [4, 8, 16, 32],
                          method: str = 'balanced',
                          verbose: bool = True) -> List[int]:
    """
    Create dynamic bucket boundaries based on sequence length distribution
    
    Args:
        lengths (List[int]): List of sequence lengths
        bucket_options (List[int]): Possible bucket counts to choose from
        method (str): 'balanced' or 'adaptive'
        verbose (bool): Whether to print statistics
        
    Returns:
        List[int]: Optimal bucket boundaries
    """
    if verbose:
        logger.info("创建动态分桶方案...")
        logger.info(f"蛋白质数量: {len(lengths)}")
        logger.info(f"长度范围: {min(lengths)}-{max(lengths)}")
    
    # Analyze distribution
    stats = DynamicBucketing.analyze_length_distribution(lengths)
    
    if verbose:
        logger.info("分布统计:")
        logger.info(f"   平均长度: {stats['mean_length']:.1f}")
        logger.info(f"   中位数: {stats['median_length']:.1f}")
        logger.info(f"   标准差: {stats['std_length']:.1f}")
        logger.info(f"   分位数: 25%={stats['percentiles']['25']:.0f}, "
              f"75%={stats['percentiles']['75']:.0f}, "
              f"95%={stats['percentiles']['95']:.0f}")
    
    # Find optimal bucketing
    optimal_count, boundaries = DynamicBucketing.optimize_buckets(
        lengths, bucket_options, method
    )
    
    # Evaluate the result
    evaluation = DynamicBucketing.evaluate_bucketing(lengths, boundaries)
    
    if verbose:
        logger.info("最优分桶方案:")
        logger.info(f"   分桶数量: {optimal_count}")
        logger.info(f"   分桶边界: {boundaries}")
        logger.info(f"   分桶大小: {evaluation['bucket_counts']}")
        logger.info(f"   平衡分数: {evaluation['balance_score']:.2f} (越低越好)")
        logger.info(f"   利用率: {evaluation['utilization_score']:.2f} (越高越好)")
        logger.info(f"   效率分数: {evaluation['efficiency_score']:.2f} (越高越好)")
    
    return boundaries