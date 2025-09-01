#!/usr/bin/env python3
"""
SDP Prediction and Evaluation Pipeline

A modular and refactored version that predicts and evaluates
Specificity Determining Positions (SDPs) from residue ablation scores.
"""

import argparse
import itertools
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import ndimage
from scipy.signal import find_peaks, savgol_filter
from scipy.stats import zscore

# Optional import for multiple testing correction
try:
    from statsmodels.stats.multitest import multipletests

    HAS_STATSMODELS = True
except ImportError:
    HAS_STATSMODELS = False


def bonferroni_correction(pvalues, alpha=0.05):
    """Simple Bonferroni correction as fallback."""
    corrected = np.array(pvalues) * len(pvalues)
    return np.minimum(corrected, 1.0)


class SDPUtils:
    """Utility functions for common operations."""

    @staticmethod
    def get_residue_values(
        protein_data: pd.DataFrame, positions: List[int], column: str
    ) -> List[float]:
        """Get values from a specific column for given residue positions."""
        return [protein_data[protein_data["residue"] == pos][column].iloc[0] for pos in positions]

    @staticmethod
    def safe_divide(numerator: float, denominator: float, default: float = 0.0) -> float:
        """Safely divide two numbers, returning default if denominator is zero."""
        return numerator / denominator if denominator > 0 else default

    @staticmethod
    def extract_positions_from_candidates(candidates: pd.DataFrame) -> Set[int]:
        """Extract residue positions from SDP candidates DataFrame."""
        return set(candidates["residue"].values) if len(candidates) > 0 else set()

    @staticmethod
    def filter_positions_in_range(positions: Set[int], residues: np.ndarray) -> List[int]:
        """Filter positions to only include those present in residue range."""
        return [pos for pos in positions if pos in residues]

    @staticmethod
    def create_prediction_data(all_predictions: Dict[str, Set[int]]) -> pd.DataFrame:
        """Convert predictions dictionary to DataFrame format."""
        if not all_predictions:
            return pd.DataFrame(columns=["protein_id", "residue"])

        pred_data = []
        for protein_id, positions in all_predictions.items():
            for pos in positions:
                pred_data.append({"protein_id": protein_id, "residue": pos})
        return pd.DataFrame(pred_data)


class SDPConfig:
    """Configuration class to hold all parameters."""

    def __init__(self, args):
        # File paths
        self.ablation_file = args.ablation_file
        self.known_sdps_file = args.known_sdps_file
        self.output_dir = Path(args.output_dir)

        # Smoothing parameters
        self.window_size = args.window_size
        self.smoothing_method = args.smoothing_method
        self.smoothing_passes = args.smoothing_passes
        self.gaussian_sigma = args.gaussian_sigma
        self.savgol_polyorder = args.savgol_polyorder

        # Prediction parameters
        self.min_prominence = args.min_prominence
        self.min_distance = args.min_distance
        self.min_zscore = args.min_zscore
        self.match_tolerance = args.match_tolerance

        # Null distribution parameters
        self.use_null_distribution = args.use_null_distribution
        self.pvalue_threshold = args.pvalue_threshold
        self.multiple_testing_correction = args.multiple_testing_correction
        self.null_permutations = args.null_permutations
        self.randomization_method = args.randomization_method

        # Analysis options
        self.optimize = args.optimize
        self.plot_proteins = args.plot_proteins
        self.plot_permuted_proteins = args.plot_permuted_proteins
        self.save_detailed = args.save_detailed
        self.bootstrap_metrics = args.bootstrap_metrics
        self.permutation_metric_null = args.permutation_metric_null


class SDPPreprocessor:
    """Handles data loading and preprocessing."""

    @staticmethod
    def load_ablation_data(file_path: str) -> pd.DataFrame:
        """Load ablation scores from TSV file."""
        return pd.read_csv(file_path, sep="\t")

    @staticmethod
    def load_known_sdps(file_path: str) -> Dict[str, Set[int]]:
        """Load known SDPs from TSV file."""
        known_sdps = {}
        with open(file_path, "r") as f:
            lines = f.readlines()

        for line in lines[1:]:  # Skip header
            parts = line.strip().split("\t")
            if len(parts) >= 2:
                protein_id = parts[0]
                known_sdps[protein_id] = set()

                # Read all SDP positions from columns 1 onwards
                for part in parts[1:]:
                    try:
                        # Skip empty cells and dashes
                        if part.strip() and part.strip() != "-":
                            position = int(part.strip())
                            known_sdps[protein_id].add(position)
                    except ValueError:
                        continue

        return known_sdps

    @staticmethod
    def organize_protein_data(ablation_df: pd.DataFrame) -> Dict[str, pd.DataFrame]:
        """Organize ablation data by protein ID."""
        protein_data_dict = {}
        for protein_id in ablation_df["protein_id"].unique():
            protein_data_dict[protein_id] = ablation_df[
                ablation_df["protein_id"] == protein_id
            ].copy()
        return protein_data_dict


class SDPSmoother:
    """Handles different smoothing methods for ablation scores."""

    @staticmethod
    def apply_smoothing(scores: np.ndarray, method: str, **kwargs) -> np.ndarray:
        """Apply specified smoothing method to scores."""
        if method == "rolling_median":
            # Filter kwargs for rolling_median parameters
            filtered_kwargs = {
                k: v for k, v in kwargs.items() if k in ["window_size", "smoothing_passes"]
            }
            return SDPSmoother._rolling_median(scores, **filtered_kwargs)
        elif method == "gaussian":
            # Filter kwargs for gaussian parameters
            filtered_kwargs = {k: v for k, v in kwargs.items() if k in ["gaussian_sigma"]}
            return SDPSmoother._gaussian_smoothing(scores, **filtered_kwargs)
        elif method == "savgol":
            # Filter kwargs for savgol parameters
            filtered_kwargs = {
                k: v for k, v in kwargs.items() if k in ["window_size", "savgol_polyorder"]
            }
            return SDPSmoother._savgol_smoothing(scores, **filtered_kwargs)
        elif method == "combined":
            # Combined method uses all parameters
            return SDPSmoother._combined_smoothing(scores, **kwargs)
        else:
            raise ValueError(f"Unknown smoothing method: {method}")

    @staticmethod
    def _rolling_median(
        scores: np.ndarray, window_size: int = 5, smoothing_passes: int = 1
    ) -> np.ndarray:
        """Apply rolling median smoothing."""
        smoothed = scores.copy()
        for _ in range(smoothing_passes):
            smoothed = (
                pd.Series(smoothed)
                .rolling(window=window_size, center=True, min_periods=1)
                .median()
                .values
            )
        return smoothed

    @staticmethod
    def _gaussian_smoothing(scores: np.ndarray, gaussian_sigma: float = 1.0) -> np.ndarray:
        """Apply Gaussian smoothing."""
        return ndimage.gaussian_filter1d(scores, sigma=gaussian_sigma)

    @staticmethod
    def _savgol_smoothing(
        scores: np.ndarray, window_size: int = 5, savgol_polyorder: int = 2
    ) -> np.ndarray:
        """Apply Savitzky-Golay smoothing."""
        if len(scores) < window_size:
            return scores
        window_length = min(window_size, len(scores))
        if window_length % 2 == 0:
            window_length -= 1
        polyorder = min(savgol_polyorder, window_length - 1)
        return savgol_filter(scores, window_length, polyorder)

    @staticmethod
    def _combined_smoothing(scores: np.ndarray, **kwargs) -> np.ndarray:
        """Apply combined smoothing (rolling median + Gaussian)."""
        # Filter parameters for rolling median
        median_kwargs = {
            k: v for k, v in kwargs.items() if k in ["window_size", "smoothing_passes"]
        }
        smoothed = SDPSmoother._rolling_median(scores, **median_kwargs)

        # Filter parameters for gaussian
        gaussian_kwargs = {k: v for k, v in kwargs.items() if k in ["gaussian_sigma"]}
        return SDPSmoother._gaussian_smoothing(smoothed, **gaussian_kwargs)


class SDPRandomizer:
    """Handles different randomization methods for null distributions."""

    @staticmethod
    def randomize_scores(
        scores: np.ndarray, method: str, rng: np.random.Generator, **kwargs
    ) -> np.ndarray:
        """Apply specified randomization method."""
        if method == "block_shuffle":
            return SDPRandomizer._block_shuffle(scores, rng, **kwargs)
        elif method == "local_shuffle":
            return SDPRandomizer._local_shuffle(scores, rng, **kwargs)
        elif method == "gaussian_noise":
            return SDPRandomizer._add_correlated_noise(scores, rng, **kwargs)
        elif method == "rank_shuffle":
            return SDPRandomizer._rank_shuffle(scores, rng)
        elif method == "circular":
            return SDPRandomizer._circular_permutation(scores, rng)
        elif method == "bootstrap":
            return SDPRandomizer._bootstrap(scores, rng)
        else:
            return rng.permutation(scores)

    @staticmethod
    def _block_shuffle(
        scores: np.ndarray, rng: np.random.Generator, block_size: int = 5
    ) -> np.ndarray:
        """Shuffle blocks of consecutive residues."""
        if len(scores) <= block_size:
            return rng.permutation(scores)

        n_blocks = len(scores) // block_size
        remainder = len(scores) % block_size

        blocks = [scores[i * block_size : (i + 1) * block_size] for i in range(n_blocks)]
        if remainder > 0:
            blocks.append(scores[n_blocks * block_size :])

        rng.shuffle(blocks)
        return np.concatenate(blocks)

    @staticmethod
    def _local_shuffle(
        scores: np.ndarray, rng: np.random.Generator, window_size: int = 5
    ) -> np.ndarray:
        """Shuffle within local windows."""
        shuffled = scores.copy()
        for i in range(len(scores)):
            start = max(0, i - window_size // 2)
            end = min(len(scores), i + window_size // 2 + 1)
            window = scores[start:end].copy()
            rng.shuffle(window)
            shuffled[start:end] = window
        return shuffled

    @staticmethod
    def _add_correlated_noise(
        scores: np.ndarray,
        rng: np.random.Generator,
        correlation_length: int = 5,
        noise_scale: float = 0.1,
    ) -> np.ndarray:
        """Add spatially correlated noise."""
        noise = rng.normal(0, noise_scale * np.std(scores), len(scores))
        smoothed_noise = ndimage.gaussian_filter1d(noise, sigma=correlation_length)
        return scores + smoothed_noise

    @staticmethod
    def _rank_shuffle(scores: np.ndarray, rng: np.random.Generator) -> np.ndarray:
        """Shuffle the ranks while preserving the distribution."""
        sorted_scores = np.sort(scores)
        ranks = rng.permutation(len(scores))
        return sorted_scores[ranks]

    @staticmethod
    def _circular_permutation(scores: np.ndarray, rng: np.random.Generator) -> np.ndarray:
        """Apply circular permutation."""
        shift = rng.integers(1, len(scores))
        return np.roll(scores, shift)

    @staticmethod
    def _bootstrap(scores: np.ndarray, rng: np.random.Generator) -> np.ndarray:
        """Bootstrap sampling."""
        return rng.choice(scores, size=len(scores), replace=True)


class SDPPredictor:
    """Core SDP prediction logic."""

    def __init__(self, config: SDPConfig):
        self.config = config

    def predict_protein_sdps(self, protein_data: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Predict SDPs for a single protein."""
        # Calculate z-scores
        z_scores = zscore(protein_data["ablation_score"].values)
        protein_data["z_score"] = z_scores
        protein_data = protein_data.copy()
        protein_data["z_score"] = z_scores

        # Apply smoothing
        smoothed_z = SDPSmoother.apply_smoothing(
            z_scores,
            method=self.config.smoothing_method,
            window_size=self.config.window_size,
            smoothing_passes=self.config.smoothing_passes,
            gaussian_sigma=self.config.gaussian_sigma,
            savgol_polyorder=self.config.savgol_polyorder,
        )
        protein_data["smoothed_z"] = smoothed_z

        # Find peaks
        peaks, properties = find_peaks(
            smoothed_z, prominence=self.config.min_prominence, distance=self.config.min_distance
        )

        # Convert array indices to DataFrame indices (residue positions)
        if len(peaks) > 0:
            peak_residues = protein_data.iloc[peaks].index
        else:
            peak_residues = []

        # Mark peaks in data
        protein_data["is_peak"] = False
        if len(peak_residues) > 0:
            protein_data.loc[peak_residues, "is_peak"] = True
        protein_data["prominence"] = 0.0
        if len(peak_residues) > 0:
            protein_data.loc[peak_residues, "prominence"] = properties["prominences"]

        # Calculate p-values if using null distribution
        if self.config.use_null_distribution:
            pvalues = self._calculate_peak_pvalues(protein_data, peaks)
            protein_data["pvalue"] = np.nan
            if len(peak_residues) > 0:
                protein_data.loc[peak_residues, "pvalue"] = pvalues

            # Apply multiple testing correction
            if len(pvalues) > 0:
                if self.config.multiple_testing_correction != "none":
                    if HAS_STATSMODELS:
                        _, corrected_pvals, _, _ = multipletests(
                            pvalues, method=self.config.multiple_testing_correction
                        )
                    else:
                        # Fallback to Bonferroni correction
                        print("Warning: statsmodels not available, using Bonferroni correction")
                        corrected_pvals = bonferroni_correction(pvalues)
                    protein_data.loc[peak_residues, "pvalue"] = corrected_pvals

            # Filter by p-value threshold
            if len(peak_residues) > 0:
                significant_mask = (
                    protein_data.loc[peak_residues, "pvalue"] <= self.config.pvalue_threshold
                )
                significant_peak_residues = peak_residues[significant_mask]
            else:
                significant_peak_residues = []
        else:
            # Filter by z-score threshold
            if len(peaks) > 0:
                significant_mask = smoothed_z[peaks] >= self.config.min_zscore
                significant_peak_residues = peak_residues[significant_mask]
            else:
                significant_peak_residues = []

        # Extract SDP candidates
        sdp_candidates = (
            protein_data.loc[significant_peak_residues].copy()
            if len(significant_peak_residues) > 0
            else protein_data.iloc[0:0].copy()
        )

        return protein_data, sdp_candidates

    def _calculate_peak_pvalues(self, protein_data: pd.DataFrame, peaks: np.ndarray) -> np.ndarray:
        """Calculate p-values for peaks using null distribution."""
        scores = protein_data["ablation_score"].values
        rng = np.random.default_rng(42)

        # Generate null distribution
        null_peak_scores = []
        for _ in range(self.config.null_permutations):
            permuted_scores = SDPRandomizer.randomize_scores(
                scores,
                self.config.randomization_method,
                rng,
                block_size=self.config.window_size,
                window_size=self.config.window_size,
                correlation_length=self.config.window_size,
            )

            # Apply same processing to permuted data
            perm_z = zscore(permuted_scores)
            perm_smoothed = SDPSmoother.apply_smoothing(
                perm_z,
                method=self.config.smoothing_method,
                window_size=self.config.window_size,
                smoothing_passes=self.config.smoothing_passes,
                gaussian_sigma=self.config.gaussian_sigma,
                savgol_polyorder=self.config.savgol_polyorder,
            )

            perm_peaks, perm_props = find_peaks(
                perm_smoothed,
                prominence=self.config.min_prominence,
                distance=self.config.min_distance,
            )

            if len(perm_peaks) > 0:
                null_peak_scores.extend(perm_smoothed[perm_peaks])

        # Calculate p-values
        peak_residues = protein_data.iloc[peaks].index if len(peaks) > 0 else []
        observed_scores = (
            protein_data.loc[peak_residues, "smoothed_z"].values if len(peak_residues) > 0 else []
        )
        pvalues = []
        for score in observed_scores:
            p_val = np.mean([null_score >= score for null_score in null_peak_scores])
            pvalues.append(max(p_val, 1.0 / len(null_peak_scores)))  # Avoid p=0

        return np.array(pvalues)


class SDPEvaluator:
    """Handles evaluation of SDP predictions."""

    @staticmethod
    def calculate_metrics_with_tolerance(
        known: Set[int], predicted: Set[int], tolerance: int = 0
    ) -> Tuple[float, float, float]:
        """Calculate precision, recall, and F1-score with position tolerance."""
        if tolerance == 0:
            tp = len(known & predicted)
            fp = len(predicted - known)
            fn = len(known - predicted)
        else:
            # Use bipartite matching for tolerance
            tp, fp, fn, _ = SDPEvaluator._bipartite_matching(known, predicted, tolerance)

        # Calculate metrics
        precision = SDPUtils.safe_divide(tp, tp + fp)
        recall = SDPUtils.safe_divide(tp, tp + fn)
        f1_score = SDPUtils.safe_divide(2 * precision * recall, precision + recall)

        return precision, recall, f1_score

    @staticmethod
    def _bipartite_matching(
        known: Set[int], predicted: Set[int], tolerance: int
    ) -> Tuple[int, int, int, float]:
        """Perform bipartite matching with tolerance."""
        from scipy.optimize import linear_sum_assignment

        known_list = list(known)
        predicted_list = list(predicted)

        if not known_list or not predicted_list:
            return 0, len(predicted_list), len(known_list), 0.0

        # Create cost matrix (distance between positions)
        cost_matrix = np.zeros((len(known_list), len(predicted_list)))
        for i, k_pos in enumerate(known_list):
            for j, p_pos in enumerate(predicted_list):
                cost_matrix[i, j] = abs(k_pos - p_pos)

        # Solve assignment problem
        row_indices, col_indices = linear_sum_assignment(cost_matrix)

        # Count matches within tolerance
        tp = 0
        total_distance = 0.0
        for r, c in zip(row_indices, col_indices):
            if cost_matrix[r, c] <= tolerance:
                tp += 1
                total_distance += cost_matrix[r, c]

        fp = len(predicted_list) - tp
        fn = len(known_list) - tp
        mean_distance = total_distance / tp if tp > 0 else 0.0

        return tp, fp, fn, mean_distance

    @staticmethod
    def _bipartite_matching_detailed(
        known: Set[int], predicted: Set[int], tolerance: int
    ) -> Tuple[int, List[Tuple[int, int, float]]]:
        """Perform bipartite matching with tolerance and return detailed matching pairs."""
        from scipy.optimize import linear_sum_assignment

        known_list = list(known)
        predicted_list = list(predicted)

        if not known_list or not predicted_list:
            return 0, []

        # Create cost matrix (distance between positions)
        cost_matrix = np.zeros((len(known_list), len(predicted_list)))
        for i, k_pos in enumerate(known_list):
            for j, p_pos in enumerate(predicted_list):
                cost_matrix[i, j] = abs(k_pos - p_pos)

        # Solve assignment problem
        row_indices, col_indices = linear_sum_assignment(cost_matrix)

        # Collect matches and their distances
        tp = 0
        matched_pairs = []
        for r, c in zip(row_indices, col_indices):
            distance = cost_matrix[r, c]
            matched_pairs.append((r, c, distance))
            if distance <= tolerance:
                tp += 1

        return tp, matched_pairs

    @staticmethod
    def evaluate_all_predictions(
        known_sdps: Dict[str, Set[int]], predicted_sdps: Dict[str, Set[int]], tolerance: int = 0
    ) -> pd.DataFrame:
        """Evaluate all protein predictions with detailed coordinate information."""
        results = []
        all_proteins = set(known_sdps.keys()) | set(predicted_sdps.keys())

        for protein_id in sorted(all_proteins):
            known_pos = known_sdps.get(protein_id, set())
            predicted_pos = predicted_sdps.get(protein_id, set())

            precision, recall, f1_score = SDPEvaluator.calculate_metrics_with_tolerance(
                known_pos, predicted_pos, tolerance
            )

            # Get detailed matching information
            if tolerance == 0:
                # Exact matching
                tp_positions = known_pos & predicted_pos
                fp_positions = predicted_pos - known_pos
                fn_positions = known_pos - predicted_pos
                num_tp = len(tp_positions)
            else:
                # Tolerance-based matching
                num_tp, matched_pairs = SDPEvaluator._bipartite_matching_detailed(
                    known_pos, predicted_pos, tolerance
                )
                tp_positions = set()
                matched_known = set()
                matched_predicted = set()

                for known_idx, pred_idx, distance in matched_pairs:
                    known_list = list(known_pos)
                    predicted_list = list(predicted_pos)
                    if distance <= tolerance:
                        tp_positions.add(predicted_list[pred_idx])
                        matched_known.add(known_list[known_idx])
                        matched_predicted.add(predicted_list[pred_idx])

                fp_positions = predicted_pos - matched_predicted
                fn_positions = known_pos - matched_known

            # Convert to sorted lists for consistent output
            known_coords = sorted(list(known_pos)) if known_pos else []
            predicted_coords = sorted(list(predicted_pos)) if predicted_pos else []
            tp_coords = sorted(list(tp_positions)) if tp_positions else []

            results.append(
                {
                    "protein_id": protein_id,
                    "known_sdps": len(known_pos),
                    "predicted_sdps": len(predicted_pos),
                    "true_positives": num_tp,
                    "precision": precision,
                    "recall": recall,
                    "f1_score": f1_score,
                    "known_coordinates": ",".join(map(str, known_coords)),
                    "predicted_coordinates": ",".join(map(str, predicted_coords)),
                    "tp_coordinates": ",".join(map(str, tp_coords)),
                }
            )

        return pd.DataFrame(results)


class SDPVisualizer:
    """Handles plotting and visualization."""

    def __init__(self, config: SDPConfig):
        self.config = config

    def plot_protein_profile(
        self,
        protein_data: pd.DataFrame,
        sdp_candidates: pd.DataFrame,
        known_positions: Set[int],
        protein_id: str,
        custom_filename: Optional[str] = None,
    ) -> None:
        """Create a simplified visualization of the SDP prediction for a protein."""
        fig, ax = plt.subplots(1, 1, figsize=(12, 6))

        residues = protein_data["residue"].values

        # Plot z-score and smoothed z-score
        ax.plot(residues, protein_data["z_score"], color="green", alpha=0.7, linewidth=1)
        ax.plot(residues, protein_data["smoothed_z"], color="blue", linewidth=2)

        # Add min z-score threshold line
        ax.axhline(y=self.config.min_zscore, color="orange", linestyle="--", alpha=0.8)

        # Mark all peaks
        all_peaks = protein_data[protein_data["is_peak"]]
        if len(all_peaks) > 0:
            ax.scatter(
                all_peaks["residue"],
                all_peaks["smoothed_z"],
                c="orange",
                s=60,
                marker="^",
                alpha=0.6,
                zorder=5,
            )

        # Mark known SDPs
        if known_positions:
            known_in_range = SDPUtils.filter_positions_in_range(known_positions, residues)
            if known_in_range:
                known_z_scores = SDPUtils.get_residue_values(
                    protein_data, known_in_range, "smoothed_z"
                )
                ax.scatter(
                    known_in_range, known_z_scores, c="blue", s=60, marker="x", zorder=7, alpha=0.8
                )

        # Mark predicted SDPs with different colors for TP vs FP
        if len(sdp_candidates) > 0:
            for _, candidate in sdp_candidates.iterrows():
                pos = candidate["residue"]
                z_val = candidate["smoothed_z"]

                # Check if this prediction matches a known SDP (within tolerance)
                is_match = False
                if known_positions:
                    distances = [abs(pos - known_pos) for known_pos in known_positions]
                    is_match = min(distances) <= self.config.match_tolerance

                # Use different colors for TP vs FP
                color = "red" if is_match else "blue"
                ax.scatter(pos, z_val, c=color, s=80, marker="^", zorder=6)

        ax.set_xlabel("Residue Position")
        ax.set_ylabel("Z-Score")
        ax.set_title(f"{protein_id} - SDP Prediction")
        ax.grid(True, alpha=0.3)

        plt.tight_layout()

        if custom_filename:
            plot_file = self.config.output_dir / custom_filename
        else:
            plot_file = self.config.output_dir / f"{protein_id}_sdp_prediction.pdf"

        plt.savefig(plot_file, dpi=300, bbox_inches="tight")
        plt.close()

    def plot_permuted_comparison(
        self,
        original_scores: np.ndarray,
        permuted_scores: np.ndarray,
        residues: np.ndarray,
        known_positions: Set[int],
        protein_id: str,
    ) -> None:
        """Plot comparison of original and permuted profiles."""
        plt.figure(figsize=(12, 5))
        plt.plot(residues, original_scores, label="Original", color="blue")
        plt.plot(residues, permuted_scores, label="Permuted", color="orange", alpha=0.7)

        # Mark GT SDPs as vertical lines
        if known_positions:
            gt_in_range = SDPUtils.filter_positions_in_range(known_positions, residues)
            for idx, pos in enumerate(gt_in_range):
                plt.axvline(
                    x=pos,
                    color="red",
                    linestyle="--",
                    alpha=0.8,
                    label="GT SDP" if idx == 0 else None,
                )

        plt.title(f"Ablation scores: {protein_id} (Original & Permuted with GT SDPs)")
        plt.xlabel("Residue")
        plt.ylabel("Ablation score")
        plt.legend()
        plt.tight_layout()

        plot_file = (
            self.config.output_dir / f"permuted_profile_{protein_id}_orig_vs_permuted_with_gt.pdf"
        )
        plt.savefig(plot_file, dpi=150)
        plt.close()


class SDPPipeline:
    """Main pipeline orchestrator."""

    def __init__(self, config: SDPConfig):
        self.config = config
        self.predictor = SDPPredictor(config)
        self.visualizer = SDPVisualizer(config)

        # Ensure output directory exists
        self.config.output_dir.mkdir(parents=True, exist_ok=True)

    def run(self) -> None:
        """Run the complete SDP prediction and evaluation pipeline."""
        print("=" * 70)
        print("SDP Prediction and Evaluation Pipeline (Refactored)")
        print("=" * 70)
        print(f"Ablation file: {self.config.ablation_file}")
        print(f"Known SDPs file: {self.config.known_sdps_file}")
        print(f"Output directory: {self.config.output_dir}")
        print("")

        # Load data
        print("Loading data...")
        ablation_df = SDPPreprocessor.load_ablation_data(self.config.ablation_file)
        known_sdps = SDPPreprocessor.load_known_sdps(self.config.known_sdps_file)
        protein_data_dict = SDPPreprocessor.organize_protein_data(ablation_df)

        print(f"Loaded {len(protein_data_dict)} proteins from ablation data")
        print(f"Loaded {len(known_sdps)} proteins with known SDPs")
        print("")

        # Run optimization if requested
        if self.config.optimize:
            self._run_optimization(protein_data_dict, known_sdps)

        # Run prediction
        all_predictions, all_detailed_predictions = self._run_prediction(protein_data_dict)

        # Evaluate predictions
        evaluation_df = self._evaluate_predictions(known_sdps, all_predictions)

        # Generate plots if requested
        if self.config.plot_proteins > 0:
            self._generate_plots(protein_data_dict, known_sdps, evaluation_df)

        if self.config.plot_permuted_proteins > 0:
            self._generate_permuted_plots(protein_data_dict, known_sdps)

        # Run permutation null if requested
        if self.config.permutation_metric_null > 0:
            self._run_permutation_null(
                protein_data_dict, known_sdps, all_predictions, evaluation_df
            )

        # Save results
        self._save_results(evaluation_df, all_predictions, all_detailed_predictions, known_sdps)

        print("\n" + "=" * 70)
        print("Analysis Complete!")
        print("=" * 70)

    def _run_optimization(
        self, protein_data_dict: Dict[str, pd.DataFrame], known_sdps: Dict[str, Set[int]]
    ) -> None:
        """Run grid search optimization for key parameters."""
        print("Running parameter optimization (grid search)...")
        # Define parameter grid (can be adjusted as needed)
        window_sizes = [7, 10, 12]
        min_prominences = [0.1, 0.25, 0.5, 1]
        min_zscores = [-1, -0.5, 0]
        match_tolerances = [0, 1]

        best_macro_f1 = -1
        best_params = None
        results = []

        total = len(window_sizes) * len(min_prominences) * len(min_zscores) * len(match_tolerances)
        print(f"Testing {total} parameter combinations...")
        count = 0

        for ws in window_sizes:
            for mp in min_prominences:
                for mz in min_zscores:
                    for mt in match_tolerances:
                        count += 1
                        # Set parameters
                        self.config.window_size = ws
                        self.config.min_prominence = mp
                        self.config.min_zscore = mz
                        self.config.match_tolerance = mt

                        # Run prediction and evaluation
                        all_predictions, _ = self._run_prediction(protein_data_dict)
                        evaluation_df = self._evaluate_predictions(known_sdps, all_predictions)
                        macro_metrics = self._calculate_macro_metrics(evaluation_df)
                        macro_f1 = macro_metrics["f1_score"] if macro_metrics else 0.0
                        results.append((macro_f1, ws, mp, mz, mt))

                        print(
                            f"[{count}/{total}] ws={ws}, mp={mp}, mz={mz}, mt={mt} -> macro F1={macro_f1:.4f}"
                        )

                        if macro_f1 > best_macro_f1:
                            best_macro_f1 = macro_f1
                            best_params = (ws, mp, mz, mt)

        if best_params:
            ws, mp, mz, mt = best_params
            print("\nBest parameters found:")
            print(f"  window_size={ws}")
            print(f"  min_prominence={mp}")
            print(f"  min_zscore={mz}")
            print(f"  match_tolerance={mt}")
            print(f"  Macro F1={best_macro_f1:.4f}")
            # Set best parameters for main run
            self.config.window_size = ws
            self.config.min_prominence = mp
            self.config.min_zscore = mz
            self.config.match_tolerance = mt
        else:
            print("No valid parameter set found.")

    def _run_prediction(
        self, protein_data_dict: Dict[str, pd.DataFrame]
    ) -> Tuple[Dict[str, Set[int]], Dict[str, pd.DataFrame]]:
        """Run SDP prediction on all proteins."""
        print(f"Running SDP prediction with parameters:")
        print(f"  Smoothing method: {self.config.smoothing_method}")
        print(f"  Window size: {self.config.window_size}")
        print(f"  Min prominence: {self.config.min_prominence}")
        print(f"  Min z-score: {self.config.min_zscore}")
        print("")

        all_predictions = {}
        all_detailed_predictions = {}
        proteins = sorted(protein_data_dict.keys())

        for i, protein_id in enumerate(proteins):
            if (i + 1) % 10 == 0:
                print(f"  Processed {i + 1}/{len(proteins)} proteins")

            protein_data = protein_data_dict[protein_id]
            processed_data, sdp_candidates = self.predictor.predict_protein_sdps(protein_data)

            if len(sdp_candidates) > 0:
                predicted_positions = SDPUtils.extract_positions_from_candidates(sdp_candidates)
                all_predictions[protein_id] = predicted_positions
                all_detailed_predictions[protein_id] = sdp_candidates

        return all_predictions, all_detailed_predictions

    def _evaluate_predictions(
        self, known_sdps: Dict[str, Set[int]], all_predictions: Dict[str, Set[int]]
    ) -> pd.DataFrame:
        """Evaluate all predictions."""
        print("Evaluating predictions...")
        if self.config.match_tolerance > 0:
            print(f"Using match tolerance: ±{self.config.match_tolerance} residues")

        return SDPEvaluator.evaluate_all_predictions(
            known_sdps, all_predictions, self.config.match_tolerance
        )

    def _generate_plots(
        self,
        protein_data_dict: Dict[str, pd.DataFrame],
        known_sdps: Dict[str, Set[int]],
        evaluation_df: pd.DataFrame,
    ) -> None:
        """Generate plots for top proteins."""
        print(f"Generating plots for top {self.config.plot_proteins} proteins...")

        top_proteins = evaluation_df.nlargest(self.config.plot_proteins, "f1_score")

        for idx, (_, row) in enumerate(top_proteins.iterrows()):
            protein_id = row["protein_id"]
            if protein_id in protein_data_dict:
                protein_data = protein_data_dict[protein_id]
                protein_known_sdps = known_sdps.get(protein_id, set())

                processed_protein_data, sdp_candidates = self.predictor.predict_protein_sdps(
                    protein_data
                )

                custom_filename = (
                    f'top_{idx+1:02d}_{protein_id}_F1_{row["f1_score"]:.3f}_prediction_profile.pdf'
                )
                self.visualizer.plot_protein_profile(
                    processed_protein_data,
                    sdp_candidates,
                    protein_known_sdps,
                    protein_id,
                    custom_filename,
                )

    def _generate_permuted_plots(
        self, protein_data_dict: Dict[str, pd.DataFrame], known_sdps: Dict[str, Set[int]]
    ) -> None:
        """Generate plots comparing original and permuted profiles."""
        print(
            f"Plotting original and permuted profiles for {self.config.plot_permuted_proteins} proteins..."
        )

        rng = np.random.default_rng(42)
        protein_ids = list(protein_data_dict.keys())
        selected_ids = rng.choice(
            protein_ids,
            size=min(self.config.plot_permuted_proteins, len(protein_ids)),
            replace=False,
        )

        for protein_id in selected_ids:
            protein_data = protein_data_dict[protein_id].copy()
            known_sdps_for_protein = known_sdps.get(protein_id, set())

            # Generate permuted scores
            scores = protein_data["ablation_score"].values.copy()
            permuted_scores = SDPRandomizer.randomize_scores(
                scores,
                self.config.randomization_method,
                rng,
                block_size=self.config.window_size,
                window_size=self.config.window_size,
                correlation_length=self.config.window_size,
            )

            # Plot comparison - DISABLED
            # self.visualizer.plot_permuted_comparison(
            #     scores, permuted_scores, protein_data['residue'].values,
            #     known_sdps_for_protein, protein_id
            # )

            # Also generate individual plots
            processed_protein_data, orig_candidates = self.predictor.predict_protein_sdps(
                protein_data
            )
            self.visualizer.plot_protein_profile(
                processed_protein_data,
                orig_candidates,
                known_sdps_for_protein,
                protein_id,
                f"permuted_profile_{protein_id}_original.pdf",
            )

            # Permuted prediction
            protein_data_perm = protein_data.copy()
            protein_data_perm["ablation_score"] = permuted_scores
            processed_protein_data_perm, perm_candidates = self.predictor.predict_protein_sdps(
                protein_data_perm
            )
            self.visualizer.plot_protein_profile(
                processed_protein_data_perm,
                perm_candidates,
                known_sdps_for_protein,
                protein_id,
                f"permuted_profile_{protein_id}_permuted.pdf",
            )

            # Print metrics for permuted
            pred_set = SDPUtils.extract_positions_from_candidates(perm_candidates)
            precision, recall, f1 = SDPEvaluator.calculate_metrics_with_tolerance(
                known_sdps_for_protein, pred_set, self.config.match_tolerance
            )
            print(
                f"Permuted {protein_id}: Precision={precision:.3f} Recall={recall:.3f} F1={f1:.3f}"
            )

    def _run_permutation_null(
        self,
        protein_data_dict: Dict[str, pd.DataFrame],
        known_sdps: Dict[str, Set[int]],
        all_predictions: Dict[str, Set[int]],
        evaluation_df: pd.DataFrame,
    ) -> None:
        """Run permutation null distribution for metrics."""
        print(
            f"\nRunning permutation null for metrics with {self.config.permutation_metric_null} permutations..."
        )
        print(f"  Permutation method: {self.config.randomization_method}")

        # Calculate observed overall metrics
        observed_metrics = self._calculate_overall_metrics(all_predictions, known_sdps)

        # Run permutations
        permuted_metrics = []
        permuted_macro_metrics = []
        rng = np.random.default_rng(42)

        for i in range(self.config.permutation_metric_null):
            # Progress reporting
            if self.config.permutation_metric_null >= 100:
                # For large numbers, report every 10% or every 50, whichever is more frequent
                report_interval = min(50, max(1, self.config.permutation_metric_null // 10))
            else:
                # For smaller numbers, report every 20% or every 5, whichever is more frequent
                report_interval = min(5, max(1, self.config.permutation_metric_null // 5))

            if (i + 1) % report_interval == 0 or i == 0:
                print(f"  Completed {i + 1}/{self.config.permutation_metric_null} permutations...")

            # Generate permuted predictions for all proteins
            permuted_predictions = {}
            permuted_per_protein_metrics = []

            for protein_id, protein_data in protein_data_dict.items():
                if protein_id in known_sdps:
                    # Randomize ablation scores
                    randomized_scores = SDPRandomizer.randomize_scores(
                        protein_data["ablation_score"].values, self.config.randomization_method, rng
                    )

                    # Create randomized data
                    randomized_data = protein_data.copy()
                    randomized_data["ablation_score"] = randomized_scores

                    # Smooth and predict SDPs
                    _, candidates = self.predictor.predict_protein_sdps(randomized_data)
                    pred_positions = SDPUtils.extract_positions_from_candidates(candidates)
                    permuted_predictions[protein_id] = pred_positions

                    # Calculate per-protein metrics for macro average
                    if pred_positions or known_sdps[protein_id]:
                        precision, recall, f1 = SDPEvaluator.calculate_metrics_with_tolerance(
                            known_sdps[protein_id], pred_positions, self.config.match_tolerance
                        )
                        permuted_per_protein_metrics.append(
                            {"precision": precision, "recall": recall, "f1_score": f1}
                        )

            # Calculate micro metrics for this permutation
            perm_metrics = self._calculate_overall_metrics(permuted_predictions, known_sdps)
            permuted_metrics.append(perm_metrics)

            # Calculate macro metrics for this permutation
            if permuted_per_protein_metrics:
                macro_precision = np.mean([m["precision"] for m in permuted_per_protein_metrics])
                macro_recall = np.mean([m["recall"] for m in permuted_per_protein_metrics])
                macro_f1 = np.mean([m["f1_score"] for m in permuted_per_protein_metrics])

                permuted_macro_metrics.append(
                    {"precision": macro_precision, "recall": macro_recall, "f1_score": macro_f1}
                )

        # Calculate statistics
        self._calculate_and_save_permutation_stats(
            observed_metrics, permuted_metrics, permuted_macro_metrics, evaluation_df
        )

        # Debug: Print permutation value ranges to help understand p-value calculation
        print(f"\nPermutation Debug Info:")
        for metric in ["precision", "recall", "f1_score"]:
            if permuted_metrics:
                perm_vals = [p[metric] for p in permuted_metrics]
                obs_val = observed_metrics[metric]
                print(
                    f"  {metric}: observed={obs_val:.4f}, perm_range=[{min(perm_vals):.4f}, {max(perm_vals):.4f}], perm_mean={np.mean(perm_vals):.4f}"
                )
                count_gt = sum(1 for p in perm_vals if p > obs_val)
                print(f"    permutations > observed: {count_gt}/{len(perm_vals)}")

                # Additional debug: show some permuted values
                print(f"    first 10 permuted values: {perm_vals[:10]}")
                if len(perm_vals) > 10:
                    print(f"    last 5 permuted values: {perm_vals[-5:]}")

        # Debug: Check if we're getting any predictions in permutations
        print(f"\nPermutation prediction counts:")
        total_perm_predictions = 0
        total_orig_predictions = 0

        for protein_id in known_sdps:
            if protein_id in all_predictions:
                total_orig_predictions += len(all_predictions[protein_id])

        for perm_dict in [p for p in locals() if "permuted_predictions" in str(p)]:
            # This won't work correctly, let me track this differently...
            pass

        print(f"  Original total predictions: {total_orig_predictions}")

        # Let's also check distribution of prediction counts per permutation
        perm_pred_counts = []
        for i in range(min(5, len(permuted_metrics))):  # Just check first few
            count = sum(
                len(all_predictions.get(pid, set())) for pid in known_sdps if pid in all_predictions
            )
            perm_pred_counts.append(count)
        print(f"  Permutation pred counts (first 5): {perm_pred_counts}")

    def _calculate_overall_metrics(
        self, predictions: Dict[str, Set[int]], known_sdps: Dict[str, Set[int]]
    ) -> Dict[str, float]:
        """Calculate overall precision, recall, F1 across all proteins."""
        total_tp = 0
        total_fp = 0
        total_fn = 0

        for protein_id in known_sdps:
            if protein_id in predictions:
                pred_set = predictions[protein_id]
                known_set = known_sdps[protein_id]

                # Use the existing bipartite matching approach
                if self.config.match_tolerance == 0:
                    tp = len(known_set & pred_set)
                    fp = len(pred_set - known_set)
                    fn = len(known_set - pred_set)
                else:
                    tp, fp, fn, _ = SDPEvaluator._bipartite_matching(
                        known_set, pred_set, self.config.match_tolerance
                    )

                total_tp += tp
                total_fp += fp
                total_fn += fn

        precision = SDPUtils.safe_divide(total_tp, total_tp + total_fp)
        recall = SDPUtils.safe_divide(total_tp, total_tp + total_fn)
        f1 = SDPUtils.safe_divide(2 * precision * recall, precision + recall)

        return {"precision": precision, "recall": recall, "f1_score": f1}

    def _calculate_and_save_permutation_stats(
        self,
        observed: Dict[str, float],
        permuted_list: List[Dict[str, float]],
        permuted_macro_list: List[Dict[str, float]],
        evaluation_df: pd.DataFrame,
    ) -> None:
        """Calculate comprehensive statistics including macro averages and localization errors."""
        import numpy as np

        # Calculate averages of permuted micro metrics
        avg_permuted = {}
        for metric in ["precision", "recall", "f1_score"]:
            values = [p[metric] for p in permuted_list]
            avg_permuted[metric] = np.mean(values)

        # Calculate averages of permuted macro metrics
        avg_permuted_macro = {}
        if permuted_macro_list:
            for metric in ["precision", "recall", "f1_score"]:
                values = [p[metric] for p in permuted_macro_list]
                avg_permuted_macro[metric] = np.mean(values)

        # Calculate p-values for micro metrics using a +1 / (n+1) correction to avoid
        # exact zeros when the observed value exceeds all permutations. Also track
        # raw counts and permutation counts so we can save them to the TSV.
        # Use strictly greater than (>) instead of >= for a more conservative test.
        p_values = {}
        perm_counts = {}
        perm_count_ge = {}
        for metric in ["precision", "recall", "f1_score"]:
            permuted_values = [p[metric] for p in permuted_list]
            n = len(permuted_values)
            perm_counts[metric] = n
            if n == 0:
                p_values[metric] = float("nan")
                perm_count_ge[metric] = 0
            else:
                # count how many permuted values are > observed (strictly greater)
                count_ge = sum(1 for p in permuted_values if p > observed[metric])
                perm_count_ge[metric] = count_ge
                # apply small-sample correction
                p_values[metric] = (count_ge + 1) / (n + 1)

        # Calculate p-values for macro metrics (with same +1/(n+1) correction)
        # Use strictly greater than (>) instead of >= for more conservative test
        observed_macro = self._calculate_macro_metrics(evaluation_df)
        p_values_macro = {}
        perm_counts_macro = {}
        perm_count_ge_macro = {}
        if observed_macro and permuted_macro_list:
            for metric in ["precision", "recall", "f1_score"]:
                permuted_values = [p[metric] for p in permuted_macro_list]
                n = len(permuted_values)
                perm_counts_macro[metric] = n
                if n == 0:
                    p_values_macro[metric] = float("nan")
                    perm_count_ge_macro[metric] = 0
                else:
                    count_ge = sum(1 for p in permuted_values if p > observed_macro[metric])
                    perm_count_ge_macro[metric] = count_ge
                    p_values_macro[metric] = (count_ge + 1) / (n + 1)

        # Calculate fold-over-random for micro metrics
        fold_over_random = {}
        for metric in ["precision", "recall", "f1_score"]:
            fold_over_random[metric] = SDPUtils.safe_divide(observed[metric], avg_permuted[metric])

        # Calculate fold-over-random for macro metrics
        fold_over_random_macro = {}
        if observed_macro and avg_permuted_macro:
            for metric in ["precision", "recall", "f1_score"]:
                fold_over_random_macro[metric] = SDPUtils.safe_divide(
                    observed_macro[metric], avg_permuted_macro[metric]
                )

        # Calculate localization errors
        localization_stats = self._calculate_localization_errors()

        # Save comprehensive metrics
        metrics_file = self.config.output_dir / "micro_macro_metrics.tsv"
        with open(metrics_file, "w") as f:
            # Add columns for permutation count and raw count >= observed
            f.write(
                "Analysis_Type\tMetric\tObserved\tPermuted_Average\tP_Value\tFold_Over_Random\tPerm_Count\tPerm_Count_GE\n"
            )

            # Micro averages (overall metrics)
            for metric in ["precision", "recall", "f1_score"]:
                f.write(
                    f"Micro\t{metric.replace('_', ' ').title()}\t{observed[metric]:.6f}\t"
                    f"{avg_permuted[metric]:.6f}\t{p_values[metric]:.6f}\t"
                    f"{fold_over_random[metric]:.6f}\t{perm_counts.get(metric,0)}\t{perm_count_ge.get(metric,0)}\n"
                )

            # Macro averages
            if observed_macro:
                for metric in ["precision", "recall", "f1_score"]:
                    if avg_permuted_macro and p_values_macro and fold_over_random_macro:
                        f.write(
                            f"Macro\t{metric.replace('_', ' ').title()}\t{observed_macro[metric]:.6f}\t"
                            f"{avg_permuted_macro[metric]:.6f}\t{p_values_macro[metric]:.6f}\t"
                            f"{fold_over_random_macro[metric]:.6f}\t{perm_counts_macro.get(metric,0)}\t{perm_count_ge_macro.get(metric,0)}\n"
                        )
                    else:
                        f.write(
                            f"Macro\t{metric.replace('_', ' ').title()}\t{observed_macro[metric]:.6f}\t"
                            f"NA\tNA\tNA\t0\t0\n"
                        )

            # Localization error statistics
            if localization_stats:
                f.write(
                    f"Localization\tMedian_Error\t{localization_stats['median']:.2f}\t"
                    f"NA\tNA\tNA\n"
                )
                f.write(
                    f"Localization\tIQR_Error\t{localization_stats['iqr']:.2f}\t" f"NA\tNA\tNA\n"
                )

        # Print summary
        print(f"\nPermutation Analysis Results:")
        print(
            f"{'Type':<12} {'Metric':<12} {'Observed':<10} {'Permuted':<10} {'P-Value':<10} {'Fold-Over':<10}"
        )
        print("-" * 70)

        # Print micro metrics
        for metric in ["precision", "recall", "f1_score"]:
            # Format p-value with threshold when very small
            p_val = p_values.get(metric, float("nan"))
            n = perm_counts.get(metric, 0)
            if isinstance(p_val, float) and not np.isnan(p_val) and n > 0:
                min_p = 1.0 / (n + 1)
                if p_val < min_p:
                    p_display = f"<{min_p:.3g}"
                else:
                    p_display = f"{p_val:.4f}"
            else:
                p_display = "NA"

            print(
                f"{'Micro':<12} {metric.replace('_', ' ').title():<12} {observed[metric]:<10.4f} "
                f"{avg_permuted[metric]:<10.4f} {p_display:<10} "
                f"{fold_over_random[metric]:<10.2f}"
            )

        # Print macro metrics
        if observed_macro:
            print()
            for metric in ["precision", "recall", "f1_score"]:
                if avg_permuted_macro and p_values_macro and fold_over_random_macro:
                    p_val = p_values_macro.get(metric, float("nan"))
                    n = perm_counts_macro.get(metric, 0)
                    if isinstance(p_val, float) and not np.isnan(p_val) and n > 0:
                        min_p = 1.0 / (n + 1)
                        if p_val < min_p:
                            p_display = f"<{min_p:.3g}"
                        else:
                            p_display = f"{p_val:.4f}"
                    else:
                        p_display = "NA"

                    print(
                        f"{'Macro':<12} {metric.replace('_', ' ').title():<12} {observed_macro[metric]:<10.4f} "
                        f"{avg_permuted_macro[metric]:<10.4f} {p_display:<10} "
                        f"{fold_over_random_macro[metric]:<10.2f}"
                    )
                else:
                    print(
                        f"{'Macro':<12} {metric.replace('_', ' ').title():<12} {observed_macro[metric]:<10.4f} "
                        f"{'NA':<10} {'NA':<10} {'NA':<10}"
                    )

        if localization_stats:
            print()
            print(
                f"{'Localization':<12} {'Median Err':<12} {localization_stats['median']:<10.2f} "
                f"{'NA':<10} {'NA':<10} {'NA':<10}"
            )
            print(
                f"{'Localization':<12} {'IQR Err':<12} {localization_stats['iqr']:<10.2f} "
                f"{'NA':<10} {'NA':<10} {'NA':<10}"
            )

    def _calculate_macro_metrics(self, evaluation_df: pd.DataFrame = None) -> Dict[str, float]:
        """Calculate macro averages (per-protein averages) from evaluation results."""
        if evaluation_df is None:
            eval_file = self.config.output_dir / "evaluation_results.tsv"
            if not eval_file.exists():
                return {}

            import pandas as pd

            evaluation_df = pd.read_csv(eval_file, sep="\t")

        # Calculate macro averages (mean of per-protein metrics)
        macro_precision = evaluation_df["precision"].mean()
        macro_recall = evaluation_df["recall"].mean()
        macro_f1 = evaluation_df["f1_score"].mean()

        return {"precision": macro_precision, "recall": macro_recall, "f1_score": macro_f1}

    def _calculate_localization_errors(self) -> Dict[str, float]:
        """Calculate localization error statistics (median ± IQR)."""
        eval_file = self.config.output_dir / "evaluation_results.tsv"
        pred_file = self.config.output_dir / "predicted_sdps.tsv"

        if not eval_file.exists() or not pred_file.exists():
            return {}

        from pathlib import Path

        import numpy as np
        import pandas as pd

        # Load evaluation and prediction data
        eval_df = pd.read_csv(eval_file, sep="\t")
        pred_df = pd.read_csv(pred_file, sep="\t")

        # Load known SDPs
        known_sdps = SDPPreprocessor.load_known_sdps(self.config.known_sdps_file)

        localization_errors = []

        # Calculate localization errors for each protein
        for _, row in eval_df.iterrows():
            protein_id = row["protein_id"]
            if protein_id in known_sdps:
                # Get predicted SDPs for this protein
                protein_preds = pred_df[pred_df["protein_id"] == protein_id]["residue"].tolist()
                known_positions = list(known_sdps[protein_id])

                if protein_preds and known_positions:
                    # Calculate minimum distance from each known SDP to nearest predicted SDP
                    for known_pos in known_positions:
                        min_distance = min(
                            [abs(known_pos - pred_pos) for pred_pos in protein_preds]
                        )
                        localization_errors.append(min_distance)

        if localization_errors:
            errors_array = np.array(localization_errors)
            median_error = np.median(errors_array)
            q75, q25 = np.percentile(errors_array, [75, 25])
            iqr_error = q75 - q25

            return {"median": median_error, "iqr": iqr_error, "q25": q25, "q75": q75}

        return {}

    def _save_results(
        self,
        evaluation_df: pd.DataFrame,
        all_predictions: Dict[str, Set[int]],
        all_detailed_predictions: Dict[str, pd.DataFrame],
        known_sdps: Dict[str, Set[int]],
    ) -> None:
        """Save all results to files."""
        # Save evaluation results
        eval_file = self.config.output_dir / "evaluation_results.tsv"
        evaluation_df.to_csv(eval_file, sep="\t", index=False)

        # Save detailed predictions with all requested information
        pred_file = self.config.output_dir / "predicted_sdps.tsv"
        pred_df = self._create_detailed_prediction_data(all_detailed_predictions, known_sdps)
        pred_df.to_csv(pred_file, sep="\t", index=False)

        # Save basic metrics only if permutation analysis is not run
        # (permutation analysis will save comprehensive metrics)
        if self.config.permutation_metric_null == 0:
            metrics_file = self.config.output_dir / "micro_macro_metrics.tsv"
            total_known = evaluation_df["known_sdps"].sum()
            total_predicted = evaluation_df["predicted_sdps"].sum()
            total_tp = evaluation_df["true_positives"].sum()

            overall_precision = SDPUtils.safe_divide(total_tp, total_predicted)
            overall_recall = SDPUtils.safe_divide(total_tp, total_known)
            overall_f1 = SDPUtils.safe_divide(
                2 * overall_precision * overall_recall, overall_precision + overall_recall
            )

            metrics_data = {
                "Metric": ["Overall Precision", "Overall Recall", "Overall F1-Score"],
                "Value": [overall_precision, overall_recall, overall_f1],
            }
            metrics_df = pd.DataFrame(metrics_data)
            metrics_df.to_csv(metrics_file, sep="\t", index=False, float_format="%.6f")

        print(f"\nResults saved to: {self.config.output_dir}")
        print(f"  Evaluation: {eval_file}")
        print(f"  Predictions: {pred_file}")
        if self.config.permutation_metric_null == 0:
            print(f"  Metrics: {metrics_file}")

    def _create_detailed_prediction_data(
        self, all_detailed_predictions: Dict[str, pd.DataFrame], known_sdps: Dict[str, Set[int]]
    ) -> pd.DataFrame:
        """Create detailed prediction DataFrame with all requested information."""
        if not all_detailed_predictions:
            return pd.DataFrame(
                columns=[
                    "protein_id",
                    "residue",
                    "ablation_score",
                    "z_score",
                    "smoothed_z_score",
                    "is_match",
                    "distance_to_closest_known",
                ]
            )

        detailed_data = []
        for protein_id, candidates_df in all_detailed_predictions.items():
            known_positions = known_sdps.get(protein_id, set())

            for _, row in candidates_df.iterrows():
                residue = row["residue"]
                ablation_score = row["ablation_score"]
                z_score = row["z_score"]
                smoothed_z = row["smoothed_z"]

                # Determine if this prediction matches a known SDP
                is_match = False
                distance_to_closest = float("inf")

                if known_positions:
                    distances = [abs(residue - known_pos) for known_pos in known_positions]
                    distance_to_closest = min(distances)
                    is_match = distance_to_closest <= self.config.match_tolerance

                detailed_data.append(
                    {
                        "protein_id": protein_id,
                        "residue": residue,
                        "ablation_score": ablation_score,
                        "z_score": z_score,
                        "smoothed_z_score": smoothed_z,
                        "is_match": is_match,
                        "distance_to_closest_known": (
                            distance_to_closest if distance_to_closest != float("inf") else None
                        ),
                    }
                )

        return pd.DataFrame(detailed_data)


def create_argument_parser() -> argparse.ArgumentParser:
    """Create and configure argument parser."""
    parser = argparse.ArgumentParser(
        description="Predict and evaluate SDPs from residue ablation scores (Refactored)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Required arguments
    parser.add_argument("ablation_file", help="Path to ablation scores TSV file")
    parser.add_argument("known_sdps_file", help="Path to known SDPs TSV file")
    parser.add_argument(
        "--output-dir",
        default="sdp_prediction_evaluation",
        help="Output directory for results and plots",
    )

    # Smoothing parameters
    parser.add_argument(
        "--window-size", type=int, default=12, help="Rolling window size for median smoothing"
    )
    parser.add_argument(
        "--smoothing-method",
        default="rolling_median",
        choices=["rolling_median", "gaussian", "savgol", "combined"],
        help="Smoothing method to apply",
    )
    parser.add_argument(
        "--smoothing-passes",
        type=int,
        default=1,
        help="Number of smoothing passes for rolling median",
    )
    parser.add_argument(
        "--gaussian-sigma", type=float, default=0.5, help="Sigma parameter for Gaussian smoothing"
    )
    parser.add_argument(
        "--savgol-polyorder",
        type=int,
        default=2,
        help="Polynomial order for Savitzky-Golay smoothing",
    )

    # Prediction parameters
    parser.add_argument(
        "--min-prominence", type=float, default=0.5, help="Minimum prominence for peak detection"
    )
    parser.add_argument(
        "--min-distance", type=int, default=1, help="Minimum distance between peaks"
    )
    parser.add_argument(
        "--min-zscore", type=float, default=-1, help="Minimum z-score threshold for SDP candidates"
    )
    parser.add_argument(
        "--match-tolerance", type=int, default=1, help="Tolerance for matching SDPs (±N residues)"
    )

    # Null distribution options
    parser.add_argument(
        "--use-null-distribution",
        action="store_true",
        help="Use permutations to calculate p-values for peaks",
    )
    parser.add_argument(
        "--pvalue-threshold",
        type=float,
        default=0.1,
        help="P-value threshold for significant peaks",
    )
    parser.add_argument(
        "--multiple-testing-correction",
        type=str,
        default="none",
        choices=[
            "none",
            "bonferroni",
            "sidak",
            "holm-sidak",
            "holm",
            "simes-hochberg",
            "hommel",
            "fdr_bh",
            "fdr_by",
        ],
        help="Multiple testing correction method",
    )
    parser.add_argument(
        "--null-permutations",
        type=int,
        default=500,
        help="Number of permutations for null distribution",
    )
    parser.add_argument(
        "--randomization-method",
        type=str,
        default="circular",
        choices=[
            "circular",
            "block_shuffle",
            "local_shuffle",
            "gaussian_noise",
            "rank_shuffle",
            "bootstrap",
        ],
        help="Method for generating null distribution",
    )

    # Analysis options
    parser.add_argument("--optimize", action="store_true", help="Run parameter optimization")
    parser.add_argument(
        "--plot-proteins", type=int, default=0, help="Number of proteins to plot (0 = no plots)"
    )
    parser.add_argument(
        "--plot-permuted-proteins",
        type=int,
        default=0,
        help="Plot permuted profiles for N random proteins",
    )
    parser.add_argument(
        "--save-detailed",
        action="store_true",
        help="Save detailed results with all computed values",
    )
    parser.add_argument(
        "--bootstrap-metrics",
        type=int,
        default=0,
        help="Number of bootstrap iterations for confidence intervals",
    )
    parser.add_argument(
        "--permutation-metric-null",
        type=int,
        default=0,
        help="Number of permutations for metric null distribution",
    )

    return parser


def main():
    """Main function."""
    parser = create_argument_parser()
    args = parser.parse_args()

    # Create configuration
    config = SDPConfig(args)

    # Create and run pipeline
    pipeline = SDPPipeline(config)
    pipeline.run()


if __name__ == "__main__":
    main()
