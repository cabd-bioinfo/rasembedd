#!/usr/bin/env python3
"""
Linear Probe for Cluster Prediction

This module implements a linear probe using multinomial logistic regression
to predict cluster assignments from protein embeddings. It uses leave-one-out
cross-validation for robust evaluation.
"""

import argparse
import os
import pickle
import sys
import warnings
from typing import Dict, List, Optional, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from sklearn.model_selection import KFold, LeaveOneOut
from sklearn.preprocessing import LabelEncoder, StandardScaler, normalize

# Suppress sklearn warnings for cleaner output
warnings.filterwarnings("ignore", category=UserWarning)


class EmbeddingNormalizer:
    """Handles normalization methods for embeddings - same as clustering_evaluation.py"""

    @staticmethod
    def normalize_pipeline(
        embeddings: np.ndarray,
        *,
        center: bool = False,
        scale: bool = False,
        pca_components: float | int = 0,
        l2: bool = True,
        random_state: int = 42,
    ) -> tuple[np.ndarray, dict]:
        """Three-step normalization pipeline matching clustering_evaluation.py.

        1) Standard normalization with optional centering and scaling
        2) PCA dimensionality reduction (components=int or variance in (0,1]); 0 disables PCA
        3) Optional L2 per-sample normalization

        Returns:
            tuple: (normalized_embeddings, info_dict)
            info_dict contains PCA information like n_components_ and explained_variance_ratio_
        """
        X = embeddings
        info = {}

        # Step 1: Standardization
        if center or scale:
            scaler = StandardScaler(with_mean=center, with_std=scale)
            X = scaler.fit_transform(X)
            info["standardization"] = {"center": center, "scale": scale}
            info["scaler"] = scaler
            if center:
                info["scaler_mean"] = scaler.mean_
            if scale:
                info["scaler_scale"] = scaler.scale_

        # Step 2: PCA reduction
        if isinstance(pca_components, (int, float)) and pca_components != 0:
            # Interpret pca_components as follows:
            # - int (>=1): number of components
            # - float > 1: treat as integer component count (e.g., 256.0 -> 256)
            # - 0 < float < 1: fraction of variance to retain
            # - float == 1.0: keep all components (use None to avoid invalid 1.0)
            if isinstance(pca_components, int):
                n_components: int | float | None = max(int(pca_components), 1)
            else:  # float
                if pca_components > 1.0:
                    n_components = max(int(round(pca_components)), 1)
                elif 0.0 < pca_components < 1.0:
                    n_components = pca_components
                else:  # pca_components == 1.0 (keep all components)
                    n_components = None

            # Cap integer components at maximum possible value to avoid errors
            if isinstance(n_components, int):
                max_components = min(X.shape[0] - 1, X.shape[1])  # -1 for degrees of freedom
                if n_components > max_components:
                    print(
                        f"Note: Reducing PCA components from {n_components} to {max_components} (max possible with {X.shape[0]} samples)"
                    )
                    n_components = max_components

            pca = PCA(n_components=n_components, random_state=random_state)
            X = pca.fit_transform(X)

            # Store PCA information
            info["pca_n_components"] = pca.n_components_
            info["pca_explained_variance_ratio_sum"] = pca.explained_variance_ratio_.sum()
            info["pca_requested_components"] = pca_components
            info["pca_transform"] = pca

        # Step 3: L2 normalization
        if l2:
            X = normalize(X, norm="l2", axis=1)
            info["l2_normalization"] = True

        return X, info


class LinearProbe:
    """
    Linear probe for predicting cluster assignments from embeddings.

    Uses multinomial logistic regression with L2 regularization and
    leave-one-out cross-validation for evaluation.
    """

    def __init__(
        self,
        regularization: float = 1.0,
        max_iter: int = 1000,
        random_state: int = 42,
        # Normalization pipeline parameters
        norm_center: bool = True,
        norm_scale: bool = False,
        norm_pca_components: Optional[Union[int, float]] = 0.95,
        norm_l2: bool = True,
        # Evaluation parameters
        cv: Union[str, int] = "loo",
    ):
        """
        Initialize the linear probe.

        Args:
            regularization: L2 regularization strength (higher = more regularization)
            max_iter: Maximum iterations for solver
            random_state: Random seed for reproducibility
            norm_center: Whether to center features (mean=0)
            norm_scale: Whether to scale features (std=1)
            norm_pca_components: PCA components (int for count, float in (0,1] for variance, 0 to disable)
            norm_l2: Whether to apply L2 normalization at the end
        """
        self.regularization = regularization
        self.max_iter = max_iter
        self.random_state = random_state

        # Normalization pipeline parameters
        self.norm_center = norm_center
        self.norm_scale = norm_scale
        self.norm_pca_components = norm_pca_components
        self.norm_l2 = norm_l2

        # Evaluation parameters
        self.cv = cv

        # Initialize components
        self.label_encoder = LabelEncoder()
        self.model = LogisticRegression(
            C=1.0 / regularization,  # sklearn uses inverse regularization
            max_iter=max_iter,
            random_state=random_state,
            solver="lbfgs",
        )

        # Store results
        self.is_fitted = False
        self.feature_names = None
        self.class_names = None
        self.normalization_info = None

    def load_data(
        self,
        embedding_file: str,
        cluster_file: str,
        protein_id_col: str = "protein_id",
        cluster_cols: Union[str, List[str]] = "cluster",
    ) -> Dict[str, Tuple[np.ndarray, np.ndarray, List[str]]]:
        """
        Load embeddings and cluster assignments.

        Args:
            embedding_file: Path to pickle file with embeddings dict
            cluster_file: Path to TSV file with cluster assignments
            protein_id_col: Column name for protein IDs
            cluster_cols: Cluster column name(s) or "all" for all non-ID columns

        Returns:
            Dictionary mapping cluster column names to (embeddings, cluster_labels, protein_ids) tuples
        """
        print(f"Loading embeddings from {embedding_file}")
        with open(embedding_file, "rb") as f:
            embeddings_dict = pickle.load(f)

        print(f"Loading cluster assignments from {cluster_file}")
        cluster_df = pd.read_csv(cluster_file, sep="\t")

        print(f"Loaded {len(embeddings_dict)} embeddings and {len(cluster_df)} cluster assignments")

        # Validate required columns
        if protein_id_col not in cluster_df.columns:
            available_cols = ", ".join(cluster_df.columns.tolist())
            raise ValueError(
                f"Protein ID column '{protein_id_col}' not found. Available: {available_cols}"
            )

        # Determine which cluster columns to process
        if cluster_cols == "all":
            available_cols = [col for col in cluster_df.columns if col != protein_id_col]
            cluster_cols_list = available_cols
            print(f"Processing all cluster columns: {cluster_cols_list}")
        elif isinstance(cluster_cols, str):
            cluster_cols_list = [cluster_cols]

        # Validate cluster columns exist
        missing_cols = [col for col in cluster_cols_list if col not in cluster_df.columns]
        if missing_cols:
            available_cols = ", ".join(cluster_df.columns.tolist())
            raise ValueError(
                f"Cluster columns not found: {missing_cols}. Available: {available_cols}"
            )

        print(f"Processing cluster columns: {cluster_cols_list}")

        # Find common proteins
        embedding_ids = set(embeddings_dict.keys())
        cluster_ids = set(cluster_df[protein_id_col].values)

        common_ids = embedding_ids & cluster_ids
        print(f"Found {len(common_ids)} proteins with both embeddings and cluster assignments")
        print(f"  Embeddings only: {len(embedding_ids - cluster_ids)}")
        print(f"  Clusters only: {len(cluster_ids - embedding_ids)}")

        if len(common_ids) == 0:
            raise ValueError("No common proteins found between embeddings and cluster assignments!")

        # Prepare results for each cluster column
        results = {}

        for cluster_col in cluster_cols_list:
            print(f"\nProcessing cluster column: {cluster_col}")

            # Filter out rows with missing cluster assignments for this column
            col_subset = cluster_df.dropna(subset=[cluster_col])
            col_cluster_ids = set(col_subset[protein_id_col].values)
            col_common_ids = embedding_ids & col_cluster_ids

            if len(col_common_ids) == 0:
                print(f"Warning: No common proteins found for cluster column '{cluster_col}'")
            cluster_cols_list = [cluster_cols]

            # Create aligned data for this cluster column
            protein_ids = sorted(list(col_common_ids))
            embeddings = np.array([embeddings_dict[pid] for pid in protein_ids])

            # Create cluster lookup and get labels
            cluster_lookup = dict(zip(col_subset[protein_id_col], col_subset[cluster_col]))
            cluster_labels = np.array([cluster_lookup[pid] for pid in protein_ids])

            print(
                f"Final dataset for {cluster_col}: {len(protein_ids)} proteins, {embeddings.shape[1]} features"
            )
            unique_labels, counts = np.unique(cluster_labels, return_counts=True)
            print(f"Cluster distribution: {dict(zip(unique_labels, counts))}")

            results[cluster_col] = (embeddings, cluster_labels, protein_ids)

        if not results:
            raise ValueError("No valid cluster columns found with data")

        return results

    def evaluate_cv(self, X: np.ndarray, y: np.ndarray, protein_ids: List[str]) -> Dict:
        """
        Perform cross-validation based on the cv parameter.

        Args:
            X: Feature matrix (n_samples, n_features)
            y: Target labels (n_samples,)
            protein_ids: List of protein IDs

        Returns:
            Dictionary with evaluation metrics
        """
        if self.cv == "none":
            return self._evaluate_training_only(X, y, protein_ids)
        elif self.cv == "loo":
            return self._evaluate_loo(X, y, protein_ids)
        elif isinstance(self.cv, int) and self.cv > 1:
            return self._evaluate_kfold(X, y, protein_ids, self.cv)
        else:
            raise ValueError(f"Invalid cv option: {self.cv}. Use 'none', 'loo', or integer > 1")

    def _evaluate_training_only(self, X: np.ndarray, y: np.ndarray, protein_ids: List[str]) -> Dict:
        """
        Calculate training metrics only (no cross-validation).

        Args:
            X: Feature matrix (n_samples, n_features)
            y: Target labels (n_samples,)
            protein_ids: List of protein IDs

        Returns:
            Dictionary with training metrics
        """
        print("Performing training evaluation only (no cross-validation)")
        print("\nWARNING: Calculating metrics on training data (not cross-validated)")
        print("These metrics may be overly optimistic and should not be used for model evaluation")

        # Encode labels
        y_encoded = self.label_encoder.fit_transform(y)
        self.class_names = self.label_encoder.classes_

        # Apply normalization pipeline
        X_norm, norm_info = EmbeddingNormalizer.normalize_pipeline(
            X,
            center=self.norm_center,
            scale=self.norm_scale,
            pca_components=self.norm_pca_components,
            l2=self.norm_l2,
            random_state=self.random_state,
        )

        # Store normalization info
        self.normalization_info = norm_info

        # Train model
        model = LogisticRegression(
            C=1.0 / self.regularization,
            max_iter=self.max_iter,
            random_state=self.random_state,
            solver="lbfgs",
        )

        model.fit(X_norm, y_encoded)

        # Get predictions on training data
        y_pred = model.predict(X_norm)
        y_prob = model.predict_proba(X_norm)

        # Calculate metrics
        from sklearn.metrics import (
            accuracy_score,
            classification_report,
            confusion_matrix,
            f1_score,
        )

        accuracy = accuracy_score(y_encoded, y_pred)
        f1_macro = f1_score(y_encoded, y_pred, average="macro")
        f1_micro = f1_score(y_encoded, y_pred, average="micro")
        f1_weighted = f1_score(y_encoded, y_pred, average="weighted")

        print(f"\nTraining Metrics (NOT cross-validated):")
        print(f"  Accuracy: {accuracy:.4f}")
        print(f"  F1 Score (macro): {f1_macro:.4f}")
        print(f"  F1 Score (micro): {f1_micro:.4f}")
        print(f"  F1 Score (weighted): {f1_weighted:.4f}")

        if self.normalization_info:
            print(f"\nNormalization Info:")
            if "pca_n_components" in self.normalization_info:
                print(f"  PCA components: {self.normalization_info['pca_n_components']}")
                print(
                    f"  PCA explained variance: {self.normalization_info['pca_explained_variance_ratio_sum']:.4f}"
                )

        # Classification report
        class_report = classification_report(
            y_encoded, y_pred, target_names=self.class_names, output_dict=True
        )

        # Confusion matrix
        conf_matrix = confusion_matrix(y_encoded, y_pred)

        # Store detailed results
        results = {
            "cv_type": "training_only",
            "accuracy": accuracy,
            "f1_macro": f1_macro,
            "f1_micro": f1_micro,
            "f1_weighted": f1_weighted,
            "classification_report": class_report,
            "confusion_matrix": conf_matrix,
            "y_true": y_encoded,
            "y_pred": y_pred,
            "y_prob": y_prob,
            "test_proteins": protein_ids,  # Actually training proteins
            "class_names": self.class_names,
            "normalization_info": self.normalization_info,
        }

        return results

    def _evaluate_kfold(
        self, X: np.ndarray, y: np.ndarray, protein_ids: List[str], n_folds: int
    ) -> Dict:
        """
        Perform k-fold cross-validation.

        Args:
            X: Feature matrix (n_samples, n_features)
            y: Target labels (n_samples,)
            protein_ids: List of protein IDs
            n_folds: Number of folds

        Returns:
            Dictionary with evaluation metrics
        """
        print(f"Performing {n_folds}-fold cross-validation...")
        print(
            f"Normalization pipeline: center={self.norm_center}, scale={self.norm_scale}, "
            f"pca_components={self.norm_pca_components}, l2={self.norm_l2}"
        )

        # Encode labels
        y_encoded = self.label_encoder.fit_transform(y)
        self.class_names = self.label_encoder.classes_

        kfold = KFold(n_splits=n_folds, shuffle=True, random_state=self.random_state)

        # Store predictions
        y_true = []
        y_pred = []
        y_prob = []
        test_proteins = []
        fold_scores = []

        print(f"Running {n_folds} fold iterations...")

        for fold, (train_idx, test_idx) in enumerate(kfold.split(X)):
            print(f"  Fold {fold+1}/{n_folds}")

            # Split data
            X_train, X_test = X[train_idx], X[test_idx]
            y_train, y_test = y_encoded[train_idx], y_encoded[test_idx]

            # Apply normalization pipeline to training data
            X_train_norm, norm_info = EmbeddingNormalizer.normalize_pipeline(
                X_train,
                center=self.norm_center,
                scale=self.norm_scale,
                pca_components=self.norm_pca_components,
                l2=self.norm_l2,
                random_state=self.random_state,
            )

            # Apply same transformation to test data using training statistics
            X_test_norm = X_test.copy()
            if self.norm_center and "scaler_mean" in norm_info:
                X_test_norm = X_test_norm - norm_info["scaler_mean"]

            if self.norm_scale and "scaler_scale" in norm_info:
                X_test_norm = X_test_norm / norm_info["scaler_scale"]

            if self.norm_pca_components and "pca_transform" in norm_info:
                X_test_norm = norm_info["pca_transform"].transform(X_test_norm)

            if self.norm_l2:
                from sklearn.preprocessing import normalize

                X_test_norm = normalize(X_test_norm, norm="l2")

            # Train model
            model = LogisticRegression(
                C=1.0 / self.regularization,
                max_iter=self.max_iter,
                random_state=self.random_state,
                solver="lbfgs",
            )

            model.fit(X_train_norm, y_train)

            # Predict
            pred = model.predict(X_test_norm)
            prob = model.predict_proba(X_test_norm)

            # Calculate fold accuracy
            from sklearn.metrics import accuracy_score

            fold_acc = accuracy_score(y_test, pred)
            fold_scores.append(fold_acc)
            print(f"    Fold {fold+1} accuracy: {fold_acc:.4f}")

            # Store results
            y_true.extend(y_test)
            y_pred.extend(pred)

            # Handle probability arrays - pad with zeros for missing classes
            n_classes = len(self.class_names)
            prob_padded = np.zeros((len(pred), n_classes))

            # Get the classes that this model was trained on
            model_classes = model.classes_

            # Map model classes to overall class indices
            for i, model_class in enumerate(model_classes):
                overall_idx = np.where(self.class_names == model_class)[0]
                if len(overall_idx) > 0:
                    prob_padded[:, overall_idx[0]] = prob[:, i]

            y_prob.extend(prob_padded)
            test_proteins.extend([protein_ids[i] for i in test_idx])

        # Store normalization info from last fold for reference
        self.normalization_info = norm_info

        # Convert to arrays
        y_true = np.array(y_true)
        y_pred = np.array(y_pred)
        y_prob = np.array(y_prob)

        # Calculate metrics
        from sklearn.metrics import (
            accuracy_score,
            classification_report,
            confusion_matrix,
            f1_score,
        )

        accuracy = accuracy_score(y_true, y_pred)
        f1_macro = f1_score(y_true, y_pred, average="macro")
        f1_micro = f1_score(y_true, y_pred, average="micro")
        f1_weighted = f1_score(y_true, y_pred, average="weighted")

        print(f"\n{n_folds}-Fold Cross-Validation Results:")
        print(f"  Accuracy: {accuracy:.4f} (±{np.std(fold_scores):.4f})")
        print(f"  F1 Score (macro): {f1_macro:.4f}")
        print(f"  F1 Score (micro): {f1_micro:.4f}")
        print(f"  F1 Score (weighted): {f1_weighted:.4f}")

        if self.normalization_info:
            print(f"\nNormalization Info:")
            if "pca_n_components" in self.normalization_info:
                print(f"  PCA components: {self.normalization_info['pca_n_components']}")
                print(
                    f"  PCA explained variance: {self.normalization_info['pca_explained_variance_ratio_sum']:.4f}"
                )

        # Classification report
        class_report = classification_report(
            y_true, y_pred, target_names=self.class_names, output_dict=True
        )

        # Confusion matrix
        conf_matrix = confusion_matrix(y_true, y_pred)

        # Store detailed results
        results = {
            "cv_type": f"{n_folds}_fold",
            "accuracy": accuracy,
            "accuracy_std": np.std(fold_scores),
            "fold_scores": fold_scores,
            "f1_macro": f1_macro,
            "f1_micro": f1_micro,
            "f1_weighted": f1_weighted,
            "classification_report": class_report,
            "confusion_matrix": conf_matrix,
            "y_true": y_true,
            "y_pred": y_pred,
            "y_prob": y_prob,
            "test_proteins": test_proteins,
            "class_names": self.class_names,
            "normalization_info": self.normalization_info,
        }

        return results

    def _evaluate_loo(self, X: np.ndarray, y: np.ndarray, protein_ids: List[str]) -> Dict:
        """
        Perform leave-one-out cross-validation.

        Args:
            X: Feature matrix (n_samples, n_features)
            y: Target labels (n_samples,)
            protein_ids: List of protein IDs

        Returns:
            Dictionary with evaluation metrics
        """
        print("Performing leave-one-out cross-validation...")
        print(
            f"Normalization pipeline: center={self.norm_center}, scale={self.norm_scale}, "
            f"pca_components={self.norm_pca_components}, l2={self.norm_l2}"
        )

        # Encode labels
        y_encoded = self.label_encoder.fit_transform(y)
        self.class_names = self.label_encoder.classes_

        loo = LeaveOneOut()
        n_samples = len(X)

        # Store predictions
        y_true = []
        y_pred = []
        y_prob = []
        test_proteins = []

        print(f"Running {n_samples} LOO iterations...")

        for i, (train_idx, test_idx) in enumerate(loo.split(X)):
            if (i + 1) % 100 == 0 or i == 0:
                print(f"  Iteration {i+1}/{n_samples}")
            # Split data
            X_train, X_test = X[train_idx], X[test_idx]
            y_train, y_test = y_encoded[train_idx], y_encoded[test_idx]
            # Apply normalization pipeline to training data
            X_train_norm, norm_info = EmbeddingNormalizer.normalize_pipeline(
                X_train,
                center=self.norm_center,
                scale=self.norm_scale,
                pca_components=self.norm_pca_components,
                l2=self.norm_l2,
                random_state=self.random_state,
            )
            # Apply same transformation to test data using training statistics
            X_test_norm = X_test.copy()
            if self.norm_center and "scaler_mean" in norm_info:
                X_test_norm = X_test_norm - norm_info["scaler_mean"]
            if self.norm_scale and "scaler_scale" in norm_info:
                X_test_norm = X_test_norm / norm_info["scaler_scale"]
            if self.norm_pca_components and "pca_transform" in norm_info:
                X_test_norm = norm_info["pca_transform"].transform(X_test_norm)
            if self.norm_l2:
                X_test_norm = normalize(X_test_norm, norm="l2")
            # Train model
            model = LogisticRegression(
                C=1.0 / self.regularization,
                max_iter=self.max_iter,
                random_state=self.random_state,
                solver="lbfgs",
            )
            model.fit(X_train_norm, y_train)
            # Predict
            pred = model.predict(X_test_norm)[0]
            # Align probabilities to all classes
            model_classes = model.classes_  # These are integer-encoded
            probas = model.predict_proba(X_test_norm)[0]
            prob_vec = np.zeros(len(self.class_names))
            # Convert model classes back to string labels using label_encoder
            for idx, cls in enumerate(model_classes):
                try:
                    # cls is an integer, convert to string label
                    string_label = self.label_encoder.inverse_transform([cls])[0]
                    # Find index in self.class_names
                    match = np.where(self.class_names == string_label)[0]
                    if len(match) > 0:
                        class_idx = match[0]
                        prob_vec[class_idx] = probas[idx]
                except Exception:
                    pass
            # Store results
            y_true.append(y_test[0])
            y_pred.append(pred)
            y_prob.append(prob_vec)
            test_proteins.append(protein_ids[test_idx[0]])

        # Store normalization info from last iteration for reference
        self.normalization_info = norm_info

        # Convert to arrays
        y_true = np.array(y_true)
        y_pred = np.array(y_pred)
        y_prob = np.array(y_prob)

        # Calculate metrics
        accuracy = accuracy_score(y_true, y_pred)
        f1_macro = f1_score(y_true, y_pred, average="macro")
        f1_micro = f1_score(y_true, y_pred, average="micro")
        f1_weighted = f1_score(y_true, y_pred, average="weighted")

        print(f"\nLeave-One-Out Results:")
        print(f"  Accuracy: {accuracy:.4f}")
        print(f"  F1 Score (macro): {f1_macro:.4f}")
        print(f"  F1 Score (micro): {f1_micro:.4f}")
        print(f"  F1 Score (weighted): {f1_weighted:.4f}")

        if self.normalization_info:
            print(f"\nNormalization Info:")
            if "pca_n_components" in self.normalization_info:
                print(f"  PCA components: {self.normalization_info['pca_n_components']}")
                print(
                    f"  PCA explained variance: {self.normalization_info['pca_explained_variance_ratio_sum']:.4f}"
                )

        # Classification report
        class_report = classification_report(
            y_true, y_pred, target_names=self.class_names, output_dict=True
        )

        # Confusion matrix
        conf_matrix = confusion_matrix(y_true, y_pred)

        # Store detailed results
        results = {
            "cv_type": "loo",
            "accuracy": accuracy,
            "f1_macro": f1_macro,
            "f1_micro": f1_micro,
            "f1_weighted": f1_weighted,
            "classification_report": class_report,
            "confusion_matrix": conf_matrix,
            "y_true": y_true,
            "y_pred": y_pred,
            "y_prob": y_prob,
            "test_proteins": test_proteins,
            "class_names": self.class_names,
            "normalization_info": self.normalization_info,
        }

        return results

    def fit_full_model(self, X: np.ndarray, y: np.ndarray) -> None:
        """
        Fit the model on the full dataset.

        Args:
            X: Feature matrix
            y: Target labels
        """
        print("Fitting full model...")
        print(
            f"Normalization pipeline: center={self.norm_center}, scale={self.norm_scale}, "
            f"pca_components={self.norm_pca_components}, l2={self.norm_l2}"
        )

        # Encode labels
        y_encoded = self.label_encoder.fit_transform(y)
        self.class_names = self.label_encoder.classes_

        # Apply normalization pipeline
        X_norm, norm_info = EmbeddingNormalizer.normalize_pipeline(
            X,
            center=self.norm_center,
            scale=self.norm_scale,
            pca_components=self.norm_pca_components,
            l2=self.norm_l2,
            random_state=self.random_state,
        )

        # Store normalization info
        self.normalization_info = norm_info

        # Fit model
        self.model.fit(X_norm, y_encoded)
        self.is_fitted = True

        print(f"Model fitted with {len(self.class_names)} classes: {self.class_names}")
        if self.normalization_info:
            if "pca_n_components" in self.normalization_info:
                print(
                    f"After normalization: {self.normalization_info['pca_n_components']} components"
                )
                print(
                    f"PCA explained variance: {self.normalization_info['pca_explained_variance_ratio_sum']:.4f}"
                )

    def save_model(self, output_dir: str, base_name: str) -> None:
        """
        Save the trained model and all associated components for later use.

        Args:
            output_dir: Directory to save model
            base_name: Base name for model files
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before saving!")

        os.makedirs(output_dir, exist_ok=True)

        # Create model package with all necessary components
        model_package = {
            "model": self.model,
            "label_encoder": self.label_encoder,
            "class_names": self.class_names,
            "normalization_params": {
                "center": self.norm_center,
                "scale": self.norm_scale,
                "pca_components": self.norm_pca_components,
                "l2": self.norm_l2,
                "random_state": self.random_state,
            },
            "normalization_info": self.normalization_info,
            "is_fitted": self.is_fitted,
        }

        model_file = os.path.join(output_dir, f"{base_name}_model.pkl")
        with open(model_file, "wb") as f:
            pickle.dump(model_package, f)

        print(f"Saved complete model to {model_file}")

        # Also save model components separately for transparency
        components_dir = os.path.join(output_dir, f"{base_name}_model_components")
        os.makedirs(components_dir, exist_ok=True)

        # Save individual components
        with open(os.path.join(components_dir, "logistic_model.pkl"), "wb") as f:
            pickle.dump(self.model, f)

        with open(os.path.join(components_dir, "label_encoder.pkl"), "wb") as f:
            pickle.dump(self.label_encoder, f)

        with open(os.path.join(components_dir, "normalization_params.pkl"), "wb") as f:
            pickle.dump(model_package["normalization_params"], f)

        if self.normalization_info:
            with open(os.path.join(components_dir, "normalization_info.pkl"), "wb") as f:
                pickle.dump(self.normalization_info, f)

        print(f"Saved model components to {components_dir}/")

    @classmethod
    def load_model(cls, model_file: str) -> "LinearProbe":
        """
        Load a previously saved model.

        Args:
            model_file: Path to saved model file

        Returns:
            Loaded LinearProbe instance
        """
        with open(model_file, "rb") as f:
            model_package = pickle.load(f)

        # Create new instance
        probe = cls(
            regularization=1.0,  # Will be overridden by loaded model
            norm_center=model_package["normalization_params"]["center"],
            norm_scale=model_package["normalization_params"]["scale"],
            norm_pca_components=model_package["normalization_params"]["pca_components"],
            norm_l2=model_package["normalization_params"]["l2"],
            random_state=model_package["normalization_params"]["random_state"],
        )

        # Restore saved state
        probe.model = model_package["model"]
        probe.label_encoder = model_package["label_encoder"]
        probe.class_names = model_package["class_names"]
        probe.normalization_info = model_package["normalization_info"]
        probe.is_fitted = model_package["is_fitted"]

        print(f"Loaded model from {model_file}")
        print(f"Model classes: {probe.class_names}")

        return probe

    def get_feature_importance(self) -> np.ndarray:
        """
        Get feature importance (coefficients) from the fitted model.

        Returns:
            Feature importance matrix (n_classes, n_features)
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted first!")

        return self.model.coef_

    def save_results(
        self, results: Dict, output_dir: str, base_name: str, save_model: bool = True
    ) -> None:
        """
        Save evaluation results to files.

        Args:
            results: Results dictionary from evaluate_loo or minimal results when LOO skipped
            output_dir: Output directory
            base_name: Base name for output files
            save_model: Whether to save the trained model
        """
        os.makedirs(output_dir, exist_ok=True)

        if results.get("skipped_loo", False):
            if results.get("training_metrics", False):
                print("LOO evaluation was skipped - saving training metrics (NOT cross-validated)")

                # Save training predictions with warning
                predictions_df = pd.DataFrame(
                    {
                        "protein_id": results["test_proteins"],  # Actually training proteins
                        "true_label": self.label_encoder.inverse_transform(results["y_true"]),
                        "predicted_label": self.label_encoder.inverse_transform(results["y_pred"]),
                        "correct": results["y_true"] == results["y_pred"],
                    }
                )

                # Add probability columns
                for i, class_name in enumerate(self.class_names):
                    predictions_df[f"prob_{class_name}"] = results["y_prob"][:, i]

                pred_file = os.path.join(output_dir, f"{base_name}_training_predictions.tsv")
                predictions_df.to_csv(pred_file, sep="\t", index=False)
                print(f"Saved training predictions to {pred_file}")

                # Save metrics summary with warning
                metrics_df = pd.DataFrame(
                    {
                        "metric": ["accuracy", "f1_macro", "f1_micro", "f1_weighted"],
                        "value": [
                            results["accuracy"],
                            results["f1_macro"],
                            results["f1_micro"],
                            results["f1_weighted"],
                        ],
                        "warning": [
                            "WARNING: These metrics are calculated on training data only and are NOT cross-validated."
                        ]
                        * 4,
                    }
                )
                metrics_file = os.path.join(output_dir, f"{base_name}_training_metrics.tsv")
                metrics_df.to_csv(metrics_file, sep="\t", index=False)
                print(f"Saved training metrics to {metrics_file}")

                # Save classification report with warning row at the top
                report_df = pd.DataFrame(results["classification_report"]).T
                # Insert warning row at the top, matching all columns
                warning_data = {}
                for col in report_df.columns:
                    if col == "precision":
                        warning_data[col] = [
                            "WARNING: This report is calculated on training data only and is NOT cross-validated."
                        ]
                    else:
                        warning_data[col] = [""]
                warning_row = pd.DataFrame(warning_data, index=["WARNING"])
                report_df = pd.concat([warning_row, report_df], axis=0)
                report_file = os.path.join(
                    output_dir, f"{base_name}_training_classification_report.tsv"
                )
                report_df.to_csv(report_file, sep="\t")
                print(f"Saved training classification report to {report_file}")

                # Plot and save confusion matrix with warning
                import matplotlib.pyplot as plt
                import seaborn as sns

                plt.figure(figsize=(8, 6))
                sns.heatmap(
                    results["confusion_matrix"],
                    annot=True,
                    fmt="d",
                    cmap="Blues",
                    xticklabels=self.class_names,
                    yticklabels=self.class_names,
                )
                plt.title("Confusion Matrix (Training Data - NOT Cross-Validated)")
                plt.xlabel("Predicted Label")
                plt.ylabel("True Label")
                plt.tight_layout()

                conf_plot = os.path.join(output_dir, f"{base_name}_training_confusion_matrix.pdf")
                plt.savefig(conf_plot, dpi=300, bbox_inches="tight")
                plt.close()
                print(f"Saved training confusion matrix plot to {conf_plot}")

            else:
                print("LOO evaluation was skipped - saving minimal results")

                # Save minimal info file
                info_data = {
                    "analysis": "linear_probe",
                    "loo_evaluation": "skipped",
                    "n_classes": (
                        len(results["class_names"])
                        if results["class_names"] is not None
                        else "unknown"
                    ),
                    "class_names": (
                        list(results["class_names"])
                        if results["class_names"] is not None
                        else "unknown"
                    ),
                }

                if results["normalization_info"]:
                    info_data.update(
                        {
                            "normalization_applied": True,
                            "pca_components": results["normalization_info"].get(
                                "pca_n_components", "none"
                            ),
                            "pca_variance_explained": results["normalization_info"].get(
                                "pca_explained_variance_ratio_sum", "none"
                            ),
                        }
                    )

                info_df = pd.DataFrame([info_data])
                info_file = os.path.join(output_dir, f"{base_name}_info.tsv")
                info_df.to_csv(info_file, sep="\t", index=False)
                print(f"Saved analysis info to {info_file}")

        else:
            # Save detailed predictions
            # Use inverse_transform only if y_true/y_pred are integer-encoded, else use as-is
            try:
                true_labels = self.label_encoder.inverse_transform(results["y_true"])
                pred_labels = self.label_encoder.inverse_transform(results["y_pred"])
            except Exception:
                true_labels = results["y_true"]
                pred_labels = results["y_pred"]
            predictions_df = pd.DataFrame(
                {
                    "protein_id": results["test_proteins"],
                    "true_label": true_labels,
                    "predicted_label": pred_labels,
                    "correct": true_labels == pred_labels,
                }
            )
            # Add probability columns, robust to string/int class names
            for i, class_name in enumerate(self.class_names):
                col_name = f"prob_{class_name}"
                predictions_df[col_name] = results["y_prob"][:, i]
            pred_file = os.path.join(output_dir, f"{base_name}_predictions.tsv")
            predictions_df.to_csv(pred_file, sep="\t", index=False)
            print(f"Saved predictions to {pred_file}")

            # Determine if this is a training-only (no CV) run
            is_training_only = (
                results.get("skipped_loo", False)
                or results.get("cv_method", "").lower() == "none"
                or results.get("cv_type", "").lower() == "training_only"
            )

            # Save metrics summary, with warning if training-only
            if is_training_only:
                metrics_df = pd.DataFrame(
                    {
                        "metric": ["accuracy", "f1_macro", "f1_micro", "f1_weighted"],
                        "value": [
                            results["accuracy"],
                            results["f1_macro"],
                            results["f1_micro"],
                            results["f1_weighted"],
                        ],
                        "warning": [
                            "WARNING: These metrics are calculated on training data only and are NOT cross-validated."
                        ]
                        * 4,
                    }
                )
            else:
                metrics_df = pd.DataFrame(
                    {
                        "metric": ["accuracy", "f1_macro", "f1_micro", "f1_weighted"],
                        "value": [
                            results["accuracy"],
                            results["f1_macro"],
                            results["f1_micro"],
                            results["f1_weighted"],
                        ],
                    }
                )
            metrics_file = os.path.join(output_dir, f"{base_name}_metrics.tsv")
            metrics_df.to_csv(metrics_file, sep="\t", index=False)
            print(f"Saved metrics to {metrics_file}")

            # Save classification report, with warning row if training-only
            report_df = pd.DataFrame(results["classification_report"]).T
            # ...existing code...
            if is_training_only:
                warning_data = {}
                for col in report_df.columns:
                    if col == "precision":
                        warning_data[col] = [
                            "WARNING: This report is calculated on training data only and is NOT cross-validated."
                        ]
                    else:
                        warning_data[col] = [""]
                warning_row = pd.DataFrame(warning_data, index=["WARNING"])
                report_df = pd.concat([warning_row, report_df], axis=0)
            report_file = os.path.join(output_dir, f"{base_name}_classification_report.tsv")
            report_df.to_csv(report_file, sep="\t")
            print(f"Saved classification report to {report_file}")

            # Plot and save confusion matrix
            import matplotlib.pyplot as plt
            import seaborn as sns

            plt.figure(figsize=(8, 6))
            sns.heatmap(
                results["confusion_matrix"],
                annot=True,
                fmt="d",
                cmap="Blues",
                xticklabels=self.class_names,
                yticklabels=self.class_names,
            )
            # Set confusion matrix plot title to include CV method
            cv_type = results.get("cv_type", None)
            if cv_type == "training_only" or is_training_only:
                title = "Confusion Matrix (Training Data - NOT Cross-Validated)"
            elif cv_type == "loo":
                title = "Confusion Matrix (Leave-One-Out CV)"
            elif cv_type and cv_type.endswith("_fold"):
                n_folds = cv_type.replace("_fold", "")
                title = f"Confusion Matrix ({n_folds}-Fold Cross-Validation)"
            else:
                title = "Confusion Matrix"
            plt.title(title)
            plt.xlabel("Predicted Label")
            plt.ylabel("True Label")
            plt.tight_layout()
            conf_plot = os.path.join(output_dir, f"{base_name}_confusion_matrix.pdf")
            plt.savefig(conf_plot, dpi=300, bbox_inches="tight")
            plt.close()
            print(f"Saved confusion matrix plot to {conf_plot}")

        # Save feature importance if model is fitted (both cases)
        if self.is_fitted:
            importance = self.get_feature_importance()
            # Ensure importance is 2D: (n_features, n_classes)
            if importance.ndim == 1:
                # Only one class: shape (n_features,)
                importance = importance[:, None]
            n_features, n_classes = importance.shape
            # Defensive: match columns to n_classes
            if len(self.class_names) != n_classes:
                columns = [f"class_{i}" for i in range(n_classes)]
            else:
                columns = self.class_names
            importance_df = pd.DataFrame(importance, columns=columns)
            importance_df["feature_idx"] = range(len(importance_df))
            importance_file = os.path.join(output_dir, f"{base_name}_feature_importance.tsv")
            importance_df.to_csv(importance_file, sep="\t", index=False)
            print(f"Saved feature importance to {importance_file}")
            # Save the trained model if requested
            if save_model:
                self.save_model(output_dir, base_name)


def main():
    """Main function for command-line interface."""
    parser = argparse.ArgumentParser(
        description="Linear probe for cluster prediction from embeddings",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument("embedding_file", help="Path to embedding pickle file")
    parser.add_argument("cluster_file", help="Path to TSV file with cluster assignments")
    parser.add_argument(
        "--protein-id-col", default="protein_id", help="Column name for protein IDs in cluster file"
    )
    parser.add_argument(
        "--cluster-col",
        default="all",
        help="Column name(s) for cluster assignments. Use 'all' to process all non-ID columns, or comma-separated list for multiple columns",
    )
    parser.add_argument(
        "--output-dir", default="linear_probe_results", help="Output directory for results"
    )
    parser.add_argument(
        "--base-name", help="Base name for output files (default: derived from embedding filename)"
    )
    parser.add_argument(
        "--regularization", type=float, default=1.0, help="L2 regularization strength"
    )
    parser.add_argument("--max-iter", type=int, default=1000, help="Maximum iterations for solver")
    parser.add_argument(
        "--no-standardize",
        action="store_true",
        help="Skip feature standardization (center=False, scale=False)",
    )
    parser.add_argument(
        "--norm-center", action="store_true", default=True, help="Center features (subtract mean)"
    )
    parser.add_argument(
        "--no-norm-center", dest="norm_center", action="store_false", help="Don't center features"
    )
    parser.add_argument(
        "--norm-scale", action="store_true", default=True, help="Scale features (divide by std)"
    )
    parser.add_argument(
        "--no-norm-scale", dest="norm_scale", action="store_false", help="Don't scale features"
    )
    parser.add_argument(
        "--norm-pca-components",
        type=float,
        default=0.95,
        help="Number of PCA components or variance to retain (default: 0.95 for 95%% variance)",
    )
    parser.add_argument(
        "--norm-l2", action="store_true", default=True, help="Apply L2 normalization"
    )
    parser.add_argument(
        "--no-norm-l2", dest="norm_l2", action="store_false", help="Don't apply L2 normalization"
    )
    parser.add_argument(
        "--cv",
        default="loo",
        help="Cross-validation method: 'none' (training only), 'loo' (leave-one-out), or integer (N-fold CV)",
    )
    parser.add_argument(
        "--random-seed", type=int, default=42, help="Random seed for reproducibility"
    )
    parser.add_argument(
        "--save-model",
        action="store_true",
        default=True,
        help="Save the trained model (default: True)",
    )
    parser.add_argument(
        "--no-save-model",
        dest="save_model",
        action="store_false",
        help="Don't save the trained model",
    )

    args = parser.parse_args()

    # Handle normalization parameters
    if args.no_standardize:
        # Override other norm settings if no-standardize is specified
        norm_center = False
        norm_scale = False
        norm_pca_components = None
        norm_l2 = False
    else:
        norm_center = args.norm_center
        norm_scale = args.norm_scale
        norm_pca_components = args.norm_pca_components
        norm_l2 = args.norm_l2

    # Handle CV parameter
    if args.cv.lower() == "none":
        cv_param = "none"
    elif args.cv.lower() == "loo":
        cv_param = "loo"
    else:
        try:
            cv_param = int(args.cv)
            if cv_param < 2:
                raise ValueError("N-fold CV requires N >= 2")
        except ValueError:
            raise ValueError(f"Invalid cv option: {args.cv}. Use 'none', 'loo', or integer >= 2")

    # Derive base name if not provided
    if args.base_name is None:
        base_name = os.path.splitext(os.path.basename(args.embedding_file))[0]
        base_name = base_name.replace("_embeddings", "").replace("embeddings", "")
    else:
        base_name = args.base_name

    print("=" * 60)
    print("Linear Probe for Cluster Prediction")
    print("=" * 60)
    print(f"Embedding file: {args.embedding_file}")
    print(f"Cluster file: {args.cluster_file}")
    print(f"Protein ID column: {args.protein_id_col}")
    print(f"Cluster column: {args.cluster_col}")
    print(f"Output directory: {args.output_dir}")
    print(f"Base name: {base_name}")
    print(f"Regularization: {args.regularization}")
    print(f"Cross-validation: {cv_param}")
    print(f"Normalization pipeline:")
    print(f"  Center: {norm_center}")
    print(f"  Scale: {norm_scale}")
    print(f"  PCA components: {norm_pca_components}")
    print(f"  L2 normalize: {norm_l2}")
    print("")

    try:
        # Initialize probe
        probe = LinearProbe(
            regularization=args.regularization,
            max_iter=args.max_iter,
            random_state=args.random_seed,
            norm_center=norm_center,
            norm_scale=norm_scale,
            norm_pca_components=norm_pca_components,
            norm_l2=norm_l2,
            cv=cv_param,
        )

        # Parse cluster columns
        if args.cluster_col == "all":
            cluster_cols = "all"
        elif "," in args.cluster_col:
            cluster_cols = [col.strip() for col in args.cluster_col.split(",")]
        else:
            cluster_cols = args.cluster_col

        # Load data
        data_dict = probe.load_data(
            args.embedding_file, args.cluster_file, args.protein_id_col, cluster_cols
        )

        # Process each cluster column
        all_results = {}

        for cluster_col, (X, y, protein_ids) in data_dict.items():
            print(f"\n{'='*60}")
            print(f"Processing cluster column: {cluster_col}")
            print(f"{'='*60}")

            # Create a new probe instance for each cluster column to avoid interference
            current_probe = LinearProbe(
                regularization=args.regularization,
                max_iter=args.max_iter,
                random_state=args.random_seed,
                norm_center=norm_center,
                norm_scale=norm_scale,
                norm_pca_components=norm_pca_components,
                norm_l2=norm_l2,
                cv=cv_param,
            )

            if cv_param == "none":
                # Training metrics only
                results = current_probe.evaluate_cv(X, y, protein_ids)
                # Also fit full model for feature importance
                current_probe.fit_full_model(X, y)
            else:
                # Perform cross-validation (LOO or K-fold)
                results = current_probe.evaluate_cv(X, y, protein_ids)
                # Fit full model for feature importance
                current_probe.fit_full_model(X, y)

            # Create output filename with cluster column name
            if len(data_dict) > 1:
                output_base = f"{base_name}_{cluster_col}"
            else:
                output_base = base_name

            # Save results
            current_probe.save_results(results, args.output_dir, output_base, args.save_model)

            # Store results for summary
            all_results[cluster_col] = results

        # Print summary for multiple cluster columns
        if len(data_dict) > 1:
            print(f"\n{'='*60}")
            print("SUMMARY - All Cluster Columns")
            print(f"{'='*60}")

            summary_data = []
            for cluster_col, results in all_results.items():
                cv_type = results.get("cv_type", "unknown")
                if cv_type == "training_only":
                    eval_type = "Training"
                elif cv_type == "loo":
                    eval_type = "LOO CV"
                elif cv_type.endswith("_fold"):
                    eval_type = cv_type.replace("_", "-").upper()
                elif results.get("skipped_loo", False):
                    eval_type = (
                        "Training" if results.get("training_metrics", False) else "No metrics"
                    )
                else:
                    eval_type = "Unknown"

                summary_data.append(
                    {
                        "cluster_column": cluster_col,
                        "evaluation_type": eval_type,
                        "accuracy": results.get("accuracy", "N/A"),
                        "f1_macro": results.get("f1_macro", "N/A"),
                        "n_classes": (
                            len(results["class_names"])
                            if "class_names" in results and results["class_names"] is not None
                            else "N/A"
                        ),
                    }
                )

            summary_df = pd.DataFrame(summary_data)
            print(summary_df.to_string(index=False, float_format="%.4f"))

            # Save summary
            summary_file = os.path.join(args.output_dir, f"{base_name}_summary.tsv")
            summary_df.to_csv(summary_file, sep="\t", index=False)
            print(f"\nSummary saved to: {summary_file}")

        print("\n" + "=" * 60)
        cv_type = cv_param
        if cv_type == "none":
            print("Analysis Complete! (Training evaluation only)")
        elif cv_type == "loo":
            print("Analysis Complete! (Leave-one-out cross-validation)")
        elif isinstance(cv_type, int):
            print(f"Analysis Complete! ({cv_type}-fold cross-validation)")
        else:
            print("Analysis Complete!")
        print(f"Results saved to: {args.output_dir}")
        if len(data_dict) > 1:
            print(f"Processed {len(data_dict)} cluster columns: {list(data_dict.keys())}")
        print("=" * 60)

    except Exception as e:
        import traceback

        print("\nError in analysis:")
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
