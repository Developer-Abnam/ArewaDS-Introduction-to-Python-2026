# -*- coding: utf-8 -*-
"""
Created on Mon Jul  6 16:55:20 2026

@author: user
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.feature_selection import VarianceThreshold
from sklearn.impute import SimpleImputer
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import (
    LinearRegression,
    Ridge,
    Lasso,
    ElasticNet,
    BayesianRidge,
    ARDRegression,
    SGDRegressor,
    HuberRegressor,
    RANSACRegressor,
    TweedieRegressor,
)
from sklearn.svm import SVR, LinearSVR
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import (
    RandomForestRegressor,
    GradientBoostingRegressor,
    AdaBoostRegressor,
    BaggingRegressor,
    ExtraTreesRegressor,
)
from sklearn.kernel_ridge import KernelRidge
from sklearn.neighbors import KNeighborsRegressor
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.multioutput import MultiOutputRegressor
import joblib
import os
import warnings
import chardet
from datetime import datetime

warnings.filterwarnings("ignore")


class MultiTargetPredictor:
    def __init__(self, output_dir="multi_target_results"):
        self.data = None
        self.X = None
        self.y = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.scaler = None
        self.model = None
        self.best_model = None
        self.best_model_name = None
        self.best_avg_r2 = -np.inf
        self.output_dir = output_dir
        self.feature_names = None
        self.target_names = None
        self.all_results = {}
        self.best_results = {}

        os.makedirs(self.output_dir, exist_ok=True)

        self.models = {
            "Linear Regression": MultiOutputRegressor(LinearRegression()),
            "Ridge Regression": MultiOutputRegressor(Ridge()),
            "Lasso Regression": MultiOutputRegressor(Lasso()),
            "ElasticNet Regression": MultiOutputRegressor(ElasticNet()),
            "Bayesian Ridge": MultiOutputRegressor(BayesianRidge()),
            "ARD Regression": MultiOutputRegressor(ARDRegression()),
            "SGD Regressor": MultiOutputRegressor(
                SGDRegressor(max_iter=1000, tol=1e-3, random_state=42)
            ),
            "Huber Regressor": MultiOutputRegressor(HuberRegressor()),
            "RANSAC Regressor": MultiOutputRegressor(RANSACRegressor(random_state=42)),
            "Tweedie Regressor": MultiOutputRegressor(
                TweedieRegressor(power=0, max_iter=1000)
            ),
            "SVR": MultiOutputRegressor(SVR()),
            "Linear SVR": MultiOutputRegressor(
                LinearSVR(max_iter=10000, random_state=42)
            ),
            "Decision Tree": MultiOutputRegressor(
                DecisionTreeRegressor(random_state=42)
            ),
            "Random Forest": MultiOutputRegressor(
                RandomForestRegressor(n_estimators=100, random_state=42)
            ),
            "Gradient Boosting": MultiOutputRegressor(
                GradientBoostingRegressor(n_estimators=100, random_state=42)
            ),
            "AdaBoost": MultiOutputRegressor(
                AdaBoostRegressor(n_estimators=100, random_state=42)
            ),
            "Bagging": MultiOutputRegressor(
                BaggingRegressor(n_estimators=100, random_state=42)
            ),
            "Extra Trees": MultiOutputRegressor(
                ExtraTreesRegressor(n_estimators=100, random_state=42)
            ),
            "Kernel Ridge": MultiOutputRegressor(KernelRidge()),
            "K-Neighbors": MultiOutputRegressor(KNeighborsRegressor()),
            "Gaussian Process": MultiOutputRegressor(GaussianProcessRegressor()),
            "Neural Network": MultiOutputRegressor(
                MLPRegressor(
                    hidden_layer_sizes=(165, 35), max_iter=1000, random_state=42
                )
            ),
        }

    def _detect_encoding(self, file_path):
        with open(file_path, "rb") as f:
            raw_data = f.read()
            result = chardet.detect(raw_data)
            return result["encoding"]

    def load_data(self, file_path, encoding=None):
        if encoding is None:
            try:
                detected_encoding = self._detect_encoding(file_path)
                print(f"Detected encoding: {detected_encoding}")
                encoding = detected_encoding
            except Exception as e:
                print(f"Could not detect encoding: {e}")
                encodings = ["utf-8", "latin-1", "cp1252", "iso-8859-1", "utf-16"]
                for enc in encodings:
                    try:
                        self.data = pd.read_csv(file_path, encoding=enc)
                        print(f"Successfully loaded with encoding: {enc}")
                        break
                    except UnicodeDecodeError:
                        continue
                else:
                    raise ValueError("Could not read the file with any common encoding")
                return self._process_data()

        self.data = pd.read_csv(file_path, encoding=encoding)
        print(f"Loaded data with encoding: {encoding}")
        return self._process_data()

    def _process_data(self):
        if len(self.data.columns) != 9:
            if len(self.data.columns) == 10:
                print(
                    "Found 10 columns. Assuming first column is an index, removing it."
                )
                self.data = self.data.iloc[:, 1:]

            if len(self.data.columns) != 9:
                raise ValueError(
                    f"CSV file must have exactly 9 columns (4 inputs + 5 targets). Found {len(self.data.columns)}."
                )

        input_cols = self.data.columns[:4]
        target_cols = self.data.columns[4:9]

        self.X = self.data[input_cols]
        self.y = self.data[target_cols]

        self.feature_names = list(input_cols)
        self.target_names = list(target_cols)

        print(f"\n{'=' * 60}")
        print(f"MULTI-TARGET PREDICTOR: Predicting all 5 targets simultaneously")
        print(f"{'=' * 60}")
        print(f"Loaded {len(self.data)} samples with {len(input_cols)} input features")
        print(f"Targets: {', '.join(target_cols)}")

        print("\nChecking for NaN values in targets:")
        for target in self.target_names:
            nan_count = self.y[target].isna().sum()
            if nan_count > 0:
                print(
                    f"  {target}: {nan_count} NaN values ({nan_count / len(self.y) * 100:.1f}%)"
                )
            else:
                print(f"  {target}: No NaN values")

        print("\nFirst 5 rows of input data:")
        print(self.X.head())
        print("\nFirst 5 rows of target data:")
        print(self.y.head())

    def preprocess_data(self, test_size=0.2, random_state=42):
        if self.X is None:
            raise ValueError("No data loaded. Please load data first.")

        selector = VarianceThreshold(threshold=0.01)
        selected_features = selector.fit_transform(self.X)
        self.feature_names = np.array(self.X.columns)[selector.get_support()]
        self.X = pd.DataFrame(selected_features, columns=self.feature_names)

        original_len = len(self.y)
        self.y = self.y.dropna()
        self.X = self.X.loc[self.y.index]
        removed = original_len - len(self.y)
        if removed > 0:
            print(f"Removed {removed} rows with NaN in target variables")

        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            self.X, self.y, test_size=test_size, random_state=random_state
        )

        self.scaler = StandardScaler()
        self.X_train = pd.DataFrame(
            self.scaler.fit_transform(self.X_train), columns=self.feature_names
        )
        self.X_test = pd.DataFrame(
            self.scaler.transform(self.X_test), columns=self.feature_names
        )

        print(f"\nData preprocessed.")
        print(f"Train size: {len(self.y_train)}, Test size: {len(self.y_test)}")
        print(f"Features after variance filtering: {len(self.feature_names)}")
        print(f"Number of targets: {len(self.target_names)}")

    def evaluate_all_models(self):
        print(f"\n{'=' * 60}")
        print("EVALUATING ALL MODELS FOR MULTI-TARGET PREDICTION")
        print(f"{'=' * 60}")

        all_results = {}

        for name, model in self.models.items():
            print(f"\n--- {name} ---")
            print(f"Training model for all 5 targets simultaneously...")

            model.fit(self.X_train, self.y_train)

            train_pred = model.predict(self.X_train)
            test_pred = model.predict(self.X_test)

            target_metrics = {}
            avg_train_r2 = 0
            avg_test_r2 = 0

            for i, target_name in enumerate(self.target_names):
                train_r2 = r2_score(self.y_train.iloc[:, i], train_pred[:, i])
                train_rmse = np.sqrt(
                    mean_squared_error(self.y_train.iloc[:, i], train_pred[:, i])
                )
                train_mae = mean_absolute_error(
                    self.y_train.iloc[:, i], train_pred[:, i]
                )

                test_r2 = r2_score(self.y_test.iloc[:, i], test_pred[:, i])
                test_rmse = np.sqrt(
                    mean_squared_error(self.y_test.iloc[:, i], test_pred[:, i])
                )
                test_mae = mean_absolute_error(self.y_test.iloc[:, i], test_pred[:, i])

                avg_train_r2 += train_r2
                avg_test_r2 += test_r2

                target_metrics[target_name] = {
                    "train_R2": train_r2,
                    "train_RMSE": train_rmse,
                    "train_MAE": train_mae,
                    "test_R2": test_r2,
                    "test_RMSE": test_rmse,
                    "test_MAE": test_mae,
                }

            avg_train_r2 /= len(self.target_names)
            avg_test_r2 /= len(self.target_names)

            print(f"  Average Train R²: {avg_train_r2:.3f}")
            print(f"  Average Test R²:  {avg_test_r2:.3f}")

            all_results[name] = {
                "avg_train_r2": avg_train_r2,
                "avg_test_r2": avg_test_r2,
                "targets": target_metrics,
            }

            if avg_test_r2 > self.best_avg_r2:
                self.best_avg_r2 = avg_test_r2
                self.best_model = model
                self.best_model_name = name
                self.best_results = target_metrics
                print(f"  ★ NEW BEST MODEL! Average Test R²: {avg_test_r2:.3f}")

        self.all_results = all_results

        self._print_summary()
        self.save_results_to_excel()
        self.save_best_model()

        return all_results

    def _print_summary(self):
        print(f"\n{'=' * 60}")
        print("SUMMARY: BEST MODEL FOR MULTI-TARGET PREDICTION")
        print(f"{'=' * 60}")
        print(f"Best Model: {self.best_model_name}")
        print(f"Average Test R²: {self.best_avg_r2:.3f}")
        print("\nPerformance per target:")
        print(
            "{:<20} {:<12} {:<12} {:<12}".format(
                "Target", "Test R²", "Test RMSE", "Test MAE"
            )
        )
        print("-" * 60)
        for target_name, metrics in self.best_results.items():
            print(
                "{:<20} {:<12.3f} {:<12.3f} {:<12.3f}".format(
                    target_name,
                    metrics["test_R2"],
                    metrics["test_RMSE"],
                    metrics["test_MAE"],
                )
            )

        print(f"\n{'=' * 60}")
        print("MODEL RANKING (by Average Test R²)")
        print(f"{'=' * 60}")
        sorted_models = sorted(
            self.all_results.items(), key=lambda x: x[1]["avg_test_r2"], reverse=True
        )
        print("{:<30} {:<15}".format("Model", "Avg Test R²"))
        print("-" * 45)
        for rank, (name, metrics) in enumerate(sorted_models, 1):
            print("{:<30} {:<15.3f}".format(name, metrics["avg_test_r2"]))

    def save_results_to_excel(self, filename=None):
        if not self.all_results:
            print("No results to save.")
            return

        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"multi_target_results_{timestamp}.xlsx"

        filepath = os.path.join(self.output_dir, filename)

        with pd.ExcelWriter(filepath, engine="openpyxl") as writer:
            summary_data = []
            for model_name, metrics in self.all_results.items():
                summary_data.append(
                    {
                        "Model": model_name,
                        "Average Train R²": metrics["avg_train_r2"],
                        "Average Test R²": metrics["avg_test_r2"],
                    }
                )
                for target_name, target_metrics in metrics["targets"].items():
                    summary_data.append(
                        {
                            "Model": f"  └ {target_name}",
                            "Train R²": target_metrics["train_R2"],
                            "Train RMSE": target_metrics["train_RMSE"],
                            "Train MAE": target_metrics["train_MAE"],
                            "Test R²": target_metrics["test_R2"],
                            "Test RMSE": target_metrics["test_RMSE"],
                            "Test MAE": target_metrics["test_MAE"],
                        }
                    )

            df_summary = pd.DataFrame(summary_data)
            df_summary.to_excel(writer, sheet_name="Summary", index=False)

            best_df = pd.DataFrame(
                [{"Model": self.best_model_name, "Average Test R²": self.best_avg_r2}]
            )
            best_df.to_excel(writer, sheet_name="Best_Model_Summary", index=False)

            best_details = []
            for target_name, metrics in self.best_results.items():
                best_details.append(
                    {
                        "Target": target_name,
                        "Train R²": metrics["train_R2"],
                        "Train RMSE": metrics["train_RMSE"],
                        "Train MAE": metrics["train_MAE"],
                        "Test R²": metrics["test_R2"],
                        "Test RMSE": metrics["test_RMSE"],
                        "Test MAE": metrics["test_MAE"],
                    }
                )
            pd.DataFrame(best_details).to_excel(
                writer, sheet_name="Best_Model_Details", index=False
            )

            sorted_models = sorted(
                self.all_results.items(),
                key=lambda x: x[1]["avg_test_r2"],
                reverse=True,
            )
            ranking_data = []
            for rank, (name, metrics) in enumerate(sorted_models, 1):
                ranking_data.append(
                    {
                        "Rank": rank,
                        "Model": name,
                        "Average Train R²": metrics["avg_train_r2"],
                        "Average Test R²": metrics["avg_test_r2"],
                    }
                )
            pd.DataFrame(ranking_data).to_excel(
                writer, sheet_name="Model_Ranking", index=False
            )

            for target_name in self.target_names:
                target_data = []
                for model_name, metrics in self.all_results.items():
                    target_metrics = metrics["targets"][target_name]
                    target_data.append(
                        {
                            "Model": model_name,
                            "Train R²": target_metrics["train_R2"],
                            "Train RMSE": target_metrics["train_RMSE"],
                            "Train MAE": target_metrics["train_MAE"],
                            "Test R²": target_metrics["test_R2"],
                            "Test RMSE": target_metrics["test_RMSE"],
                            "Test MAE": target_metrics["test_MAE"],
                        }
                    )
                df_target = pd.DataFrame(target_data).sort_values(
                    "Test R²", ascending=False
                )
                sheet_name = target_name[:31]
                df_target.to_excel(writer, sheet_name=sheet_name, index=False)

        print(f"\nResults saved to: {filepath}")
        return filepath

    def save_best_model(self):
        if self.best_model is None:
            print("No best model to save.")
            return

        filename = (
            f"best_multi_target_model_{self.best_model_name.replace(' ', '_')}.pkl"
        )
        filepath = os.path.join(self.output_dir, filename)

        joblib.dump(
            {
                "model": self.best_model,
                "scaler": self.scaler,
                "feature_names": self.feature_names,
                "target_names": self.target_names,
                "model_name": self.best_model_name,
                "best_avg_r2": self.best_avg_r2,
                "best_results": self.best_results,
                "metadata": {
                    "creation_date": pd.Timestamp.now(),
                    "input_features": list(self.feature_names),
                    "targets": list(self.target_names),
                },
            },
            filepath,
        )

        print(f"\nBest model saved to: {filepath}")

    def predict(self, input_data):
        if self.best_model is None:
            raise ValueError("No model available. Run evaluate_all_models() first.")

        if isinstance(input_data, pd.DataFrame):
            input_scaled = self.scaler.transform(input_data[self.feature_names])
        else:
            input_df = pd.DataFrame([input_data], columns=self.feature_names)
            input_scaled = self.scaler.transform(input_df)

        predictions = self.best_model.predict(input_scaled)

        return pd.DataFrame(predictions, columns=self.target_names)

    def plot_parity(self):
        if self.best_model is None:
            print("No best model available. Run evaluate_all_models() first.")
            return

        train_pred = self.best_model.predict(self.X_train)
        test_pred = self.best_model.predict(self.X_test)

        n_targets = len(self.target_names)
        n_cols = min(3, n_targets)
        n_rows = (n_targets + n_cols - 1) // n_cols

        fig, axes = plt.subplots(n_rows, n_cols, figsize=(6 * n_cols, 5 * n_rows))
        if n_targets == 1:
            axes = [axes]
        else:
            axes = axes.flatten()

        for i, target_name in enumerate(self.target_names):
            ax = axes[i]
            y_train = self.y_train.iloc[:, i]
            y_test = self.y_test.iloc[:, i]

            ax.scatter(
                y_train,
                train_pred[:, i],
                alpha=0.6,
                label="Train",
                color="blue",
                marker="o",
                s=50,
            )
            ax.scatter(
                y_test,
                test_pred[:, i],
                alpha=0.6,
                label="Test",
                color="red",
                marker="s",
                s=50,
            )

            min_val = min(
                y_train.min(),
                y_test.min(),
                train_pred[:, i].min(),
                test_pred[:, i].min(),
            )
            max_val = max(
                y_train.max(),
                y_test.max(),
                train_pred[:, i].max(),
                test_pred[:, i].max(),
            )
            ax.plot(
                [min_val, max_val], [min_val, max_val], "k--", alpha=0.5, linewidth=1
            )

            train_r2 = r2_score(y_train, train_pred[:, i])
            test_r2 = r2_score(y_test, test_pred[:, i])

            ax.set_xlabel("Experimental Value", fontsize=10)
            ax.set_ylabel("Predicted Value", fontsize=10)
            ax.set_title(f"{target_name}", fontsize=12, fontweight="bold")
            ax.legend(loc="best", fontsize=9)
            ax.grid(True, alpha=0.3)

            textstr = f"Train R² = {train_r2:.3f}\nTest R² = {test_r2:.3f}"
            props = dict(boxstyle="round", facecolor="wheat", alpha=0.8)
            ax.text(
                0.05,
                0.95,
                textstr,
                transform=ax.transAxes,
                fontsize=9,
                verticalalignment="top",
                bbox=props,
            )

        for i in range(n_targets, len(axes)):
            fig.delaxes(axes[i])

        plt.suptitle(
            f"Multi-Target Parity Plots\nBest Model: {self.best_model_name}",
            fontsize=14,
            fontweight="bold",
        )
        plt.tight_layout()
        plt.show()

    def plot_feature_importance(self, top_n=20):
        if self.best_model is None:
            print("No best model available. Run evaluate_all_models() first.")
            return

        estimator = self.best_model.estimators_[0]

        if hasattr(estimator, "coef_"):
            all_coefs = np.array([est.coef_ for est in self.best_model.estimators_])
            importance = np.mean(np.abs(all_coefs), axis=0)
        elif hasattr(estimator, "feature_importances_"):
            all_importances = np.array(
                [est.feature_importances_ for est in self.best_model.estimators_]
            )
            importance = np.mean(all_importances, axis=0)
        else:
            print("Feature importance not available for this model type")
            return

        idx = np.argsort(importance)[::-1][:top_n]
        sorted_names = [self.feature_names[i] for i in idx]
        sorted_importance = importance[idx]

        plt.figure(figsize=(10, 6))
        bars = plt.barh(range(len(sorted_importance)), sorted_importance[::-1])
        plt.yticks(range(len(sorted_importance)), sorted_names[::-1])
        plt.xlabel("Average Feature Importance Across All Targets", fontsize=12)
        plt.ylabel("Features", fontsize=12)
        plt.title(
            f"Top {top_n} Features - {self.best_model_name}\n(Averaged across all 5 targets)",
            fontsize=14,
            fontweight="bold",
        )
        plt.tight_layout()
        plt.show()

    def plot_target_correlations(self):
        plt.figure(figsize=(8, 6))
        corr_matrix = self.y_train.corr()
        sns.heatmap(
            corr_matrix,
            annot=True,
            cmap="coolwarm",
            center=0,
            fmt=".2f",
            square=True,
            cbar_kws={"shrink": 0.8},
        )
        plt.title("Target Correlations (Training Set)", fontsize=14, fontweight="bold")
        plt.tight_layout()
        plt.show()


# ============================================================================
# USAGE EXAMPLE
# ============================================================================

if __name__ == "__main__":
    print("=" * 80)
    print("MULTI-TARGET PREDICTOR")
    print("A single model that predicts all 5 targets simultaneously")
    print("=" * 80)

    predictor = MultiTargetPredictor(output_dir="multi_target_results")

    predictor.load_data("dataset.csv")
    predictor.preprocess_data(test_size=0.2, random_state=42)
    predictor.evaluate_all_models()

    predictor.plot_parity()
    predictor.plot_feature_importance(top_n=20)
    predictor.plot_target_correlations()

    print("\n" + "=" * 60)
    print("MAKING PREDICTIONS FOR NEW DATA")
    print("=" * 60)

    sample = predictor.X_test.iloc[0:1]
    predictions = predictor.predict(sample)
    print("\nInput features:")
    print(sample)
    print("\nPredicted targets:")
    print(predictions)

    print("\n" + "=" * 80)
    print("ANALYSIS COMPLETE!")
    print(f"Results saved in: {predictor.output_dir}")
    print("=" * 80)
