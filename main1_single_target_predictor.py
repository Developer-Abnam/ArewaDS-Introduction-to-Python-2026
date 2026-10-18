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


class Property_Predictor:
    def __init__(self, output_dir="best_models", mode="single", target_index=0):
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
        self.best_r2 = -np.inf
        self.output_dir = output_dir
        self.history = None
        self.feature_names = None
        self.mode = mode
        self.target_index = target_index
        self.all_results = {}
        self.best_results = {}

        os.makedirs(self.output_dir, exist_ok=True)

        self.models = {
            "Linear Regression": LinearRegression(),
            "Ridge Regression": Ridge(),
            "Lasso Regression": Lasso(),
            "ElasticNet Regression": ElasticNet(),
            "Bayesian Ridge Regression": BayesianRidge(),
            "ARD Regression": ARDRegression(),
            "SGD Regressor": SGDRegressor(max_iter=1000, tol=1e-3, random_state=42),
            "Huber Regressor": HuberRegressor(),
            "RANSAC Regressor": RANSACRegressor(random_state=42),
            "Tweedie Regressor": TweedieRegressor(power=0, max_iter=1000),
            "Support Vector Regression": SVR(),
            "Linear Support Vector Regression": LinearSVR(
                max_iter=10000, random_state=42
            ),
            "Decision Tree Regressor": DecisionTreeRegressor(random_state=42),
            "Random Forest": RandomForestRegressor(n_estimators=100, random_state=42),
            "Gradient Boosting": GradientBoostingRegressor(
                n_estimators=100, random_state=42
            ),
            "AdaBoost Regressor": AdaBoostRegressor(n_estimators=100, random_state=42),
            "Bagging Regressor": BaggingRegressor(n_estimators=100, random_state=42),
            "Extra Trees Regressor": ExtraTreesRegressor(
                n_estimators=100, random_state=42
            ),
            "Kernel Ridge Regression": KernelRidge(),
            "K-Neighbors Regressor": KNeighborsRegressor(),
            "Gaussian Process Regressor": GaussianProcessRegressor(),
            "Neural Network": MLPRegressor(
                hidden_layer_sizes=(165, 35), max_iter=1000, random_state=42
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
                    f"CSV file must have exactly 9 columns. Found {len(self.data.columns)}."
                )

        input_cols = self.data.columns[:4]
        target_cols = self.data.columns[4:9]

        self.X = self.data[input_cols]
        self.y = self.data[target_cols]

        self.feature_names = list(input_cols)
        self.target_names = list(target_cols)

        if self.mode == "multi":
            print(f"\n{'=' * 60}")
            print(f"MULTI-TARGET MODE: Predicting all 5 targets simultaneously")
            print(f"{'=' * 60}")
            print(
                f"Loaded {len(self.data)} samples with {len(input_cols)} input features"
            )
            print(f"Targets: {', '.join(target_cols)}")
        else:
            print(f"\n{'=' * 60}")
            print(
                f"SINGLE-TARGET MODE: Predicting target: {target_cols[self.target_index]}"
            )
            print(f"{'=' * 60}")
            print(
                f"Loaded {len(self.data)} samples with {len(input_cols)} input features"
            )
            self.y = self.y[[target_cols[self.target_index]]]
            self.target_names = [target_cols[self.target_index]]

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

    def train_model(self, model_name):
        if model_name not in self.models:
            raise ValueError(f"Model {model_name} not available")

        base_model = self.models[model_name]

        if self.mode == "multi":
            self.model = MultiOutputRegressor(base_model)
            print(f"\n=== Training {model_name} for ALL targets ===")
            self.model.fit(self.X_train, self.y_train)

            test_pred = self.model.predict(self.X_test)
            avg_r2 = 0
            print("\nResults for each target:")
            for i, target_name in enumerate(self.target_names):
                r2 = r2_score(self.y_test.iloc[:, i], test_pred[:, i])
                rmse = np.sqrt(
                    mean_squared_error(self.y_test.iloc[:, i], test_pred[:, i])
                )
                mae = mean_absolute_error(self.y_test.iloc[:, i], test_pred[:, i])
                avg_r2 += r2
                print(f"  {target_name}: R²={r2:.3f}, RMSE={rmse:.3f}, MAE={mae:.3f}")
            avg_r2 /= len(self.target_names)
            print(f"\nAverage R²: {avg_r2:.3f}")

            if avg_r2 > self.best_r2:
                self.best_r2 = avg_r2
                self.best_model = self.model
                self.best_model_name = model_name
                print(f"New best model found: {model_name} with Avg R²: {avg_r2:.3f}")

        else:
            print(f"\n=== Training {model_name} for target: {self.target_names[0]} ===")

            y_train = self.y_train.iloc[:, 0]
            y_test = self.y_test.iloc[:, 0]

            self.model = make_pipeline(
                SimpleImputer(strategy="mean"), StandardScaler(), base_model
            )

            self.model.fit(self.X_train, y_train)

            train_pred = self.model.predict(self.X_train)
            test_pred = self.model.predict(self.X_test)

            train_r2 = r2_score(y_train, train_pred)
            train_rmse = np.sqrt(mean_squared_error(y_train, train_pred))
            train_mae = mean_absolute_error(y_train, train_pred)

            test_r2 = r2_score(y_test, test_pred)
            test_rmse = np.sqrt(mean_squared_error(y_test, test_pred))
            test_mae = mean_absolute_error(y_test, test_pred)

            print(
                f"\nTraining: R²={train_r2:.3f}, RMSE={train_rmse:.3f}, MAE={train_mae:.3f}"
            )
            print(f"Test: R²={test_r2:.3f}, RMSE={test_rmse:.3f}, MAE={test_mae:.3f}")

            if test_r2 > self.best_r2:
                self.best_r2 = test_r2
                self.best_model = self.model
                self.best_model_name = model_name
                print(f"New best model found: {model_name} with R²: {test_r2:.3f}")

    def evaluate_all_models(self):
        print(f"\n{'=' * 60}")
        print("EVALUATING ALL MODELS")
        print(f"{'=' * 60}")

        all_results = {}
        best_results = []

        for name, base_model in self.models.items():
            print(f"\n--- {name} ---")

            if self.mode == "multi":
                model = MultiOutputRegressor(base_model)
                model.fit(self.X_train, self.y_train)
                train_pred = model.predict(self.X_train)
                test_pred = model.predict(self.X_test)

                avg_r2 = 0
                target_metrics = {}
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
                    test_mae = mean_absolute_error(
                        self.y_test.iloc[:, i], test_pred[:, i]
                    )

                    avg_r2 += test_r2
                    target_metrics[target_name] = {
                        "train_R2": train_r2,
                        "train_RMSE": train_rmse,
                        "train_MAE": train_mae,
                        "test_R2": test_r2,
                        "test_RMSE": test_rmse,
                        "test_MAE": test_mae,
                    }
                    print(f"  {target_name}: Test R²={test_r2:.3f}")
                avg_r2 /= len(self.target_names)
                print(f"  Average Test R²: {avg_r2:.3f}")

                if avg_r2 > self.best_r2:
                    self.best_r2 = avg_r2
                    self.best_model = model
                    self.best_model_name = name
                    self.best_results = target_metrics
                    print(f"  New best model! Avg R²: {avg_r2:.3f}")

                all_results[name] = {"avg_r2": avg_r2, "targets": target_metrics}

            else:
                y_train = self.y_train.iloc[:, 0]
                y_test = self.y_test.iloc[:, 0]

                model = make_pipeline(
                    SimpleImputer(strategy="mean"), StandardScaler(), base_model
                )
                model.fit(self.X_train, y_train)

                train_pred = model.predict(self.X_train)
                test_pred = model.predict(self.X_test)

                train_r2 = r2_score(y_train, train_pred)
                train_rmse = np.sqrt(mean_squared_error(y_train, train_pred))
                train_mae = mean_absolute_error(y_train, train_pred)

                test_r2 = r2_score(y_test, test_pred)
                test_rmse = np.sqrt(mean_squared_error(y_test, test_pred))
                test_mae = mean_absolute_error(y_test, test_pred)

                print(f"  Train R²={train_r2:.3f}, Test R²={test_r2:.3f}")

                if test_r2 > self.best_r2:
                    self.best_r2 = test_r2
                    self.best_model = model
                    self.best_model_name = name
                    self.best_results = {
                        "train_R2": train_r2,
                        "train_RMSE": train_rmse,
                        "train_MAE": train_mae,
                        "test_R2": test_r2,
                        "test_RMSE": test_rmse,
                        "test_MAE": test_mae,
                    }
                    print(f"  New best model! Test R²: {test_r2:.3f}")

                all_results[name] = {
                    "train_R2": train_r2,
                    "train_RMSE": train_rmse,
                    "train_MAE": train_mae,
                    "test_R2": test_r2,
                    "test_RMSE": test_rmse,
                    "test_MAE": test_mae,
                }

        self.all_results = all_results

        print(f"\n{'=' * 60}")
        print("SUMMARY")
        print(f"{'=' * 60}")
        print(f"Best model: {self.best_model_name}")
        print(f"Best score: {self.best_r2:.3f}")

        self.save_results_to_excel()
        self.save_best_model()

        return all_results

    def save_results_to_excel(self, filename=None):
        if not self.all_results:
            print("No results to save.")
            return

        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            if self.mode == "multi":
                filename = f"results_multi_target_{timestamp}.xlsx"
            else:
                filename = f"results_{self.target_names[0]}_{timestamp}.xlsx"

        filepath = os.path.join(self.output_dir, filename)

        with pd.ExcelWriter(filepath, engine="openpyxl") as writer:
            if self.mode == "multi":
                summary_data = []
                for model_name, metrics in self.all_results.items():
                    summary_data.append(
                        {"Model": model_name, "Average R²": metrics["avg_r2"]}
                    )
                    for target_name, target_metrics in metrics["targets"].items():
                        summary_data.append(
                            {
                                "Model": f"  └ {target_name}",
                                "Test R²": target_metrics["test_R2"],
                                "Test RMSE": target_metrics["test_RMSE"],
                                "Test MAE": target_metrics["test_MAE"],
                                "Train R²": target_metrics["train_R2"],
                                "Train RMSE": target_metrics["train_RMSE"],
                                "Train MAE": target_metrics["train_MAE"],
                            }
                        )

                df_summary = pd.DataFrame(summary_data)
                df_summary.to_excel(writer, sheet_name="Summary", index=False)

                best_per_target = []
                for target_name in self.target_names:
                    best_model = max(
                        [
                            (name, metrics["targets"][target_name]["test_R2"])
                            for name, metrics in self.all_results.items()
                        ],
                        key=lambda x: x[1],
                    )
                    best_metrics = self.all_results[best_model[0]]["targets"][
                        target_name
                    ]
                    best_per_target.append(
                        {
                            "Target": target_name,
                            "Best Model": best_model[0],
                            "Test R²": best_metrics["test_R2"],
                            "Test RMSE": best_metrics["test_RMSE"],
                            "Test MAE": best_metrics["test_MAE"],
                            "Train R²": best_metrics["train_R2"],
                            "Train RMSE": best_metrics["train_RMSE"],
                            "Train MAE": best_metrics["train_MAE"],
                        }
                    )

                df_best = pd.DataFrame(best_per_target)
                df_best.to_excel(writer, sheet_name="Best_Per_Target", index=False)

                for target_name in self.target_names:
                    target_data = []
                    for model_name, metrics in self.all_results.items():
                        target_metrics = metrics["targets"][target_name]
                        target_data.append(
                            {
                                "Model": model_name,
                                "Test R²": target_metrics["test_R2"],
                                "Test RMSE": target_metrics["test_RMSE"],
                                "Test MAE": target_metrics["test_MAE"],
                                "Train R²": target_metrics["train_R2"],
                                "Train RMSE": target_metrics["train_RMSE"],
                                "Train MAE": target_metrics["train_MAE"],
                            }
                        )
                    df_target = pd.DataFrame(target_data).sort_values(
                        "Test R²", ascending=False
                    )
                    sheet_name = target_name[:31]
                    df_target.to_excel(writer, sheet_name=sheet_name, index=False)

            else:
                results_data = []
                for model_name, metrics in self.all_results.items():
                    results_data.append(
                        {
                            "Model": model_name,
                            "Test R²": metrics["test_R2"],
                            "Test RMSE": metrics["test_RMSE"],
                            "Test MAE": metrics["test_MAE"],
                            "Train R²": metrics["train_R2"],
                            "Train RMSE": metrics["train_RMSE"],
                            "Train MAE": metrics["train_MAE"],
                        }
                    )

                df_results = pd.DataFrame(results_data).sort_values(
                    "Test R²", ascending=False
                )
                df_results.to_excel(writer, sheet_name="Results", index=False)

                best_df = pd.DataFrame(
                    [
                        {
                            "Target": self.target_names[0],
                            "Best Model": self.best_model_name,
                            "Test R²": self.best_results["test_R2"],
                            "Test RMSE": self.best_results["test_RMSE"],
                            "Test MAE": self.best_results["test_MAE"],
                            "Train R²": self.best_results["train_R2"],
                            "Train RMSE": self.best_results["train_RMSE"],
                            "Train MAE": self.best_results["train_MAE"],
                        }
                    ]
                )
                best_df.to_excel(writer, sheet_name="Best_Model", index=False)

        print(f"\nResults saved to: {filepath}")
        return filepath

    def save_best_model(self):
        if self.best_model is None:
            print("No best model to save.")
            return

        if self.mode == "multi":
            filename = f"best_model_{self.best_model_name.replace(' ', '_')}_multi.pkl"
        else:
            filename = f"best_model_{self.best_model_name.replace(' ', '_')}_{self.target_names[0]}.pkl"

        filepath = os.path.join(self.output_dir, filename)

        joblib.dump(
            {
                "model": self.best_model,
                "scaler": self.scaler,
                "feature_names": self.feature_names,
                "target_names": self.target_names,
                "model_name": self.best_model_name,
                "mode": self.mode,
                "best_r2": self.best_r2,
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

    def plot_parity(self):
        if self.best_model is None:
            print("No best model available. Run evaluate_all_models() first.")
            return

        if self.mode == "multi":
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
                    [min_val, max_val],
                    [min_val, max_val],
                    "k--",
                    alpha=0.5,
                    linewidth=1,
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
                f"Parity Plots - {self.best_model_name}", fontsize=14, fontweight="bold"
            )
            plt.tight_layout()
            plt.show()

        else:
            y_train = self.y_train.iloc[:, 0]
            y_test = self.y_test.iloc[:, 0]
            train_pred = self.best_model.predict(self.X_train)
            test_pred = self.best_model.predict(self.X_test)

            plt.figure(figsize=(8, 7))

            plt.scatter(
                y_train,
                train_pred,
                alpha=0.6,
                label="Train",
                color="blue",
                marker="o",
                s=60,
            )
            plt.scatter(
                y_test,
                test_pred,
                alpha=0.6,
                label="Test",
                color="red",
                marker="s",
                s=60,
            )

            min_val = min(
                y_train.min(), y_test.min(), train_pred.min(), test_pred.min()
            )
            max_val = max(
                y_train.max(), y_test.max(), train_pred.max(), test_pred.max()
            )
            plt.plot(
                [min_val, max_val], [min_val, max_val], "k--", alpha=0.5, linewidth=1.5
            )

            train_r2 = r2_score(y_train, train_pred)
            test_r2 = r2_score(y_test, test_pred)

            plt.xlabel("Experimental Value", fontsize=12)
            plt.ylabel("Predicted Value", fontsize=12)
            plt.title(
                f"Parity Plot - {self.best_model_name}\n{self.target_names[0]}",
                fontsize=14,
                fontweight="bold",
            )
            plt.legend(loc="best", fontsize=10)
            plt.grid(True, alpha=0.3)

            textstr = f"Train R² = {train_r2:.3f}\nTest R² = {test_r2:.3f}"
            props = dict(boxstyle="round", facecolor="wheat", alpha=0.8)
            plt.text(
                0.05,
                0.95,
                textstr,
                transform=plt.gca().transAxes,
                fontsize=10,
                verticalalignment="top",
                bbox=props,
            )

            plt.tight_layout()
            plt.show()

    def plot_feature_importance(self, top_n=20):
        if self.best_model is None:
            print("No best model available. Run evaluate_all_models() first.")
            return

        model = self.best_model
        if hasattr(model, "named_steps"):
            for step_name, step in model.named_steps.items():
                if hasattr(step, "coef_") or hasattr(step, "feature_importances_"):
                    model = step
                    break

        if hasattr(model, "coef_"):
            importance = np.abs(model.coef_)
        elif hasattr(model, "feature_importances_"):
            importance = model.feature_importances_
        else:
            print("Feature importance not available for this model type")
            return

        if len(importance.shape) > 1:
            importance = np.mean(importance, axis=0)

        idx = np.argsort(importance)[::-1][:top_n]
        sorted_names = [self.feature_names[i] for i in idx]
        sorted_importance = importance[idx]

        plt.figure(figsize=(10, 6))
        bars = plt.barh(range(len(sorted_importance)), sorted_importance[::-1])
        plt.yticks(range(len(sorted_importance)), sorted_names[::-1])
        plt.xlabel("Feature Importance", fontsize=12)
        plt.ylabel("Features", fontsize=12)
        plt.title(
            f"Top {top_n} Features - {self.best_model_name}",
            fontsize=14,
            fontweight="bold",
        )
        plt.tight_layout()
        plt.show()


# ============================================================================
# USAGE EXAMPLES
# ============================================================================


def example_single_target():
    print("\n" + "=" * 80)
    print("EXAMPLE 1: SINGLE-TARGET MODE (Predicting only one property)")
    print("=" * 80)

    predictor = Property_Predictor(
        output_dir="results_single_target", mode="single", target_index=4
    )

    predictor.load_data("dataset.csv")
    predictor.preprocess_data(test_size=0.2, random_state=42)
    predictor.evaluate_all_models()
    predictor.plot_parity()
    predictor.plot_feature_importance()

    return predictor


def example_multi_target():
    print("\n" + "=" * 80)
    print("EXAMPLE 2: MULTI-TARGET MODE (Predicting all 5 properties simultaneously)")
    print("=" * 80)

    predictor = Property_Predictor(output_dir="results_multi_target", mode="multi")

    predictor.load_data("dataset.csv")
    predictor.preprocess_data(test_size=0.2, random_state=42)
    predictor.evaluate_all_models()
    predictor.plot_parity()
    predictor.plot_feature_importance()

    return predictor


if __name__ == "__main__":
    predictor = example_multi_target()

    print("\n" + "=" * 80)
    print("ANALYSIS COMPLETE!")
    print(f"Results saved in: {predictor.output_dir}")
    print("=" * 80)
