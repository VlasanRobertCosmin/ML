import os
import numpy as np
import matplotlib.pyplot as plt

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    confusion_matrix,
    cohen_kappa_score,
    roc_curve,
    roc_auc_score,
)
from sklearn.feature_selection import (
    VarianceThreshold,
    SelectKBest,
    mutual_info_classif,
    SequentialFeatureSelector,
)
from scipy.stats import wilcoxon

# --------------------------------------------------------------------
# Global result directory
# --------------------------------------------------------------------
RESULT_DIR = "results"
os.makedirs(RESULT_DIR, exist_ok=True)


# --------------------------------------------------------------------
# Helper: define models used in Lecture 6
#using the get_model function ewe create fresh new models for every run 
# --------------------------------------------------------------------
def get_models():
    """
    Returns a dict {model_name: callable_that_creates_fresh_model}
    """

    def make_mlp():
        # Simple NN; standardization is done outside
        return MLPClassifier(
            hidden_layer_sizes=(20,),
            activation="relu",
            solver="adam",
            max_iter=500,
            random_state=None,
        )

    def make_dt():
        return DecisionTreeClassifier(
            criterion="gini",
            max_depth=None,
            random_state=None,
        )

    def make_bagging():
        base = DecisionTreeClassifier(
            criterion="gini",
            max_depth=None,
            random_state=None,
        )
        return BaggingClassifier(
            estimator=base,
            n_estimators=50,
            max_samples=0.8,
            max_features=1.0,
            random_state=None,
        )

    return {
        "NN": make_mlp,
        "DT": make_dt,
        "Bagging": make_bagging,
    }


# --------------------------------------------------------------------
# Helper: evaluate a classifier for one train/test split
# --------------------------------------------------------------------
def evaluate_classifier(model, X_train, X_test, y_train, y_test):
    """
    Fits model and returns metrics + data needed for ROC.
    """
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    # For ROC we need scores; use predict_proba if available, else decision_function
    if hasattr(model, "predict_proba"):
        y_score = model.predict_proba(X_test)[:, 1]
    elif hasattr(model, "decision_function"):
        y_score = model.decision_function(X_test)
    else:
        # Fallback: use predictions as scores
        y_score = y_pred

    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    kappa = cohen_kappa_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)
    fpr, tpr, _ = roc_curve(y_test, y_score)
    auc = roc_auc_score(y_test, y_score)

    return {
        "acc": acc,
        "f1": f1,
        "kappa": kappa,
        "cm": cm,
        "fpr": fpr,
        "tpr": tpr,
        "auc": auc,
        "y_pred": y_pred,
    }


# --------------------------------------------------------------------
# Lecture 6: Random subsampling CV + metrics + Wilcoxon + saving
# --------------------------------------------------------------------
def lecture6_experiment(n_runs=30, test_size=0.3):
    print("=== Lecture 6: Performance evaluation on Wisconsin dataset ===")

    data = load_breast_cancer()
    X = data.data
    y = data.target
    feature_names = data.feature_names
    target_names = data.target_names
    print(f"Dataset: Wisconsin breast cancer")
    print(f"Samples: {X.shape[0]}, Features: {X.shape[1]}")
    print(f"Classes: {list(target_names)}\n")

    models = get_models()

    # Standardize features (mainly for NN; trees don't need it but it's ok)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Storage for results per model
    acc_hist = {name: [] for name in models}
    f1_hist = {name: [] for name in models}
    kappa_hist = {name: [] for name in models}
    auc_hist = {name: [] for name in models}
    cm_sum = {name: np.zeros((2, 2), dtype=int) for name in models}

    # ROC for last run
    roc_last_run = {}

    for run in range(1, n_runs + 1):
        X_train, X_test, y_train, y_test = train_test_split(
            X_scaled,
            y,
            test_size=test_size,
            stratify=y,
            random_state=run,
        )

        print(f"Run {run}/{n_runs}")
        for name, maker in models.items():
            model = maker()
            metrics = evaluate_classifier(model, X_train, X_test, y_train, y_test)

            acc_hist[name].append(metrics["acc"])
            f1_hist[name].append(metrics["f1"])
            kappa_hist[name].append(metrics["kappa"])
            auc_hist[name].append(metrics["auc"])
            cm_sum[name] += metrics["cm"]
            roc_last_run[name] = (metrics["fpr"], metrics["tpr"], metrics["auc"])

        print(
            "  Accuracies: "
            + ", ".join([f"{name}={acc_hist[name][-1]:.3f}" for name in models])
        )

    print("\n=== Summary over random subsampling runs ===")

    # Collect lines to save to file
    summary_lines = []
    summary_lines.append("=== Lecture 6 Results ===\n")
    summary_lines.append(f"Number of runs: {n_runs}\n\n")

    # Summary per model + save confusion matrix plots
    for name in models:
        mean_acc = np.mean(acc_hist[name])
        std_acc = np.std(acc_hist[name])
        mean_f1 = np.mean(f1_hist[name])
        mean_kappa = np.mean(kappa_hist[name])
        mean_auc = np.mean(auc_hist[name])

        print(f"\nModel: {name}")
        print(f"  Accuracy: {mean_acc:.4f} ± {std_acc:.4f}")
        print(f"  F1-score (mean): {mean_f1:.4f}")
        print(f"  Cohen's kappa (mean): {mean_kappa:.4f}")
        print(f"  AUC (mean): {mean_auc:.4f}")
        print("  Confusion matrix (sum over runs):")
        print(cm_sum[name])

        summary_lines.append(f"Model: {name}\n")
        summary_lines.append(f"  Mean Accuracy: {mean_acc:.4f}\n")
        summary_lines.append(f"  Std Accuracy: {std_acc:.4f}\n")
        summary_lines.append(f"  Mean F1: {mean_f1:.4f}\n")
        summary_lines.append(f"  Mean Kappa: {mean_kappa:.4f}\n")
        summary_lines.append(f"  Mean AUC: {mean_auc:.4f}\n")
        summary_lines.append("  Confusion matrix (sum over runs):\n")
        summary_lines.append(str(cm_sum[name]) + "\n\n")

        # Save confusion matrix as image
        plt.figure()
        plt.imshow(cm_sum[name], cmap="Blues")
        plt.title(f"Confusion Matrix – {name}")
        plt.colorbar()
        plt.xlabel("Predicted")
        plt.ylabel("Actual")
        cm_path = os.path.join(RESULT_DIR, f"cm_{name}.png")
        plt.savefig(cm_path, dpi=300)
        plt.close()
        print(f"  Confusion matrix image saved to: {cm_path}")

    # Wilcoxon test for best two classifiers (by mean accuracy)
    mean_accs = {name: np.mean(acc_hist[name]) for name in models}
    sorted_by_acc = sorted(mean_accs.items(), key=lambda x: x[1], reverse=True)
    best1, best2 = sorted_by_acc[0][0], sorted_by_acc[1][0]

    print("\n=== Wilcoxon signed-rank test (accuracy) ===")
    print(f"Best two models (by mean accuracy): {best1}, {best2}")
    stat, p = wilcoxon(acc_hist[best1], acc_hist[best2])
    print(f"Wilcoxon statistic: {stat:.4f}, p-value: {p:.6f}")
    alpha = 0.05
    if p <= alpha:
        conclusion = "Reject H0 -> significant difference between the two models."
    else:
        conclusion = "Cannot reject H0 -> no significant difference detected."
    print("Result:", conclusion)

    summary_lines.append("=== Wilcoxon signed-rank test (accuracy) ===\n")
    summary_lines.append(f"Best models: {best1} vs {best2}\n")
    summary_lines.append(f"Statistic: {stat:.4f}\n")
    summary_lines.append(f"p-value: {p:.6f}\n")
    summary_lines.append(f"Conclusion: {conclusion}\n")

    # Save all lecture 6 numeric results to file
    lec6_path = os.path.join(RESULT_DIR, "lecture6_results.txt")
    with open(lec6_path, "w") as f:
        f.writelines(summary_lines)
    print(f"\nLecture 6 numeric results saved to: {lec6_path}")

    # ROC curve for best model (last run)
    print("\n=== ROC curve for best model (last run) ===")
    fpr, tpr, auc_last = roc_last_run[best1]

    plt.figure()
    plt.plot(fpr, tpr, label=f"{best1} (AUC = {auc_last:.3f})")
    plt.plot([0, 1], [0, 1], linestyle="--")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(f"ROC curve - {best1} (Wisconsin)")
    plt.legend(loc="lower right")
    plt.grid(True)

    roc_path = os.path.join(RESULT_DIR, f"roc_{best1}.png")
    plt.savefig(roc_path, dpi=300)
    plt.close()
    print(f"ROC curve saved to: {roc_path}")


# --------------------------------------------------------------------
# Lecture 7: Feature selection + grid search + saving
# --------------------------------------------------------------------
def lecture7_feature_selection_and_tuning():
    print("\n\n=== Lecture 7: Feature selection & parameter tuning (Bagging) ===")

    data = load_breast_cancer()
    X = data.data
    y = data.target
    feature_names = np.array(data.feature_names)

    print(f"Dataset: Wisconsin breast cancer")
    print(f"Samples: {X.shape[0]}, Features: {X.shape[1]}")

    # Single train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.3,
        stratify=y,
        random_state=42,
    )

    # ------------------------------------------------------------
    # Filter method 1: VarianceThreshold
    # ------------------------------------------------------------
    print("\n--- Filter 1: VarianceThreshold (remove near-constant features) ---")
    vt = VarianceThreshold(threshold=0.0)
    X_train_vt = vt.fit_transform(X_train)
    selected_mask_vt = vt.get_support()
    selected_features_vt = feature_names[selected_mask_vt]

    print(f"Original number of features: {X_train.shape[1]}")
    print(f"Remaining after VarianceThreshold: {X_train_vt.shape[1]}")
    print("Selected features (VarianceThreshold):")
    print(selected_features_vt)

    vt_path = os.path.join(RESULT_DIR, "fs_variance_threshold.txt")
    with open(vt_path, "w") as f:
        f.write("Selected features (VarianceThreshold):\n")
        for feat in selected_features_vt:
            f.write(feat + "\n")
    print(f"VarianceThreshold selected features saved to: {vt_path}")

    # ------------------------------------------------------------
    # Filter method 2: SelectKBest with mutual information
    # ------------------------------------------------------------
    print("\n--- Filter 2: SelectKBest (mutual_info_classif) ---")
    k = 10
    skb = SelectKBest(score_func=mutual_info_classif, k=k)
    X_train_skb = skb.fit_transform(X_train, y_train)
    selected_mask_skb = skb.get_support()
    selected_features_skb = feature_names[selected_mask_skb]

    print(f"Keeping top {k} features by mutual information.")
    print("Selected features (SelectKBest):")
    print(selected_features_skb)

    skb_path = os.path.join(RESULT_DIR, "fs_selectkbest.txt")
    with open(skb_path, "w") as f:
        f.write("Selected features (SelectKBest – mutual_info_classif):\n")
        for feat in selected_features_skb:
            f.write(feat + "\n")
    print(f"SelectKBest selected features saved to: {skb_path}")

    # ------------------------------------------------------------
    # Wrapper method: SequentialFeatureSelector (forward) with Bagging
    # ------------------------------------------------------------
    print("\n--- Wrapper: SequentialFeatureSelector (forward) with Bagging ---")

    base_tree = DecisionTreeClassifier(
        criterion="gini", max_depth=None, random_state=0
    )
    bagging = BaggingClassifier(
        estimator=base_tree,
        n_estimators=50,
        max_samples=0.8,
        max_features=1.0,
        random_state=0,
    )

    n_select = 10
    sfs = SequentialFeatureSelector(
        bagging,
        n_features_to_select=n_select,
        direction="forward",
        scoring="accuracy",
        cv=5,
        n_jobs=-1,
    )

    sfs.fit(X_train, y_train)
    sfs_mask = sfs.get_support()
    selected_features_sfs = feature_names[sfs_mask]

    print(f"SequentialFeatureSelector selected {n_select} features:")
    print(selected_features_sfs)

    sfs_path = os.path.join(RESULT_DIR, "fs_sfs.txt")
    with open(sfs_path, "w") as f:
        f.write("Selected features (SequentialFeatureSelector):\n")
        for feat in selected_features_sfs:
            f.write(feat + "\n")
    print(f"SFS selected features saved to: {sfs_path}")

    # Data reduced to wrapper-selected features
    X_train_sfs = X_train[:, sfs_mask]
    X_test_sfs = X_test[:, sfs_mask]

    # ------------------------------------------------------------
    # Grid search for Bagging hyperparameters
    # ------------------------------------------------------------
    print("\n--- Grid search for Bagging hyperparameters ---")

    bagging_base = BaggingClassifier(
        estimator=DecisionTreeClassifier(random_state=0),
        random_state=0,
    )

    param_grid = {
        "n_estimators": [20, 50, 100],
        "max_samples": [0.6, 0.8, 1.0],
        "max_features": [0.5, 1.0],
    }

    grid = GridSearchCV(
        estimator=bagging_base,
        param_grid=param_grid,
        scoring="accuracy",
        cv=5,
        n_jobs=-1,
        verbose=0,
    )

    grid.fit(X_train_sfs, y_train)

    print("Best parameters found:")
    print(grid.best_params_)
    print(f"Best CV accuracy: {grid.best_score_:.4f}")

    gs_path = os.path.join(RESULT_DIR, "grid_search_results.txt")
    with open(gs_path, "w") as f:
        f.write("Best Grid Search Parameters:\n")
        f.write(str(grid.best_params_) + "\n")
        f.write(f"Best CV accuracy: {grid.best_score_:.4f}\n")
    print(f"Grid search results saved to: {gs_path}")

    # ------------------------------------------------------------
    # Final test evaluation with best Bagging model
    # ------------------------------------------------------------
    best_bagging = grid.best_estimator_
    best_bagging.fit(X_train_sfs, y_train)
    y_pred_test = best_bagging.predict(X_test_sfs)
    test_acc = accuracy_score(y_test, y_pred_test)
    test_f1 = f1_score(y_test, y_pred_test)
    cm_test = confusion_matrix(y_test, y_pred_test)

    print("\n--- Final evaluation on test set (wrapper-selected features) ---")
    print(f"Test accuracy: {test_acc:.4f}")
    print(f"Test F1-score: {test_f1:.4f}")
    print("Confusion matrix (test):")
    print(cm_test)

    final_perf_path = os.path.join(RESULT_DIR, "final_test_performance.txt")
    with open(final_perf_path, "w") as f:
        f.write("Final Test Set Performance (Wrapper-selected features)\n")
        f.write(f"Accuracy: {test_acc:.4f}\n")
        f.write(f"F1-Score: {test_f1:.4f}\n")
        f.write("Confusion matrix:\n")
        f.write(str(cm_test) + "\n")
    print(f"Final test performance saved to: {final_perf_path}")


# --------------------------------------------------------------------
# Optional note for Wine dataset (Lecture 6 optional homework)
# --------------------------------------------------------------------
def optional_note():
    print(
        "\nNOTE: For the optional Wine homework, you can load the Wine dataset\n"
        "(e.g., sklearn.datasets.load_wine), run the same models with CV and\n"
        "apply the Friedman test (scipy.stats.friedmanchisquare) on their\n"
        "accuracy vectors. This is not implemented here but follows the same\n"
        "structure as the Wisconsin experiments."
    )


# --------------------------------------------------------------------
# Main
# --------------------------------------------------------------------
if __name__ == "__main__":
    # Lecture 6 experiments (performance evaluation)
    lecture6_experiment(n_runs=30, test_size=0.3)

    # Lecture 7 (feature selection + parameter tuning)
    lecture7_feature_selection_and_tuning()

    # Optional note
    optional_note()
