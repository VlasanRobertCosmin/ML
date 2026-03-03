# SVM on the Wisconsin Breast Cancer dataset — with result
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler, OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC, LinearSVC
from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay, accuracy_score
from sklearn.impute import SimpleImputer
from sklearn.decomposition import PCA

warnings.filterwarnings("ignore")

# ==========================================================
# Setup paths
# ==========================================================
DATA_PATH = "breast-cancer-wisconsin.csv"
RESULTS_DIR = "results"
os.makedirs(RESULTS_DIR, exist_ok=True)

# ==========================================================
# Load and clean dataset
# ==========================================================
df = pd.read_csv(DATA_PATH, na_values=["?", "NA", "NaN", "nan", "", " "])
df = df.drop(df.columns[0], axis=1)  # drop first column (non-informative)
target_col = df.columns[-1]
X = df.drop(columns=[target_col])
y = df[target_col].copy()

# Encode target (2/4 → 0/1, or B/M → 0/1, etc.)
if y.dtype == object:
    mapping = {"benign": 0, "malignant": 1, "b": 0, "m": 1, "2": 0, "4": 1}
    y = y.astype(str).str.strip().str.lower().map(mapping).fillna(y)
y = pd.factorize(y)[0] if not np.issubdtype(y.dtype, np.number) else y.astype(int)

# Numeric/categorical columns
num_cols = X.select_dtypes(include=[np.number]).columns.tolist()
cat_cols = [c for c in X.columns if c not in num_cols]

numeric_transform = Pipeline([("imputer", SimpleImputer(strategy="mean"))])
categorical_transform = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("onehot", OneHotEncoder(handle_unknown="ignore"))
])
pre_base = ColumnTransformer([
    ("num", numeric_transform, num_cols),
    ("cat", categorical_transform, cat_cols)
])

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# ==========================================================
# Model training — GridSearch over scalers + kernels
# ==========================================================
scalers = {
    "standard": StandardScaler(),
    "minmax": MinMaxScaler(),
    "robust": RobustScaler()
}

param_grid = [{
    "clf": [SVC()],
    "clf__kernel": ["linear", "rbf", "poly"],
    "clf__C": [0.1, 1, 10],
    "clf__gamma": ["scale", "auto"],
    "clf__degree": [2, 3]
}]

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
results = []
best_models = {}

def make_pipeline(scaler, estimator):
    return Pipeline([("pre", pre_base), ("scale", scaler), ("clf", estimator)])

for scaler_name, scaler in scalers.items():
    pipe = make_pipeline(scaler, SVC())
    grid = GridSearchCV(pipe, param_grid, cv=cv, scoring="accuracy", n_jobs=-1)
    grid.fit(X_train, y_train)

    best_models[scaler_name] = grid.best_estimator_
    y_pred = grid.predict(X_test)
    acc = accuracy_score(y_test, y_pred)

    results.append({
        "Scaler": scaler_name,
        "Best Params": str(grid.best_params_),
        "CV Mean Accuracy": grid.best_score_,
        "Test Accuracy": acc
    })

# Save results summary
results_df = pd.DataFrame(results)
results_csv_path = os.path.join(RESULTS_DIR, "results_summary.csv")
results_df.to_csv(results_csv_path, index=False)
print(f"\n Saved summary to {results_csv_path}")

# Pick the best model
best_scaler_name = results_df.loc[results_df["CV Mean Accuracy"].idxmax(), "Scaler"]
best_model = best_models[best_scaler_name]
print(f"\n Best model uses scaler: {best_scaler_name}")
print(best_model)

# ==========================================================
# Save Confusion Matrix plot
# ==========================================================
y_pred = best_model.predict(X_test)
fig, ax = plt.subplots(figsize=(5, 5))
ConfusionMatrixDisplay.from_predictions(y_test, y_pred, ax=ax)
ax.set_title(f"Confusion Matrix — {best_scaler_name}")
plt.tight_layout()
cm_path = os.path.join(RESULTS_DIR, "confusion_matrix.png")
plt.savefig(cm_path)
plt.close()
print(f"✅ Saved confusion matrix to {cm_path}")

# ==========================================================
# Variable importance plot
# ==========================================================
def get_feature_names(preprocessor):
    feature_names = []
    feature_names.extend(num_cols)
    if cat_cols:
        ohe = preprocessor.named_transformers_["cat"].named_steps["onehot"]
        feature_names.extend(ohe.get_feature_names_out(cat_cols))
    return feature_names

feature_names = get_feature_names(best_model.named_steps["pre"])
clf = best_model.named_steps["clf"]
kernel = getattr(clf, "kernel", None)
importance_path = os.path.join(RESULTS_DIR, "variable_importance.png")

if kernel == "linear":
    coefs = clf.coef_.ravel()
    order = np.argsort(np.abs(coefs))[::-1][:15]
    plt.figure(figsize=(7, 5))
    plt.barh(np.array(feature_names)[order][::-1], np.array(coefs)[order][::-1])
    plt.title("Top Features (Linear SVM Coefficients)")
else:
    r = permutation_importance(best_model, X_test, y_test, n_repeats=10, random_state=42)
    importances = r.importances_mean
    order = np.argsort(np.abs(importances))[::-1][:15]
    plt.figure(figsize=(7, 5))
    plt.barh(np.array(feature_names)[order][::-1], np.array(importances)[order][::-1])
    plt.title("Top Features (Permutation Importance)")

plt.xlabel("Importance (absolute)")
plt.tight_layout()
plt.savefig(importance_path)
plt.close()
print(f"✅ Saved variable importance to {importance_path}")

# ==========================================================
# PCA visualization of support vectors
# ==========================================================
from sklearn.decomposition import PCA
Z_train = best_model.named_steps["scale"].fit_transform(
    pre_base.fit_transform(X_train)
)
Z_test = best_model.named_steps["scale"].transform(
    pre_base.transform(X_test)
)
pca = PCA(n_components=2, random_state=42)
Z_train2 = pca.fit_transform(Z_train)
Z_test2 = pca.transform(Z_test)

viz_svm = SVC(kernel="rbf", C=1.0, gamma="scale")
viz_svm.fit(Z_train2, y_train)

h = 0.02
x_min, x_max = Z_train2[:, 0].min() - 1, Z_train2[:, 0].max() + 1
y_min, y_max = Z_train2[:, 1].min() - 1, Z_train2[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                     np.arange(y_min, y_max, h))
Z_pred = viz_svm.predict(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)

plt.figure(figsize=(7, 6))
plt.contourf(xx, yy, Z_pred, alpha=0.25)
plt.scatter(Z_train2[:, 0], Z_train2[:, 1], c=y_train, s=40, edgecolor="k")
plt.scatter(viz_svm.support_vectors_[:, 0], viz_svm.support_vectors_[:, 1],
            s=80, facecolors="none", edgecolors="k", linewidths=1.5, label="Support Vectors")
plt.legend()
plt.title("SVM Decision Surface (PCA projection)")
plt.xlabel("PCA 1")
plt.ylabel("PCA 2")
plt.tight_layout()
pca_plot_path = os.path.join(RESULTS_DIR, "svm_pca_surface.png")
plt.savefig(pca_plot_path)
plt.close()
print(f"✅ Saved PCA visualization to {pca_plot_path}")

print("\nAll done! Check your 'results/' folder for CSV + plots.")
