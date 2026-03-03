import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error, r2_score

from sklearn.tree import DecisionTreeRegressor, plot_tree
from sklearn.ensemble import (
    BaggingRegressor,
    RandomForestRegressor,
    GradientBoostingRegressor
)
from sklearn.neural_network import MLPRegressor

# ======================================================
# 0. Settings
# ======================================================
DATA_FILE = "CarPrice.csv"  
TARGET_COL = "price"        
OUTPUT_DIR = "outputs"

os.makedirs(OUTPUT_DIR, exist_ok=True)

# ======================================================
# 1. Load data
# ======================================================
df = pd.read_csv(DATA_FILE)
df = df.dropna(axis=0)

X = df.drop(columns=[TARGET_COL])
y = df[TARGET_COL]

# ======================================================
# 2. Train–test split
# ======================================================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ======================================================
# 3. Preprocessing
# ======================================================
numeric_cols = X.select_dtypes(include=["int64", "float64"]).columns
categorical_cols = X.select_dtypes(include=["object", "category"]).columns

preprocessor = ColumnTransformer(
    transformers=[
        ("num", StandardScaler(), numeric_cols),
        ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_cols),
    ]
)

# ======================================================
# 4. Evaluation helper
# ======================================================
results = []

def evaluate_model(pipeline, name):
    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_test)

    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, y_pred)

    results.append({"model": name, "rmse": rmse, "r2": r2})

    print(f"{name}: RMSE={rmse:.3f}, R2={r2:.3f}")
    print("-" * 40)


# ======================================================
# 5. Decision Tree
# ======================================================
dt_model = DecisionTreeRegressor(max_depth=4, random_state=42)

dt_pipeline = Pipeline([
    ("preprocess", preprocessor),
    ("model", dt_model)
])

evaluate_model(dt_pipeline, "Decision Tree")


# Save the tree plot
dt_pipeline.fit(X_train, y_train)

tree = dt_pipeline.named_steps["model"]
fitted_preprocessor = dt_pipeline.named_steps["preprocess"]

cat_feature_names = []
if categorical_cols.any():
    try:
        ohe = fitted_preprocessor.named_transformers_["cat"]
        cat_feature_names = ohe.get_feature_names_out(categorical_cols)
    except:
        cat_feature_names = ohe.get_feature_names(categorical_cols)

all_feature_names = np.concatenate([numeric_cols, cat_feature_names])

fig = plt.figure(figsize=(24, 12))
plot_tree(tree, feature_names=all_feature_names, filled=True, rounded=True, max_depth=3, fontsize=8)
plt.title("Decision Tree (Top 3 Levels)")

tree_path = os.path.join(OUTPUT_DIR, "decision_tree_top3.png")
fig.savefig(tree_path, dpi=300, bbox_inches="tight")
plt.close(fig)


# ======================================================
# 6. Bagging (fixed for your sklearn!)
# ======================================================
bagging_model = BaggingRegressor(
    DecisionTreeRegressor(random_state=42),  # estimator goes as positional argument
    n_estimators=100,
    random_state=42,
    n_jobs=-1
)

bagging_pipeline = Pipeline([
    ("preprocess", preprocessor),
    ("model", bagging_model)
])

evaluate_model(bagging_pipeline, "Bagging")


# ======================================================
# 7. Random Forest
# ======================================================
rf_model = RandomForestRegressor(
    n_estimators=200,
    random_state=42,
    n_jobs=-1
)

rf_pipeline = Pipeline([
    ("preprocess", preprocessor),
    ("model", rf_model)
])

evaluate_model(rf_pipeline, "Random Forest")


# ======================================================
# 8. Gradient Boosting
# ======================================================
gb_model = GradientBoostingRegressor(
    n_estimators=200,
    learning_rate=0.05,
    max_depth=3,
    random_state=42
)

gb_pipeline = Pipeline([
    ("preprocess", preprocessor),
    ("model", gb_model)
])

evaluate_model(gb_pipeline, "Gradient Boosting")


# ======================================================
# 9. Neural Network
# ======================================================
nn_model = MLPRegressor(
    hidden_layer_sizes=(64, 32),
    max_iter=500,
    random_state=42
)

nn_pipeline = Pipeline([
    ("preprocess", preprocessor),
    ("model", nn_model)
])

evaluate_model(nn_pipeline, "Neural Network")


# ======================================================
# 10. Save results table
# ======================================================
results_df = pd.DataFrame(results)
results_csv = os.path.join(OUTPUT_DIR, "model_results.csv")
results_df.to_csv(results_csv, index=False)
print("Saved results to:", results_csv)


# ======================================================
# 11. Save RMSE bar plot
# ======================================================
fig2 = plt.figure(figsize=(8, 5))
plt.bar(results_df["model"], results_df["rmse"])
plt.ylabel("RMSE")
plt.title("Model Comparison (RMSE)")
plt.xticks(rotation=45, ha="right")

rmse_plot = os.path.join(OUTPUT_DIR, "model_rmse.png")
fig2.savefig(rmse_plot, dpi=300, bbox_inches="tight")
plt.close(fig2)

print("Saved RMSE plot to:", rmse_plot)
