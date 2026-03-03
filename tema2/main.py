# tips_linear_regression_save.py

import pandas as pd
import statsmodels.api as sm
import matplotlib.pyplot as plt
import seaborn as sns
import os


output_dir = "tips_results"
os.makedirs(output_dir, exist_ok=True)


tips = pd.read_csv("tips.csv")

tips['sex'] = tips['sex'].map({'Male': 0, 'Female': 1})
tips['smoker'] = tips['smoker'].map({'No': 0, 'Yes': 1})
tips['time'] = tips['time'].map({'Lunch': 0, 'Dinner': 1})
day_map = {'Thur': 3, 'Fri': 4, 'Sat': 5, 'Sun': 6}
tips['day'] = tips['day'].map(day_map)


tips['tiprate'] = tips['tip'] / tips['total_bill']

X = tips[['total_bill', 'sex', 'smoker', 'day', 'time', 'size']]
y = tips['tiprate']
X = sm.add_constant(X)


model = sm.OLS(y, X).fit()


summary_text = model.summary().as_text()
with open(os.path.join(output_dir, "linear_regression_summary.txt"), "w") as f:
    f.write(summary_text)
                              
coef_table = pd.DataFrame({
    "Variable": model.params.index,
    "Coefficient": model.params.values,
    "P-value": model.pvalues.values,
    "Std Err": model.bse.values
})
coef_table.to_csv(os.path.join(output_dir, "model_coefficients.csv"), index=False)


plt.figure(figsize=(6,4))
sns.scatterplot(x='size', y='tiprate', data=tips, alpha=0.7)
sns.regplot(x='size', y='tiprate', data=tips, scatter=False, color='red')
plt.title("Tip Rate vs Party Size")
plt.xlabel("Party Size")
plt.ylabel("Tip Rate (tip / total bill)")
plt.tight_layout()
plt.savefig(os.path.join(output_dir, "tiprate_vs_size.png"), dpi=300)
plt.close()


plt.figure(figsize=(6,4))
sns.residplot(x=model.fittedvalues, y=model.resid, lowess=True, color='purple')
plt.xlabel("Fitted Values")
plt.ylabel("Residuals")
plt.title("Residuals vs Fitted Values")
plt.tight_layout()
plt.savefig(os.path.join(output_dir, "residuals_plot.png"), dpi=300)
plt.close()

plt.figure(figsize=(6,4))
sns.histplot(tips['tiprate'], bins=20, kde=True, color='skyblue')
plt.title("Distribution of Tip Rate")
plt.xlabel("Tip Rate (tip / total bill)")
plt.tight_layout()
plt.savefig(os.path.join(output_dir, "tiprate_distribution.png"), dpi=300)
plt.close()

print(" Analysis complete! Files saved in folder:", output_dir)
