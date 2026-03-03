import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


df = pd.read_csv("CAR DETAILS FROM CAR DEKHO.csv")


with open("results.txt", "w") as f:
    f.write("=== VEHICLE DATA ANALYSIS ===\n\n")

    f.write(f"Total number of cars: {len(df)}\n")
    f.write(f"Columns: {list(df.columns)}\n\n")

   
    min_price = df["selling_price"].min()
    max_price = df["selling_price"].max()

    car_min = df[df["selling_price"] == min_price]
    car_max = df[df["selling_price"] == max_price]

    f.write("Car with MIN price:\n")
    f.write(car_min.to_string(index=False))
    f.write("\n\nCar with MAX price:\n")
    f.write(car_max.to_string(index=False))
    f.write("\n\n")


    threshold = 500000  # you can change this
    count_above = (df["selling_price"] > threshold).sum()
    f.write(f"Cars priced above {threshold}: {count_above}\n\n")

    
    mean_price = df["selling_price"].mean()
    std_price = df["selling_price"].std()

    f.write(f"📊 Mean price: {mean_price:.2f}\n")
    f.write(f"📊 Standard deviation: {std_price:.2f}\n\n")

plt.figure(figsize=(8, 5))
plt.scatter(df["km_driven"], df["selling_price"], alpha=0.5)
plt.xlabel("Mileage (km)")
plt.ylabel("Price (₹)")
plt.title("Mileage vs Price")
plt.grid(True)
plt.tight_layout()
plt.savefig("plot_mileage_vs_price.png", dpi=300)
plt.close()


fig, axes = plt.subplots(1, 2, figsize=(10, 4))

sns.boxplot(y=df["selling_price"], ax=axes[0])
axes[0].set_title("Boxplot - Price")

sns.violinplot(y=df["selling_price"], ax=axes[1])
axes[1].set_title("Violin Plot - Price")

plt.tight_layout()
plt.savefig("plot_box_violin_price.png", dpi=300)
plt.close()


print("Results saved in: results.txt")

