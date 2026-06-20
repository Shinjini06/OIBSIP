# ============================================================
#  OASIS INFOBYTE -- Data Science Internship | Task 2
#  Unemployment Analysis with Python
#  Author : Shinjini
# ============================================================

import sys
import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns
import warnings
warnings.filterwarnings("ignore")

print("=" * 55)
print("  UNEMPLOYMENT ANALYSIS - OASIS INFOBYTE TASK 2")
print("=" * 55)

# -- 1. Load Datasets
df1 = pd.read_csv(r"C:\Users\SHINJINI\Downloads\unemployment\Unemployment in India.csv")
df2 = pd.read_csv(r"C:\Users\SHINJINI\Downloads\unemployment\Unemployment_Rate_upto_11_2020.csv")

# Clean column names
df1.columns = df1.columns.str.strip()
df2.columns = df2.columns.str.strip()

print("\nDataset 1 Shape:", df1.shape)
print("Dataset 2 Shape:", df2.shape)
print("\nDataset 1 Sample:")
print(df1.head())
print("\nDataset 2 Sample:")
print(df2.head())

# -- 2. Preprocessing
df1["Date"] = pd.to_datetime(df1["Date"].str.strip(), dayfirst=True)
df1["Estimated Unemployment Rate (%)"] = pd.to_numeric(
    df1["Estimated Unemployment Rate (%)"], errors="coerce")

df2["Date"] = pd.to_datetime(df2["Date"].str.strip(), dayfirst=True)
df2["Estimated Unemployment Rate (%)"] = pd.to_numeric(
    df2["Estimated Unemployment Rate (%)"], errors="coerce")

print("\nMissing values (Dataset 1):", df1.isnull().sum().sum())
print("Missing values (Dataset 2):", df2.isnull().sum().sum())

# -- 3. Key Statistics
print("\nUnemployment Rate Stats (Dataset 1):")
print(df1["Estimated Unemployment Rate (%)"].describe())

print("\nUnemployment Rate Stats (Dataset 2):")
print(df2["Estimated Unemployment Rate (%)"].describe())

# -- 4. Visualizations
fig = plt.figure(figsize=(16, 14))
fig.suptitle("Unemployment Analysis in India - Oasis Infobyte Task 2",
             fontsize=15, fontweight="bold", y=1.01)

# 4a. Unemployment Rate Over Time (Dataset 1 - Pre-COVID)
ax1 = fig.add_subplot(3, 2, 1)
monthly_avg = df1.groupby("Date")["Estimated Unemployment Rate (%)"].mean().reset_index()
ax1.plot(monthly_avg["Date"], monthly_avg["Estimated Unemployment Rate (%)"],
         color="#4C72B0", linewidth=2, marker="o", markersize=3)
ax1.set_title("Avg Unemployment Rate Over Time\n(Pre-COVID Period)")
ax1.set_xlabel("Date")
ax1.set_ylabel("Unemployment Rate (%)")
ax1.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
plt.setp(ax1.xaxis.get_majorticklabels(), rotation=30, ha="right")

# 4b. Unemployment Rate During COVID (Dataset 2)
ax2 = fig.add_subplot(3, 2, 2)
monthly_avg2 = df2.groupby("Date")["Estimated Unemployment Rate (%)"].mean().reset_index()
ax2.plot(monthly_avg2["Date"], monthly_avg2["Estimated Unemployment Rate (%)"],
         color="#DD8452", linewidth=2, marker="o", markersize=3)
ax2.set_title("Avg Unemployment Rate Over Time\n(COVID Period: 2020)")
ax2.set_xlabel("Date")
ax2.set_ylabel("Unemployment Rate (%)")
ax2.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
plt.setp(ax2.xaxis.get_majorticklabels(), rotation=30, ha="right")

# 4c. Top 10 States by Avg Unemployment (Dataset 1)
ax3 = fig.add_subplot(3, 2, 3)
state_avg = (df1.groupby("Region")["Estimated Unemployment Rate (%)"]
             .mean().sort_values(ascending=False).head(10))
colors = sns.color_palette("Blues_r", len(state_avg))
ax3.barh(state_avg.index, state_avg.values, color=colors, edgecolor="white")
ax3.set_title("Top 10 States by Avg\nUnemployment Rate")
ax3.set_xlabel("Unemployment Rate (%)")
ax3.invert_yaxis()

# 4d. Rural vs Urban
ax4 = fig.add_subplot(3, 2, 4)
if "Area" in df1.columns:
    area_avg = df1.groupby("Area")["Estimated Unemployment Rate (%)"].mean()
    ax4.bar(area_avg.index, area_avg.values,
            color=["#4C72B0", "#55A868"], edgecolor="white", width=0.4)
    ax4.set_title("Rural vs Urban\nUnemployment Rate")
    ax4.set_ylabel("Unemployment Rate (%)")
else:
    ax4.text(0.5, 0.5, "Area data not available", ha="center", va="center")
    ax4.set_title("Rural vs Urban")

# 4e. State-wise Heatmap (Dataset 2)
ax5 = fig.add_subplot(3, 1, 3)
df2["Month"] = df2["Date"].dt.strftime("%b %Y")
pivot = df2.pivot_table(
    values="Estimated Unemployment Rate (%)",
    index="Region",
    columns="Month",
    aggfunc="mean"
)
pivot = pivot.dropna(thresh=3)
sns.heatmap(pivot, cmap="YlOrRd", ax=ax5, linewidths=0.3,
            cbar_kws={"label": "Unemployment Rate (%)"})
ax5.set_title("State-wise Unemployment Rate Heatmap (2020)", fontsize=11)
ax5.set_xlabel("")
ax5.set_ylabel("State")
plt.setp(ax5.xaxis.get_majorticklabels(), rotation=30, ha="right")

plt.tight_layout()
plt.savefig(r"C:\Users\SHINJINI\Downloads\unemployment_results.png", dpi=150, bbox_inches="tight")
plt.show()
print("\nPlot saved as unemployment_results.png in Downloads folder")

# -- 5. Key Insights
print("\nKEY INSIGHTS:")
print(f"  Average Unemployment Rate (pre-COVID): {df1['Estimated Unemployment Rate (%)'].mean():.2f}%")
print(f"  Average Unemployment Rate (2020 COVID): {df2['Estimated Unemployment Rate (%)'].mean():.2f}%")
print(f"  Highest recorded rate (2020): {df2['Estimated Unemployment Rate (%)'].max():.2f}%")
peak_row = df2.loc[df2["Estimated Unemployment Rate (%)"].idxmax()]
print(f"  Peak state: {peak_row['Region']} on {peak_row['Date'].strftime('%B %Y')}")

print("\nTask 2 Complete! Push this file and unemployment_results.png to your OIBSIP repo.")
