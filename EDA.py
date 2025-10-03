import pandas as pd

# Load dataset
df = pd.read_csv("Dataset.csv")

# ===== Basic Overview =====
print("\n🔷 Dataset Shape:", df.shape)
print("\n🔷 Column Names:", df.columns.tolist())

# ===== Data Types =====
print("\n🔷 Data Types:\n", df.dtypes)

# ===== Missing Values =====
print("\n🔷 Missing Values:\n", df.isnull().sum())

# ===== Summary Statistics =====
print("\n🔷 Summary Statistics:\n", df.describe(include='all'))

# ===== Unique Counts =====
print("\n🔷 Unique Counts:")
print("• Unique Cities:", df['City'].nunique())
print("• Unique Cuisines (raw):", df['Cuisines'].nunique())
print("• Unique Price Ranges:", df['Price range'].nunique())
print("• Unique Ratings:", df['Aggregate rating'].nunique())

# ===== Top 10 Cities =====
print("\n🔷 Top 10 Cities by Restaurant Count:")
print(df['City'].value_counts().head(10))

# ===== Top 10 Primary Cuisines =====
print("\n🔷 Top 10 Cuisines (Primary only):")
primary_cuisine = df['Cuisines'].dropna().apply(lambda x: str(x).split(',')[0].strip())
print(primary_cuisine.value_counts().head(10))

# ===== Price Range Distribution =====
print("\n🔷 Price Range Distribution:")
print(df['Price range'].value_counts().sort_index())

# ===== Rating Distribution =====
print("\n🔷 Rating Distribution:")
print(df['Aggregate rating'].value_counts().sort_index())

# ===== Additional Insight (Optional) =====
print("\n💡 Restaurants with No Ratings:", len(df[df['Aggregate rating'] == 0]))
print("💡 Percentage Unrated: {:.2f}%".format(len(df[df['Aggregate rating'] == 0]) / len(df) * 100))
