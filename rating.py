import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

# Load dataset
df = pd.read_csv("Dataset.csv")

# Select features and target
features = ['Cuisines', 'City', 'Price range', 'Average Cost for two']
target = 'Aggregate rating'

# Drop missing values
df = df[features + [target]].dropna()

# Feature engineering
df['Cuisine Count'] = df['Cuisines'].apply(lambda x: len(str(x).split(',')))
df['City Popularity'] = df['City'].map(df['City'].value_counts())

# Prepare X and y
features_extended = ['Cuisines', 'City', 'Price range', 'Average Cost for two', 'Cuisine Count', 'City Popularity']
X = df[features_extended]
y = df[target]

# Define preprocessing
categorical = ['Cuisines', 'City']
preprocessor = ColumnTransformer(
    transformers=[
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical)
    ],
    remainder='passthrough'
)

# Create pipeline
pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('regressor', RandomForestRegressor(n_estimators=100, random_state=42))
])

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train
pipeline.fit(X_train, y_train)

# Predict
y_pred = pipeline.predict(X_test)

# Evaluate
print("\n📊 Random Forest Rating Prediction Report")
print("-----------------------------------------")
print(f"Mean Squared Error: {mean_squared_error(y_test, y_pred):.3f}")
print(f"R² Score: {r2_score(y_test, y_pred):.3f}")

# Show sample predictions
sample = X_test.copy()
sample['Actual Rating'] = y_test
sample['Predicted Rating'] = y_pred
print("\n📈 Sample Predictions:")
print(sample[['City', 'Cuisines', 'Price range', 'Average Cost for two', 'Actual Rating', 'Predicted Rating']].head(10))
