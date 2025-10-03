import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, accuracy_score

# Load dataset
df = pd.read_csv("Dataset.csv")

# Select relevant features
features = ['City', 'Price range', 'Average Cost for two', 'Aggregate rating', 'Votes']
target = 'Cuisines'

# Drop rows with missing values
df = df[features + [target]].dropna()

# Simplify cuisines to the first type
df['Primary Cuisine'] = df['Cuisines'].apply(lambda x: str(x).split(',')[0].strip())

# Keep only the top 10 cuisines
top_cuisines = df['Primary Cuisine'].value_counts().nlargest(10).index
df = df[df['Primary Cuisine'].isin(top_cuisines)]

# Define X and y
X = df[features]
y = df['Primary Cuisine']

# Preprocessing: encode categorical, pass through numerical
categorical = ['City']
preprocessor = ColumnTransformer([
    ('cat', OneHotEncoder(handle_unknown='ignore'), categorical)
], remainder='passthrough')

# Pipeline
pipeline = Pipeline([
    ('preprocessor', preprocessor),
    ('classifier', RandomForestClassifier(n_estimators=100, random_state=42))
])

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train
pipeline.fit(X_train, y_train)

# Predict
y_pred = pipeline.predict(X_test)

# Accuracy and report
accuracy = accuracy_score(y_test, y_pred) * 100
print(f"\n✅ Overall Accuracy: {accuracy:.2f}%")

# Classification report in percentage
report = classification_report(y_test, y_pred, output_dict=True)
print("\n📊 Cuisine Classification Report (%):")
for label, metrics in report.items():
    if label in ['accuracy', 'macro avg', 'weighted avg']:
        continue
    print(f"\nCuisine: {label}")
    print(f"  Precision: {metrics['precision'] * 100:.2f}%")
    print(f"  Recall:    {metrics['recall'] * 100:.2f}%")
    print(f"  F1-score:  {metrics['f1-score'] * 100:.2f}%")
    print(f"  Support:   {metrics['support']}")

print(f"\n🎯 Final Accuracy: {accuracy:.2f}%")
