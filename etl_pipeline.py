import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer
import os

# Load data
df = pd.read_csv('data/raw_data.csv')  # Read raw CSV file

# Identify column types
num_cols = df.select_dtypes(include=['float64', 'int64']).columns
cat_cols = df.select_dtypes(include=['object']).columns

# Impute missing values
df[num_cols] = SimpleImputer(strategy='mean').fit_transform(df[num_cols])
df[cat_cols] = SimpleImputer(strategy='most_frequent').fit_transform(df[cat_cols])

# Encode categorical variables
for col in cat_cols:
    df[col] = LabelEncoder().fit_transform(df[col])

# Scale numerical features
df[num_cols] = StandardScaler().fit_transform(df[num_cols])

# Save the cleaned data
os.makedirs("output", exist_ok=True)
df.to_csv("output/cleaned_data.csv", index=False)

print("✅ ETL pipeline completed. Cleaned data saved to output/cleaned_data.csv")
