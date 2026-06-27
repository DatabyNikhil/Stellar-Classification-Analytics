import duckdb
import pandas as pd

print("🚀 Starting Phase 3: Machine Learning Model Training Pipeline")

# 1. Fire up a fresh DuckDB connection instance
con = duckdb.connect()

# 2. Ingesting and transforming the data using our verified production SQL logic
print("📦 Ingesting and transforming raw data from train.csv...")
query = """
SELECT 
    id, alpha, delta, u, g, r, i, z,
    CASE WHEN redshift < 0 THEN 0 ELSE redshift END AS redshift,
    spectral_type, galaxy_population, class
FROM 'train.csv'
WHERE u > 0;
"""

# Executing the query and directly dumping the clean result into a Pandas DataFrame
df = con.execute(query).df()

# 3. Verifying the dataframe integrity and architecture
print(f"✅ Clean DataFrame Loaded Successfully! Shape: {df.shape}")
print("\n--- First 5 Rows of Cleaned Dataset ---")
print(df.head())
# 4. Deep structural checkup of column data types
print("\n--- Structural Audit of DataFrame Columns ---")
print(df.info())

# 5. Inspecting the raw categories inside text columns
print("\n--- Unique Categories in Spectral Type ---")
print(df['spectral_type'].value_counts())
# 6. Converting text columns to numbers (Categorical Encoding)
print("\n🔄 Converting categorical text columns to numeric codes...")

# Creating explicit mapping dictionaries for the columns
spectral_map = {'M': 0, 'A/F': 1, 'G/K': 2, 'O/B': 3}
galaxy_map = {'Blue_Cloud': 0, 'Red_Sequence': 1}
class_map = {'GALAXY': 0, 'STAR': 1, 'QSO': 2}

# Applying the mapping to the dataframe columns
df['spectral_type'] = df['spectral_type'].map(spectral_map)
df['galaxy_population'] = df['galaxy_population'].map(galaxy_map)
df['class'] = df['class'].map(class_map)

# 7. Verification check to confirm all columns are now numeric
print("\n--- Verification: Post-Encoding Data Types ---")
print(df[['spectral_type', 'galaxy_population', 'class']].dtypes)

print("\n--- First 5 Rows of Encoded Columns ---")
print(df[['spectral_type', 'galaxy_population', 'class']].head())
# 8. Separating Features (X) and Target Label (y)
print("\n🎯 Separating features and target label...")

# We drop 'class' because it's the target, and 'id' because it's just a serial number
X = df.drop(columns=['id', 'class'])
y = df['class']

# Verifying the shape of split datasets
print(f"📐 Features matrix shape (X): {X.shape}")
print(f"📐 Target vector shape (y): {y.shape}")

print("\n--- Final Feature Columns for Model Training ---")
print(list(X.columns))
# 9. Importing Machine Learning Splitter from Scikit-Learn
print("\n✂️ Splitting data into Training (80%) and Testing (20%) sets...")
from sklearn.model_selection import train_test_split

# Performing the structural split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

print(f"📉 Training Features Shape: {X_train.shape} | Training Labels Shape: {y_train.shape}")
print(f"📉 Testing Features Shape: {X_test.shape} | Testing Labels Shape: {y_test.shape}")
print("✅ Validation Setup Complete! Ready for Model Selection.")
# 10. Initializing and Training the LightGBM Classifier
print("\n🤖 Initializing LightGBM Classifier and starting model training...")
import lightgbm as lgb

# Setting up the model core parameters for multi-class classification
model = lgb.LGBMClassifier(
    objective='multiclass',
    num_class=3,
    random_state=42,
    n_estimators=100,
    learning_rate=0.1
)

# Fitting the model on our training dataset
print("⏳ Training model on 4.6 Lakh rows... This might take a few seconds...")
model.fit(X_train, y_train)
print("🎯 Model Training Complete Successfully!")

# 11. Evaluating the model on the unseen test data
y_pred = model.predict(X_test)

from sklearn.metrics import balanced_accuracy_score
accuracy = balanced_accuracy_score(y_test, y_pred)
print(f"🏆 Kaggle Metric Score (Balanced Accuracy) on Test Set: {accuracy * 100:.2f}%")
# 12. Loading the actual Kaggle Test Dataset for final submission
print("\n🎬 Starting Final Phase: Generating Kaggle Submission File...")

# We read test.csv and apply the exact same redshift capping logic
# CRITICAL: We DO NOT drop any rows here (No 'WHERE u > 0' filter) because Kaggle needs predictions for ALL rows.
test_query = """
SELECT 
    id, alpha, delta, u, g, r, i, z,
    CASE WHEN redshift < 0 THEN 0 ELSE redshift END AS redshift,
    spectral_type, galaxy_population
FROM 'test.csv';
"""
test_df = con.execute(test_query).df()

# Applying the exact same categorical text encoding mappings
test_df['spectral_type'] = test_df['spectral_type'].map(spectral_map)
test_df['galaxy_population'] = test_df['galaxy_population'].map(galaxy_map)

# Isolating features by dropping 'id' just like we did in training
X_kaggle_test = test_df.drop(columns=['id'])

# Running our trained 95.18% model to predict the hidden classes
print("⏳ Predicting classes for the real unseen Kaggle test dataset...")
kaggle_preds = model.predict(X_kaggle_test)

# Inverting the class map to convert numeric codes (0, 1, 2) back to text strings (GALAXY, STAR, QSO)
inverse_class_map = {0: 'GALAXY', 1: 'STAR', 2: 'QSO'}
test_df['class'] = pd.Series(kaggle_preds).map(inverse_class_map)

# Creating the final clean submission sheet containing only 'id' and 'class'
final_submission = test_df[['id', 'class']]

# Saving the final submission file inside your project directory
submission_path = 'final_submission.csv'
final_submission.to_csv(submission_path, index=False)

print(f"🎉 Boom! Final Kaggle Submission File generated successfully at: {submission_path}")
print(f"📊 Total Rows Predicted: {final_submission.shape[0]} | Columns: {final_submission.shape[1]}")