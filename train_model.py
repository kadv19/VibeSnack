import pandas as pd
import numpy as np
import joblib
import os
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, FunctionTransformer
from sklearn.metrics import accuracy_score, confusion_matrix

# Feature Engineering Functions
def get_time_category(hour):
    if 7 <= hour <= 11: return "morning"
    if 12 <= hour <= 16: return "afternoon"
    if 17 <= hour <= 20: return "evening"
    return "night"

def add_time_category(X):
    # X is a DataFrame
    X = X.copy()
    # If input is a dict or single row DataFrame, handle it
    if 'hour' in X.columns:
        X['time_of_day_category'] = X['hour'].apply(get_time_category)
    return X


from sklearn.base import BaseEstimator, TransformerMixin

class TimeCategoryEncoder(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self
    def transform(self, X):
        # X is expected to be a DataFrame or 2D array with 'hour' in first column if we select it
        # But ColumnTransformer passes the selected column.
        # If we select 'hour', we get a 1D/2D array of hours.
        hours = X.iloc[:, 0] if isinstance(X, pd.DataFrame) else X[:, 0]
        cats = [get_time_category(h) for h in hours]
        return pd.DataFrame(cats, columns=['time_of_day_category'])

def train():
    print("Loading data...")
    try:
        df = pd.read_csv("snack_data.csv")
    except FileNotFoundError:
        print("Error: snack_data.csv not found. Run data_generator.py first.")
        return

    # Features and Target
    X = df[['hour', 'mood', 'hunger', 'diet', 'context']]
    y = df['snack_id'] # Predicting ID directly

    # Preprocessing Pipeline
    # 1. Time category generation (custom transformer)
    # 2. Encoding
    
    # We need to handle the 'hour' transformation within the pipeline or before.
    # To keep it simple for inference, let's do it inside the pipeline if possible, 
    # or just expect 'time_of_day_category' to be passed or derived.
    # The requirement says "Convert hour to time_of_day_category".
    # Let's use a FunctionTransformer for hour -> time_category, but we need to be careful with column names.
    # Simpler approach: The input to the model will be the raw features, and we transform them.
    
    # However, ColumnTransformer applies to specific columns.
    # Let's define the transformers.
    
    categorical_features = ['mood', 'context']
    ordinal_features = ['diet'] # veg/non-veg
    # We will derive time_category from hour, then encode it. 
    # But standard ColumnTransformer takes existing columns.
    # So we'll assume the input X has 'hour'. We can use a custom transformer to generate 'time_of_day_category' 
    # and then encode it. Or just bin 'hour' directly.
    
    # Let's try a simpler approach for the pipeline:
    # We will treat 'hour' as a numerical feature to be binned, OR we just pass 'time_of_day_category' if we pre-calculate it.
    # The prompt says "Convert hour to time_of_day_category... Use pipeline".
    # Let's make a custom transformer that takes the whole DF, adds the column, and passes it on? 
    # Scikit-learn pipelines usually work on arrays or specific columns.
    
    # Let's stick to what's robust:
    # We will have a pre-processing step that runs BEFORE the ColumnTransformer in the inference utils.
    # BUT, for the pipeline to be self-contained, it's better if it handles it.
    # Let's use 'hour' as a numerical feature or categorical? 
    # The prompt says: "Convert hour to time_of_day_category... Encode diet and time_of_day_category".
    
    # Let's create a custom transformer for the hour column.


    # Wait, if we use TimeCategoryEncoder, we then need to OneHot or Ordinal encode that result.
    # Pipeline within ColumnTransformer?
    
    time_pipe = Pipeline([
        ('time_cat', TimeCategoryEncoder()),
        ('encoder', OneHotEncoder(handle_unknown='ignore')) # Using OneHot for time categories (morning/afternoon/etc) is safer/standard
    ])
    
    preprocessor = ColumnTransformer(
        transformers=[
            ('time', time_pipe, ['hour']),
            ('cat', OneHotEncoder(handle_unknown='ignore'), ['mood', 'context']),
            ('diet', OneHotEncoder(handle_unknown='ignore'), ['diet']), # OneHot is fine for binary too
            ('num', 'passthrough', ['hunger'])
        ]
    )
    
    model = Pipeline([
        ('preprocessor', preprocessor),
        ('classifier', RandomForestClassifier(n_estimators=100, random_state=42))
    ])
    
    # Split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    print("Training model...")
    model.fit(X_train, y_train)
    
    # Evaluate
    print("Evaluating...")
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f"Accuracy: {acc:.4f}")
    
    # Top-3 Accuracy
    probs = model.predict_proba(X_test)
    # Get class labels
    classes = model.classes_
    top3_acc = 0
    for i in range(len(y_test)):
        true_label = y_test.iloc[i]
        # Get indices of top 3 probs
        top3_indices = np.argsort(probs[i])[-3:]
        top3_classes = classes[top3_indices]
        if true_label in top3_classes:
            top3_acc += 1
    top3_acc /= len(y_test)
    print(f"Top-3 Accuracy: {top3_acc:.4f}")
    
    print("\nConfusion Matrix:")
    print(confusion_matrix(y_test, y_pred))
    
    # Save
    if not os.path.exists("models"):
        os.makedirs("models")
        
    joblib.dump(model, "models/snack_model.joblib")
    print("\nModel saved to models/snack_model.joblib")

if __name__ == "__main__":
    train()
