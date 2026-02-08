import pandas as pd
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
from xgboost import XGBClassifier
import joblib

def prepare_model():
    print("Step 1: Loading clinical records...")
    try:
        file_name = 'heart_failure_clinical_records_dataset.csv'
        if not os.path.exists(file_name):
            print(f"Error: {file_name} not found.")
            return False
            
        df = pd.read_csv(file_name)
        
        print("Step 2: Cleaning and Splitting data...")
        X = df.drop('DEATH_EVENT', axis=1)
        y = df['DEATH_EVENT']
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        print("Step 3: Applying SMOTE...")
        smote = SMOTE(random_state=42)
        X_train_res, y_train_res = smote.fit_resample(X_train, y_train)
        
        print("Step 4: Scaling features...")
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train_res)
        
        print("Step 5: Training Optimized XGBoost Model...")
        model = XGBClassifier(
            n_estimators=1500,
            learning_rate=0.01,
            max_depth=3,
            subsample=0.8,
            random_state=42,
            use_label_encoder=False,
            eval_metric='logloss'
        )
        
        model.fit(X_train_scaled, y_train_res)
        
        print("Step 6: Saving assets...")
        # Using joblib for BOTH model and scaler is often more stable for Streamlit
        joblib.dump(model, "best_xgb_model.pkl") 
        joblib.dump(scaler, "scaler.pkl")
        
        print("\nSuccess! 'best_xgb_model.pkl' and 'scaler.pkl' are ready.")
        return True
        
    except Exception as e:
        print(f"An error occurred: {e}")
        return False

if __name__ == "__main__":
    prepare_model()

