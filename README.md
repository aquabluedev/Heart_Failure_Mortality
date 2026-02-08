# Heart Failure Mortality Prediction

This project uses Machine Learning to predict the mortality risk of patients with heart failure based on 12 clinical features.

## Research & Model Selection
In this study, I compared three distinct approaches to identify the most robust predictor for clinical heart failure mortality data:
1. **Tree-Based:** XGBoost
2. **Geometric:** Support Vector Machines (SVM)
3. **Deep Learning:** Artificial Neural Networks (PyTorch)

**Final Selection:** **XGBoost** was selected as the production model. It achieved an **82% accuracy** and a **0.70 F1-score**, which is the most effective among three approaches at balancing precision and recall on our data.

## Technical Workflow
- **Data Engineering:** Used **SMOTE** over-sampling to address class imbalance (32% mortality rate) and **StandardScaler** for feature normalization.
- **Optimization:** Conducted hyperparameter tuning via **GridSearchCV**.
- **Interpretability:** Analyzed feature importance to confirm that **Serum Creatinine**, **Ejection Fraction**, and **Age** are the primary biological drivers in the model.
- **Deployment:** Developed an interactive dashboard using **Streamlit**.

## How to Run
1. **Clone the repository:**
   ```bash
   git clone [https://github.com/YOUR_USERNAME/heart-failure-prediction.git](https://github.com/YOUR_USERNAME/heart-failure-prediction.git)

2. **Install dependencies:**
    ```
    pip install -r requirements.txt

3. **Prepare the model:**
    ```
    python model.py

4. **Launch the dashboard:**
    ```
    streamlit run app.py

## Dataset
The dataset contains 299 clinical records of heart failure patients, originally sourced from the BMC Medical Informatics and Decision Making journal.


