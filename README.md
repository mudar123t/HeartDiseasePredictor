# Heart Disease Prediction Project

This project uses machine learning to predict the risk of heart disease based on medical features.  
The goal is to test different models understand the effect of data quality and build a stable prediction system.

---

## Project Structure

project/
│
├── notebooks/                    # Work on the old dataset
│   ├── 01_exploration.ipynb
│   ├── 02_preprocessing.ipynb
│   ├── 03_feature_selection.ipynb
│   ├── 04_modeling.ipynb
│   └── 04_feature_modeling.ipynb
│
├── notebooks_new/                # Work on the new dataset
│   ├── 01_exploration_new.ipynb
│   ├── 02_feature_modeling_new.ipynb
│   └── 03_evaluation.ipynb
│
├── src/
│   ├── feature_selection.py
│   └── models.py
│
├── data/
│   ├── heart_disease.csv         # Old dataset
│   └── heart.csv                 # New dataset
│
└── results/
    ├── model_results.csv
    └── evaluation_plots/



---

## Datasets

### **Old Dataset**
- Produced weak and unstable results.  
- Feature selection was inconsistent and models performed poorly.  
- Data quality issues affected model learning.

### **New Dataset**
- A second dataset from Kaggle with no missing values.  
- Clearer feature patterns and better structure.  
- Models performed much better and more consistently.

---

## Main Steps

### **1. Exploration**
We checked dataset structure looked for missing values and studied feature distributions.

### **2. Preprocessing**
We encoded categorical features scaled numerical values and split the data into training and testing sets.

### **3. Feature Selection**
We used statistical and model based methods to find the most important features especially for the old dataset.

### **4. Modeling**
We trained several machine learning models including  
KNN Decision Tree SVM Gradient Boosting AdaBoost and XGBoost.  
We compared accuracy precision recall and F1 scores.

### **5. Optimization**
Some models showed signs of overfitting.  
We tuned hyperparameters to reach more balanced performance.

### **6. Evaluation**
We analyzed metrics confusion matrices feature importance and ROC curves.

---

## Key Findings

- The old dataset produced weak and unstable results.  
- The new dataset gave higher accuracy and better model stability.  
- Boosting models performed the best especially AdaBoost and Gradient Boosting.  
- Data quality had a major effect on the final performance.

---

## Technologies Used

- Python  
- pandas numpy matplotlib seaborn  
- scikit-learn  
- XGBoost  
- Jupyter Notebook  

---

## How to Run the Project

### 1. Install required libraries

pip install -r requirements.txt



### 2. Open the notebooks in the `notebooks_new` folder  
### 3. Run the exploration modeling and evaluation steps in order  
### 4. Check the `results` folder for metrics and plots  

---

## Credits

This project was completed as part of the Introduction to Data Science course.  
**Instructor:** Zeki Kus  
**Students:** Mudar Shawakh and MHD Alhabeb Alshalah  
