# Loan Approval Prediction

Machine learning project for predicting loan approval status using supervised classification models.

## Project Objective

Build and compare three classifiers for binary loan approval prediction:

- Naive Bayes
- Logistic Regression
- K-Nearest Neighbors

The notebook includes data exploration, preprocessing, hyperparameter tuning, and model comparison.

## Repository Structure

- Loan_Approval_Prediction.ipynb: Main end-to-end analysis notebook.
- loan_approval_data.csv: Dataset used for training and evaluation.
- requirements.txt: Python dependencies for reproducible setup.
- .gitignore: Git ignore rules for version control.
- LICENSE: MIT License for the project.
- README.md: Project documentation.

## Dataset

The dataset contains demographic, financial, and credit-related fields, including:

- age
- income
- home_ownership
- emplyment_length
- loan_intent
- loan_amount
- loan_interest_rate
- loan_income_ratio
- payment_default_on_file
- credit_history_length
- loan_approval_status (target)

Additional columns used in the raw dataset but removed during preprocessing:

- id
- max_allowed_loan

## Workflow Summary

1. Data loading and initial inspection.
2. Exploratory data analysis (distributions, categorical plots, correlation heatmap).
3. Preprocessing:
	- Drop non-modeling columns.
	- Label encode categorical features.
	- Train-test split with stratification.
	- Feature scaling with StandardScaler.
	- Missing value handling with mean imputation.
4. Model training and hyperparameter tuning via GridSearchCV for:
	- GaussianNB
	- LogisticRegression
	- KNeighborsClassifier
5. Evaluation with:
	- Accuracy
	- Precision
	- Recall
	- F1-score
	- ROC-AUC
	- Confusion matrix and ROC curves
6. Final model comparison and best-model selection by test accuracy.

## Requirements

Python 3.10+ is required. Install dependencies using the requirements.txt file:

```bash
pip install -r requirements.txt
```

Or install manually:

```bash
pip install pandas numpy matplotlib seaborn scikit-learn jupyter
```

See [requirements.txt](requirements.txt) for specific package versions.

## How To Run

1. Clone the repository.
2. Create and activate a virtual environment:
   ```bash
   python -m venv .venv
   .venv\Scripts\activate  # On Windows
   source .venv/bin/activate  # On macOS/Linux
   ```
3. Install dependencies from requirements.txt:
   ```bash
   pip install -r requirements.txt
   ```
4. Open Loan_Approval_Prediction.ipynb in VS Code or Jupyter:
   ```bash
   jupyter notebook Loan_Approval_Prediction.ipynb
   ```
5. Run all cells in order.

## Notes

- The notebook contains tuned model runs and visual outputs.
- To reproduce current results exactly, keep the random_state values used in the notebook.

## Author

**Amryahmath**  
GitHub: [@Amryahmath](https://github.com/Amryahmath)

## Future Improvements

- Add a requirements.txt file for reproducible setup.
- Add model persistence for deployment (for example, joblib).
- Add an inference script or API endpoint for real-time predictions.