# Breast Cancer Classification

This project uses the breast cancer dataset to compare two classification
models:

- Random Forest
- XGBoost

The complete workflow is in
[`breast cancer classification notebook.ipynb`](./breast%20cancer%20classification%20notebook.ipynb).

## Workflow

The notebook performs the following steps:

1. Loads the dataset from [`breast-cancer.csv`](./breast-cancer.csv).
2. Separates the target (`diagnosis`) from the input features.
3. Removes the identifier column (`id`).
4. Splits the data into training, validation, and test sets.
5. Fits `StandardScaler` on the training data only.
6. Applies the fitted scaler to the validation and test data.
7. Fits PCA on the scaled training data only.
8. Applies the fitted PCA transformation to the validation and test data.
9. Trains and validates a Random Forest model.
10. Trains and validates an XGBoost model.
11. Displays accuracy, precision, recall, F1 score, ROC-AUC,
    classification reports, and confusion matrices.

## Avoiding data leakage

The scaler and PCA objects must be fitted only with the training data:

```python
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)
X_test_scaled = scaler.transform(X_test)

X_train_pca = pca.fit_transform(X_train_scaled)
X_val_pca = pca.transform(X_val_scaled)
X_test_pca = pca.transform(X_test_scaled)
```

Validation and test data are only transformed using statistics and PCA
components learned from the training set. They must not be used in
`fit_transform`.

## Requirements

Install the required Python packages in the notebook environment:

```python
%pip install pandas numpy scikit-learn matplotlib seaborn xgboost
```

The notebook also requires a Jupyter-compatible Python kernel.

## Running the notebook

1. Open `breast cancer classification notebook.ipynb` in Jupyter or VS Code.
2. Select a Python kernel with the required packages installed.
3. Run the cells from top to bottom.
4. Review the validation metrics and confusion matrices for both models.

## Model evaluation

The validation set is used to compare the models during development. The
test set should remain untouched until the final model and any model
hyperparameters have been selected.

After selecting the best model, the training and validation data can be
combined, the preprocessing can be refitted on that combined data, and the
final model can then be evaluated once on the test set.

## Data

You can find the data here https://www.kaggle.com/datasets/imtkaggleteam/breast-cancer/data