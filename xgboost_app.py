import streamlit as st
import xgboost as xgb
from sklearn.model_selection import train_test_split

# Load data
@st.cache(persist=True)
def load_data():
    return # load your data

data = load_data()

# Split data into train and test sets
X = data.drop("target", axis=1)
y = data["target"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# Define XGBoost model
xgb_model = xgb.XGBClassifier()

# Define hyperparameter tuning
param_grid = {
    'max_depth': [3, 5, 7],
    'n_estimators': [50, 100, 200],
    'learning_rate': [0.1, 0.01, 0.001]
}

# Perform hyperparameter tuning
xgb_cv = GridSearchCV(xgb_model, param_grid, cv=3)
xgb_cv.fit(X_train, y_train)

# Print best hyperparameters
st.write("Best hyperparameters:", xgb_cv.best_params_)

# Get test set predictions
y_pred = xgb_cv.predict(X_test)

# Show model evaluation metrics
st.write("Accuracy:", accuracy_score(y_test, y_pred))
st.write("Precision:", precision_score(y_test, y_pred))
st.write("Recall:", recall_score(y_test, y_pred))

# Show feature importance
importance = xgb_cv.best_estimator_.feature_importances_
st.bar_chart(importance)

