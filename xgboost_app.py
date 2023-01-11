# import streamlit as st
# import xgboost as xgb
# import sklearn
# from sklearn.model_selection import train_test_split
# import pandas as pd
# import sys

# # Load data


# @st.cache(persist=True)
# def load_data():
#     try:
#         data = pd.read_csv("path/to/data.csv")
#         return data
#     except FileNotFoundError:
#         st.error("File not found at the specified location, please check the file path")
#     except Exception as e:
#         st.error("Error while loading data: " + str(e))


# data = load_data()

# if "target" not in data.columns:
#     if "Target" in data.columns:
#         st.warning("Target column found instead of target, using Target column as target")
#         data = data.rename(columns={"Target":"target"})
#     else:
#         st.error("target column not present in the data, Please check the data and add the target column")
#         sys.exit()
# # rest of the code



# # @st.cache(persist=True)
# # def load_data():
# #     try:
# #         data = pd.read_csv("path/to/data.csv")
# #         return data
# #     except FileNotFoundError:
# #         st.error("File not found at the specified location, please check the file path")
# #     except Exception as e:
# #         st.error("Error while loading data: " + str(e))



# # # Load data
# # @st.cache(persist=True)
# # def load_data():
# #     return # load your data

# # data = load_data()

# # # Split data into train and test sets
# # X = data.drop("target", axis=1)
# # y = data["target"]
# # X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# # # Define XGBoost model
# # xgb_model = xgb.XGBClassifier()

# # # Define hyperparameter tuning
# # param_grid = {
# #     'max_depth': [3, 5, 7],
# #     'n_estimators': [50, 100, 200],
# #     'learning_rate': [0.1, 0.01, 0.001]
# # }

# # # Perform hyperparameter tuning
# # xgb_cv = GridSearchCV(xgb_model, param_grid, cv=3)
# # xgb_cv.fit(X_train, y_train)

# # # Print best hyperparameters
# # st.write("Best hyperparameters:", xgb_cv.best_params_)

# # # Get test set predictions
# # y_pred = xgb_cv.predict(X_test)

# # # Show model evaluation metrics
# # st.write("Accuracy:", accuracy_score(y_test, y_pred))
# # st.write("Precision:", precision_score(y_test, y_pred))
# # st.write("Recall:", recall_score(y_test, y_pred))

# # # Show feature importance
# # importance = xgb_cv.best_estimator_.feature_importances_
# # st.bar_chart(importance)
# # Load data
# # @st.cache(persist=True)
# # def load_data():
# #     return # load your data

# # data = load_data()

# # @st.cache(persist=True)
# # def load_data():
# #     return # load your data

# # data = load_data()

# # if "target" not in data.columns:
# #     if "Target" in data.columns:
# #         st.warning("Target column found instead of target, using Target column as target")
# #         data = data.rename(columns={"Target":"target"})
# #     else:
# #         st.error("target column not present in the data, Please check the data and add the target column")
# #         return
    
    
# # if "target" not in data.columns:
# #     if "Target" in data.columns:
# #         st.warning("Target column found instead of target, using Target column as target")
# #         data = data.rename(columns={"Target":"target"})
# #     else:
# #         st.error("target column not present in the data, Please check the data and add the target column")
# #         return

# # Split data into train and test sets
# X = data.drop("target", axis=1)
# y = data["target"]
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
   
# # Define XGBoost model
# xgb_model = xgb.XGBClassifier()

# # Define hyperparameter tuning
# param_grid = {
#     'max_depth': [3, 5, 7],
#     'n_estimators': [50, 100, 200],
#     'learning_rate': [0.1, 0.01, 0.001]
# }


#     # Perform hyperparameter tuning
# xgb_cv = GridSearchCV(xgb_model, param_grid, cv=3)
# xgb_cv.fit(X_train, y_train)

#     # Print best hyperparameters
# st.write("Best hyperparameters:", xgb_cv.best_params_)

#     # Get test set predictions
# y_pred = xgb_cv.predict(X_test)

#     # Show model evaluation metrics
# st.write("Accuracy:", accuracy_score(y_test, y_pred))
# st.write("Precision:", precision_score(y_test, y_pred))
# st.write("Recall:", recall_score(y_test, y_pred))

#     # Show feature importance
# importance = xgb_cv.best_estimator_.feature_importances_
# st.bar_chart(importance)

import streamlit as st
import xgboost as xgb

# load your data
@st.cache(allow_output_mutation=True)
def load_data():
    return load_data

# train the xgboost model
@st.cache(allow_output_mutation=True)
def train_xgb(data):
    dtrain = xgb.DMatrix(data.drop(columns=['target']), label=data['target'])
    param = {'max_depth': 2, 'eta': 1, 'objective': 'binary:logistic'}
    model = xgb.train(param, dtrain, 2)
    return model

# main function
def main():
    data = load_data()
    st.write("Shape of data: ", data.shape)

    st.header("Train XGBoost Model")
    model = train_xgb(data)

    st.header("Make predictions")
    input_data = st.text_input("Enter input data as a string:")
    input_data = input_data.strip().split(',')
    input_data = [float(x) for x in input_data]
    prediction = model.predict(xgb.DMatrix(input_data))
    st.write("Prediction: ", prediction)

if __name__ == '__main__':
    main()
