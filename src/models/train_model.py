import joblib
import sys
import pandas as pd 
from yaml import safe_load
from xgboost import XGBRegressor
from sklearn.ensemble import RandomForestRegressor
from pathlib import Path

TARGET = 'trip_duration'

def load_dataframe(data):
    df = pd.read_csv(data)
    return df

def make_X_y(dataframe:pd.DataFrame,target_column:str):
    df_copy = dataframe.copy()

    X = df_copy.drop(columns=target_column)
    y = df_copy[target_column]

    return X,y


def train_model(model,X_train,y_train):
    model.fit(X_train,y_train)

    return model

def save_model(model,save_path):
    joblib.dump(value=model,filename=save_path)

def main():

    current_path = Path(__file__)

    root_path = current_path.parent.parent.parent

    training_data_path = root_path/sys.argv[1]

    train_data = load_dataframe(training_data_path)

    X_train,y_train = make_X_y(dataframe=train_data,target_column=TARGET)

    with open('params.yaml') as f:
        params = safe_load(f)
    
    model_params = params['train_model']['random_forest_regressor']

    model_params = {k: v for k, v in model_params.items() if k != 'model'}
    
    regressor = RandomForestRegressor(**model_params)
    # regressor = XGBRegressor()

    regressor = train_model(model=regressor,X_train=X_train,y_train=y_train)

    model_output_path = root_path/'models'/'models'

    model_output_path.mkdir(exist_ok=True)

    save_model(model=regressor,save_path=model_output_path/'xgbreg.joblib')

if __name__ == "__main__":
    main()