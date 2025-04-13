import sys
import joblib
import pandas as pd 
from pathlib import Path
from sklearn.metrics import r2_score

TARGET = 'trip_duration'

model_name = 'xgbreg.joblib'

def load_dataframe(path):
    df = pd.read_csv(path)
    return df

def make_X_y(dataframe:pd.DataFrame,target_column:str):
    dataframe = dataframe.copy()

    X = dataframe.drop(columns=target_column)
    y = dataframe[target_column]

    return X,y

def get_predictions(model,x:pd.DataFrame):
    y_pred = model.predict(x)

    return y_pred

def calculate_r2_score(y_actual,y_predicted):
    score = r2_score(y_true=y_actual,y_pred=y_predicted)

    return score

def main():

    current_path = Path(__file__)

    root_path = current_path.parent.parent.parent

    for ind in range(1,3):

        data_path = root_path/'data'/'processed'/'final'/sys.argv[ind]

        data = load_dataframe(path=data_path)

        X_test,y_test = make_X_y(dataframe=data,target_column=TARGET)

        model_path = root_path/'models'/'models'/model_name


        model = joblib.load(model_path)

        y_pred = get_predictions(model=model,x=X_test)

        score = calculate_r2_score(y_actual=y_test,y_predicted=y_pred)

        print(f'\nThe score for dataset {sys.argv[ind]} is {score}')
    
if __name__ == "__main__":
    main()
