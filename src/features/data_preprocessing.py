import joblib
import sys
import numpy as np
import pandas as pd
from yaml import safe_load
from pathlib import Path
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler, MinMaxScaler, PowerTransformer
from src.features.outliers_removal import OutlierRemover

COLUMN_NAMES = ['pickup_latitude',
                'pickup_longitude',
                'dropoff_latitude',
                'dropoff_longitude']

TARGET_COLUMN = "trip_duration"

def save_transformer(path,object):
    joblib.dump(value=object,filename=path)

def remove_outliers(dataframe:pd.DataFrame,percentiles:list,column_names:list) -> pd.DataFrame:
    df = dataframe.copy()

    outlier_transformer = OutlierRemover(percentile_values=percentiles,col_subset=column_names)

    outlier_transformer.fit(dataframe)

    return outlier_transformer

def train_preprocessor(data:pd.DataFrame):
    ohe_columns = ['vendor_id']
    standard_scale_columns = ['haversine_distance','euclidean_distance','manhattan_distance']
    min_max_scale_columns = ['pickup_latitude','pickup_longitude','dropoff_latitude','dropoff_longitude']

    preprocessor = ColumnTransformer(transformers=[('one_hot',OneHotEncoder(drop='first',sparse_output=False,handle_unknown='ignore'),ohe_columns),
                                                   ('min_max',MinMaxScaler(),min_max_scale_columns),
                                                   ('standard_scale',StandardScaler(),standard_scale_columns)],
                                                   remainder='passthrough',n_jobs=1,verbose_feature_names_out=False)
    
    preprocessor.set_output(transform='pandas')

    preprocessor.fit(data)

    return preprocessor


def transform_data(transformer,data:pd.DataFrame):

    data_transformed = transformer.transform(data)

    return data_transformed


def transform_output(target:pd.Series):
    power_transfomer = PowerTransformer(method='yeo-johnson',standardize=True)

    target_transformed = power_transfomer.fit(target.values.reshape(-1,1))

    return power_transfomer

def read_dataframe(path):
    df = pd.read_csv(path)

    return df

def save_dataframe(dataframe:pd.DataFrame,save_path):
    dataframe.to_csv(save_path,index=False)


def main():
    current_path = Path(__file__)

    root_path = current_path.parent.parent.parent

    input_path = root_path/'data'/'processed'/'build-features'

    with open('params.yaml') as f:
        params = safe_load(f)

    percentiles = list(params['data_preprocessing']['percentiles'])

    save_transformers_path = root_path/'models'/'transformers'

    save_transformers_path.mkdir(exist_ok=True)

    save_data_path = root_path/'data'/'processed'/'final'

    save_data_path.mkdir(exist_ok=True)

    for filename in sys.argv[1:]:
        complete_input_path = input_path/filename

        if (filename=='train.csv') or (filename=='val.csv'):
            df = read_dataframe(complete_input_path)

            X = df.drop(columns=TARGET_COLUMN)

            y = df[TARGET_COLUMN]

            outlier_transformer = remove_outliers(dataframe=X,percentiles=percentiles,column_names=COLUMN_NAMES)

            save_transformer(path=save_transformers_path/'outliers.joblib',
                             object=outlier_transformer)
            
            df_without_outliers = transform_data(transformer=outlier_transformer,data=X)

            preprocessor = train_preprocessor(data=df_without_outliers)

            save_transformer(path=save_transformers_path/'preprocessor.joblib',object=preprocessor)

            X_trans = transform_data(transformer=preprocessor,data=X)

            output_transformer = transform_output(y)

            y_trans = transform_data(transformer=output_transformer,data=y.values.reshape(-1,1))

            X_trans['trip_duration'] = y_trans

            save_transformer(path=save_transformers_path/'output_transformer.joblib',object=output_transformer)

            save_dataframe(dataframe=X_trans,save_path=save_data_path/filename)
        elif filename == 'val.csv':
            df = read_dataframe(complete_input_path)
            # make X and y
            X = df.drop(columns=TARGET_COLUMN)
            y = df[TARGET_COLUMN]
            # load the transfomer
            outlier_transformer = joblib.load(save_transformers_path / 'outliers.joblib')
            df_without_outliers = transform_data(transformer=outlier_transformer,
                                                data=X)                
            # load the preprocessor
            preprocessor = joblib.load(save_transformers_path / 'preprocessor.joblib')
            # transform the data
            X_trans = transform_data(transformer=preprocessor,
                                    data=X)
            # load the output transformer
            output_transformer = joblib.load(save_transformers_path / 'output_transformer.joblib') 
            # transform the target
            y_trans = transform_data(transformer=output_transformer,
                                    data=y.values.reshape(-1,1))
            # save the transformed output to the df
            X_trans['trip_duration'] = y_trans
            
            # save the transformed data
            save_dataframe(dataframe=X_trans,
                        save_path=save_data_path / filename)
            
        elif filename == 'test.csv':
            df = read_dataframe(complete_input_path)
            # load the transfomer
            outlier_transformer = joblib.load(save_transformers_path / 'outliers.joblib')
            df_without_outliers = transform_data(transformer=outlier_transformer,
                                                data=df)                
            # load the preprocessor
            preprocessor = joblib.load(save_transformers_path / 'preprocessor.joblib')
            # transform the data
            X_trans = transform_data(transformer=preprocessor,
                                    data=df)
            # save the transformed data
            save_dataframe(dataframe=X_trans,
                        save_path=save_data_path / filename)





if __name__ == '__main__':
    main()



