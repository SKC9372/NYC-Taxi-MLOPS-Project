import sys
import logging 
from yaml import safe_load
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split
from src.logger import CustomLogger,create_log_path


log_file_path = create_log_path('make_dataset')

dataset_logger = CustomLogger(logger_name='make_dataset',log_filename=log_file_path)

dataset_logger.set_log_level(level=logging.INFO)

def load_raw_data(input_path:Path) -> pd.DataFrame:
    raw_data = pd.read_csv(input_path)
    rows,columns = raw_data.shape

    dataset_logger.save_logs(msg=f"{input_path.stem} data read having {rows} rows and {columns} columns",
                             log_level='info')
    
    return raw_data

def train_val_split(data:pd.DataFrame,test_size:float,train_size:float,random_state:int) -> tuple[pd.DataFrame,pd.DataFrame]:

    train_data,val_data = train_test_split(data,train_size=train_size,test_size=test_size,
                                            random_state=random_state)
    
    dataset_logger.save_logs(msg=f'Data is split into train split with shape {train_data.shape} and validation split with shape {val_data.shape}',
                             log_level='info')
    dataset_logger.save_logs(msg=f"The parameter values are {test_size} for test_size and {random_state} for random_state",
                             log_level='info')
    
    return train_data,val_data


def save_data(data:pd.DataFrame,output_path:Path):
    data.to_csv(output_path,index=False)
    dataset_logger.save_logs(msg=f"{output_path.stem+output_path.suffix} data saved successfully to the output folder",
                             log_level='info')
    
def read_params(input_file):
    try:
        with open(input_file) as f:
            params_file = safe_load(f)
    except FileExistsError as e:
        dataset_logger.save_logs(msg='Parameters file not found,Switching to default values for train_test_split',
                                 log_level='error')
        
        default_dict = {'test_size':0.25,
                        'random_sate':None,
                        'train_size':0.75}
        
        test_size =default_dict['test_size']
        random_state=default_dict['random_sate']
        train_size = default_dict['train_size']

        return test_size,random_state,train_size
    
    else:
        dataset_logger.save_logs(msg=f"Parameters file loaded successfully",
                                 log_level='info')
        
        test_size = params_file['make_dataset']['test_size']
        random_state= params_file['make_dataset']['random_state']
        train_size = params_file['make_dataset']['train_size']
        return test_size, random_state, train_size
    
def main():
    
    input_file_name = sys.argv[1]

    current_path = Path(__file__)

    root_path = current_path.parent.parent.parent

    interim_data_path = root_path/'data'/'interim'

    interim_data_path.mkdir(exist_ok=True)

    raw_df_path = root_path/'data'/'raw'/'extracted'/input_file_name

    raw_df = load_raw_data(input_path=raw_df_path)

    test_size,random_sate,train_size = read_params('params.yaml')

    train_df,val_df = train_val_split(data=raw_df,test_size=test_size,random_state=random_sate,train_size=train_size)

    save_data(data=train_df,output_path=interim_data_path/'train.csv')

    save_data(data=val_df,output_path=interim_data_path/'val.csv')


if __name__ == '__main__':
    main()

