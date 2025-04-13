import logging
from pathlib import Path
from zipfile import ZipFile
from src.logger import CustomLogger,create_log_path



log_file_path = create_log_path('extract_dataset')

extract_logger = CustomLogger(logger_name='extract_dataset',log_filename=log_file_path)

extract_logger.set_log_level(level=logging.INFO)

def extract_zip_file(input_path:Path,output_path:Path):
    with ZipFile(file=input_path) as f:
        f.extractall(path=output_path)
        input_file_name = input_path.stem + input_path.suffix
        extract_logger.save_logs(msg=f"{input_file_name} extracted successuflly at the target path",
                                 log_level='info')
        
def main():

    current_path = Path(__file__)

    root_path = current_path.parent.parent.parent

    raw_data_path = root_path/'data'/'raw'

    output_path = raw_data_path/'extracted'

    output_path.mkdir(parents=True,exist_ok=True)

    input_path = raw_data_path/'zipped'

    extract_zip_file(input_path=input_path/'train.zip',output_path=output_path)

    extract_zip_file(input_path=input_path/'test.zip',output_path=output_path)


if __name__=="__main__":
    main()