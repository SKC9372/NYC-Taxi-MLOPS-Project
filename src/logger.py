import logging
from pathlib import Path
import datetime as dt 


def create_log_path(module_name:str) -> str:
    current_date = dt.date.today()

    # create log file in root directory

    root_path = Path(__file__).parent.parent

    log_dir_path = root_path/'logs'

    log_dir_path.mkdir(exist_ok=True)

    module_log_path = log_dir_path/module_name

    module_log_path.mkdir(exist_ok=True,parents=True)

    current_date_str = current_date.strftime("%d-%m-%Y")

    log_file_name = module_log_path/(current_date_str + '.log')

    return log_file_name

class CustomLogger:
    def __init__(self,logger_name,log_filename):
        
        self.__logger = logging.getLogger(name=logger_name)

        self.__log_path = log_filename

        file_hander = logging.FileHandler(filename=self.__log_path,mode='a')

        self.__logger.addHandler(hdlr=file_hander)

        log_format = "%(asctime)s - %(levelname)s - %(message)s"

        time_format = '%d-%m-%Y %H:%M:%S'

        formatter = logging.Formatter(fmt=log_format,datefmt=time_format)

        file_hander.setFormatter(fmt=formatter)

    def get_log_path(self):
        return self.__log_path
    
    def get_logger(self):
        return self.__logger
    
    def set_log_level(self,level=logging.DEBUG):
        logger = self.get_logger()
        logger.setLevel(level=level)

    def save_logs(self,msg,log_level='info'):
        logger = self.get_logger()
        
        if log_level=='debug':
            logger.debug(msg=msg)

        elif log_level=='info':
            logger.info(msg=msg)

        elif log_level=='warning':
            logger.warning(msg=msg)

        elif log_level=='error':
            logger.error(msg=msg)

        elif log_level=='exception':
            logger.exception(msg=msg)

        elif log_level=='critical':
            logger.critical(msg=msg)


if __name__ == '__main__':
     
     logger = CustomLogger(logger_name='my_logger',
                           log_filename=create_log_path('test'))
     
     logger.set_log_level()

     logger.save_logs('save me code is breaking',log_level='critical')