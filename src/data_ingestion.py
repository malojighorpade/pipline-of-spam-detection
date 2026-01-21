import pandas as pd
import os
import yaml
from sklearn.model_selection import train_test_split
import logging

log_dir="logs"
os.makedirs(log_dir,exist_ok=True)
logger=logging.getLogger('data_ingestion')
logger.setLevel('DEBUG')

console_handler=logging.StreamHandler()
console_handler.setLevel('DEBUG')

file_handeler=os.path.join(log_dir,'data_ingestion.log')
file_h=logging.FileHandler(file_handeler)
file_h.setLevel('DEBUG')

formatter=logging.Formatter('%(asctime)s-%(name)s-%(levelname)s-%(message)s')
console_handler.setFormatter(formatter)
file_h.setFormatter(formatter)

logger.addHandler(console_handler)
logger.addHandler(file_h)

# def load_params(params_path: str)->dict:
#     """Load YAML parameters file."""
#     try:
#         with open('params.path','r')as file:
#             params=yaml.safe_load(file)
#             logger.debug('Parameters retrived from %s',params_path)
#             return params
#     except FileNotFoundError as e:
#         logger.error('Parameters file not found: %s',params_path)
#         raise 
#     except yaml.YAMLError as e:
#         logger.error('Error parsing YAML file: %s',params_path)
#         raise
#     except Exception as e:
#         logger.error('Unexpected error loading parameters: %s',str(e))
#         raise

def load_data(data_path:str)->pd.DataFrame:
    """Loading data from a CSV file"""
    try:
        df=pd.read_csv(data_path)
        logger.debug('Data Loaded from %s',data_path)
        return df
    except pd.errors.ParserError as e:
        logger.error('Error parsing CSV file: %s',data_path)
        raise
    except Exception as e:
        logger.error('Unexpected error loading data: %s',str(e))
        raise

def preprocess_data(df:pd.DataFrame)->pd.DataFrame:
    """preprocessing the data"""
    try:
        df.drop(columns=['Unnamed: 2','Unnamed: 3','Unnamed: 4'],inplace=True)
        df.rename(columns={'v1':'label','v2':'message'},inplace=True)
        logger.debug('Data Preprocessing completed')
        return df
    except KeyError as e:
        logger.error('Error in preprocessing data: %s',str(e))
        raise
    except Exception as e:
        logger.error('Unexpected error in preprocessing data: %s',str(e))
        raise

def save_data(train_data:pd.DataFrame,test_data:pd.DataFrame,save_path:str)->None:
    """Saving the data into train and test CSV files"""

    try:
        raw_data_path=os.path.join(save_path,'raw')
        os.makedirs(raw_data_path,exist_ok=True)
        train_data.to_csv(os.path.join(raw_data_path,'train.csv'),index=False)
        test_data.to_csv(os.path.join(raw_data_path,'test.csv'),index=False)
        logger.debug('Data saved to %s',raw_data_path)
    except Exception as e:
        logger.error('Error saving data: %s',str(e))
        raise
def main():
    # params=load_params(params_path='params.yaml')
    # test_size=params['data_ingestion']['test_size']
    test_size  =0.2
    data_path='expremients/spam.csv'
    df=load_data(data_path=data_path)
    final_df=preprocess_data(df)
    train_data,test_data=train_test_split(final_df,test_size=test_size,random_state=42)
    save_data(train_data=train_data,test_data=test_data,save_path='./data')
if __name__=='__main__':
    main()  