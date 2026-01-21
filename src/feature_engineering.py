import os
import pandas as pd
import logging
from sklearn.feature_extraction.text import TfidfVectorizer
import yaml

log_dir='logs'
os.makedirs(log_dir,exist_ok=True)
logger=logging.getLogger('feature_engineering')
logger.setLevel('DEBUG')
console_handler=logging.StreamHandler()
console_handler.setLevel('DEBUG')
logger.addHandler(console_handler)
file_handler_path=os.path.join(log_dir,'feature_engineering.log')
file_h=logging.FileHandler(file_handler_path)
file_h.setLevel('DEBUG')
logger.addHandler(file_h)
formatter=logging.Formatter('%(asctime)s-%(name)s-%(levelname)s-%(message)s')
console_handler.setFormatter(formatter)
file_h.setFormatter(formatter)

def load_params(params_path: str) -> dict:
    """Load YAML parameters file."""
    try:
        with open(params_path,'r')as file:
            params=yaml.safe_load(file)
            logger.debug('Parameters retrived from %s',params_path)
            return params
    except FileNotFoundError as e:
        logger.error('Parameters file not found: %s',params_path)
        raise
    except yaml.YAMLError as e:
        logger.error('Error parsing YAML file: %s',params_path)
        raise
    except Exception as e:
        logger.error('Unexpected error loading parameters: %s',str(e))
        raise
def  load_data(file_path:str)->pd.DataFrame:
    df=pd.read_csv(file_path)
    df.fillna('',inplace=True)
    logger.debug('Data loaded from %s',file_path)
    return df
def apply_tfidf(train_data:pd.DataFrame,test_data:pd.DataFrame,max_features:int)->tuple:
    vectorizor=TfidfVectorizer(max_features=max_features)
    X_train=train_data['message'].values
    X_test=test_data['message'].values
    y_train=train_data['label'].values
    y_test=test_data['label'].values

    X_train_blow=vectorizor.fit_transform(X_train)
    X_test_blow=vectorizor.transform(X_test)
    train_df=pd.DataFrame(X_train_blow.toarray())
    test_df=pd.DataFrame(X_test_blow.toarray())
    logger.debug('TF-IDF applied with max_features=%d',max_features)
    return train_df,test_df
def save(df:pd.DataFrame,file_path:str)->None:
    os.makedirs(os.path.dirname(file_path),exist_ok=True)
    df.to_csv(file_path,index=False)
    logger.debug('Data saved to %s',file_path)
def main():
    # params=load_params(params_path='params.yaml')
    # max_features=params['feature_engineering']['max_features']
    max_features=600
    train_data=load_data('./data/processed/train_processed.csv' )
    test_data=load_data('./data/processed/test_processed.csv')
    X_train,X_test=apply_tfidf(train_data,test_data,max_features)
    save(X_train,os.path.join('./data/feature_engineered/X_train.csv'))
    save(X_test,os.path.join('./data/feature_engineered/X_test.csv'))
if __name__=='__main__':
    main()
