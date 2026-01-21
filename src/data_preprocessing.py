import pandas as pd
import os
import logging
from sklearn.preprocessing import LabelEncoder
from nltk.stem import PorterStemmer
from nltk.corpus import stopwords
import nltk
import string

# Download required NLTK resources
nltk.download('stopwords')
nltk.download('punkt')

# ================= LOGGER SETUP =================
log_dir = "logs"
os.makedirs(log_dir, exist_ok=True)

logger = logging.getLogger('data_preprocessing')
logger.setLevel(logging.DEBUG)

# Prevent duplicate handlers
if not logger.handlers:
    console_handler = logging.StreamHandler()
    file_handler = logging.FileHandler(os.path.join(log_dir, 'data_preprocessing.log'))

    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    console_handler.setFormatter(formatter)
    file_handler.setFormatter(formatter)

    logger.addHandler(console_handler)
    logger.addHandler(file_handler)

# ================= TEXT TRANSFORMATION =================
stop_words = set(stopwords.words('english'))
ps = PorterStemmer()

def transform_data(message):
    """Clean and stem a single message."""
    try:
        message = message.lower()
        words = nltk.word_tokenize(message)
        words = [w for w in words if w.isalnum()]
        words = [w for w in words if w not in stop_words and w not in string.punctuation]
        words = [ps.stem(w) for w in words]
        return ' '.join(words)
    except Exception as e:
        logger.error("Error transforming message: %s", e)
        raise

# ================= DATA PROCESSING =================
def process_data(df, text_column='message', target_column='label'):
    """Transform text data and encode labels."""
    try:
        logger.debug("Starting data processing")

        le = LabelEncoder()
        df[target_column] = le.fit_transform(df[target_column])

        logger.debug("Removing duplicate rows")
        df.drop_duplicates(inplace=True)

        logger.debug("Applying text transformation")
        df[text_column] = df[text_column].apply(transform_data)

        logger.debug("Data processing completed")
        return df

    except Exception as e:
        logger.error("Error processing data: %s", e)
        raise

# ================= MAIN FUNCTION =================
def main():
    text_column = 'message'
    target_column = 'label'

    train_data = pd.read_csv('data/raw/train.csv')
    test_data = pd.read_csv('data/raw/test.csv')
    logger.debug("Data loaded successfully")

    train_processed = process_data(train_data, text_column, target_column)
    test_processed = process_data(test_data, text_column, target_column)

    data_path = os.path.join('data', 'processed')
    os.makedirs(data_path, exist_ok=True)

    train_processed.to_csv(os.path.join(data_path, 'train_processed.csv'), index=False)
    test_processed.to_csv(os.path.join(data_path, 'test_processed.csv'), index=False)

    logger.debug("Processed data saved successfully")

# ================= RUN =================
if __name__ == "__main__":
    main()
