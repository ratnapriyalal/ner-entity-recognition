import pandas as pd
import spacy
import logging
from typing import List, Dict

class DataPreprocessor:
    def __init__(self, dataset_path: str):
        """
        Initialize data preprocessor
        
        Args:
            dataset_path (str): Path to NER dataset
        """
        self.logger = logging.getLogger(__name__)
        logging.basicConfig(level=logging.INFO)
        
        try:
            self.dataset = pd.read_csv(dataset_path)
            self.logger.info("Dataset successfully loaded")
        except Exception as e:
            self.logger.error(f"Dataset loading failed: {e}")
            raise

    def preprocess_text(self, text: str) -> str:
        """
        Advanced text preprocessing
        
        Args:
            text (str): Input text
        
        Returns:
            Cleaned text
        """
        # Load SpaCy model
        nlp = spacy.load('en_core_web_sm')
        
        # Preprocessing steps
        doc = nlp(text.lower())
        
        # Remove stopwords and punctuation
        cleaned_tokens = [
            token.lemma_ for token in doc 
            if not token.is_stop and not token.is_punct
        ]
        
        return ' '.join(cleaned_tokens)

    def prepare_dataset(self) -> pd.DataFrame:
        """
        Prepare dataset for model training
        
        Returns:
            Preprocessed DataFrame
        """
        try:
            # Apply preprocessing to text column
            self.dataset['cleaned_text'] = self.dataset['text'].apply(self.preprocess_text)
            
            self.logger.info("Dataset preprocessing completed")
            return self.dataset
        
        except Exception as e:
            self.logger.error(f"Dataset preparation failed: {e}")
            raise