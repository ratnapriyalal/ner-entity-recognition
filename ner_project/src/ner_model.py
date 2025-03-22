import torch
from transformers import AutoTokenizer, AutoModelForTokenClassification
from typing import List, Dict
import logging
import numpy as np

class NERModel:
    def __init__(self, model_name: str = 'Jean-Baptiste/roberta-large-ner-english'):
        """
        Initialize NER model with comprehensive logging
        
        Args:
            model_name (str): Transformer model name
        """
        # Configure more detailed logging
        logging.basicConfig(
            level=logging.DEBUG,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger(__name__)
        
        try:
            # Load tokenizer and model with verbose logging
            self.logger.info(f"Attempting to load model: {model_name}")
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.model = AutoModelForTokenClassification.from_pretrained(model_name)
            
            # Move model to appropriate device
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            self.logger.info(f"Using device: {self.device}")
            self.model.to(self.device)
            
            # Detailed model configuration logging
            self.logger.info("Model Configuration:")
            self.logger.info(f"Number of labels: {self.model.config.num_labels}")
            self.logger.info(f"Label2ID: {self.model.config.label2id}")
            self.logger.info(f"ID2Label: {self.model.config.id2label}")
            
        except Exception as e:
            self.logger.error(f"Comprehensive model loading failed: {e}")
            raise

    def predict_entities(self, text: str) -> List[Dict]:
        """
        Enhanced entity prediction with comprehensive logging
        
        Args:
            text (str): Input text
        
        Returns:
            List of entity predictions
        """
        try:
            self.logger.debug(f"Processing text: {text}")
            
            # Tokenization with detailed logging
            inputs = self.tokenizer(
                text, 
                return_tensors="pt", 
                padding=True, 
                truncation=True, 
                max_length=512
            ).to(self.device)
            
            self.logger.debug(f"Input IDs shape: {inputs['input_ids'].shape}")
            self.logger.debug(f"Attention mask shape: {inputs['attention_mask'].shape}")
            
            # Predict entities with comprehensive logging
            with torch.no_grad():
                outputs = self.model(**inputs)
                
            self.logger.debug(f"Logits shape: {outputs.logits.shape}")
            
            # Process predictions
            predictions = self._process_predictions(inputs, outputs)
            
            self.logger.info(f"Extracted {len(predictions)} entities")
            return predictions
        
        except Exception as e:
            self.logger.error(f"Comprehensive entity prediction failed: {str(e)}")
            # Log full traceback
            import traceback
            self.logger.error(traceback.format_exc())
            return []

    def _process_predictions(self, inputs, outputs):
        """
        Enhanced prediction processing with detailed logging
        
        Args:
            inputs: Tokenizer inputs
            outputs: Model outputs
        
        Returns:
            Processed entity predictions
        """
        try:
            # Get token-level predictions with logging
            logits = outputs.logits
            predictions = torch.argmax(logits, dim=2)
            
            # Convert to numpy for easier processing
            predictions = predictions.cpu().numpy()
            input_ids = inputs['input_ids'].cpu().numpy()
            
            # Collect entities with comprehensive tracking
            entities = []
            for sent_idx, (sent_pred, sent_ids) in enumerate(zip(predictions, input_ids)):
                self.logger.debug(f"Processing sentence {sent_idx}")
                
                for token_idx, (token_pred, token_id) in enumerate(zip(sent_pred, sent_ids)):
                    # Skip special tokens with logging
                    if token_id in [self.tokenizer.cls_token_id, 
                                    self.tokenizer.sep_token_id, 
                                    self.tokenizer.pad_token_id]:
                        self.logger.debug(f"Skipping special token: {token_id}")
                        continue
                    
                    # Get token and its predicted entity
                    token = self.tokenizer.decode([token_id])
                    label = self.model.config.id2label.get(token_pred, 'O')
                    
                    # Detailed entity extraction logging
                    if label != 'O':
                        entity_info = {
                            'token': token,
                            'entity_type': label,
                            'token_id': int(token_id),
                            'prediction_id': int(token_pred)
                        }
                        entities.append(entity_info)
                        self.logger.debug(f"Found entity: {entity_info}")
            
            return entities
        
        except Exception as e:
            self.logger.error(f"Comprehensive prediction processing failed: {str(e)}")
            import traceback
            self.logger.error(traceback.format_exc())
            return []