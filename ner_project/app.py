from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from src.ner_model import NERModel
import logging
import uvicorn

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI and NER model
app = FastAPI(
    title="Named Entity Recognition API",
    description="API for extracting named entities from text"
)

# Initialize NER model 
ner_model = NERModel()

class TextInput(BaseModel):
    text: str

@app.post("/predict")
async def predict_entities(input_data: TextInput):
    """
    Enhanced prediction with comprehensive error handling
    """
    try:
        # Log input text
        logger.info(f"Received text for prediction: {input_data.text}")
        
        # Predict entities
        entities = ner_model.predict_entities(input_data.text)
        
        # Log prediction results
        logger.info(f"Prediction completed. Entities found: {len(entities)}")
        
        return {"entities": entities}
    except Exception as e:
        # Comprehensive error logging
        logger.error(f"Prediction error: {e}")
        logger.error(traceback.format_exc())
        raise HTTPException(
            status_code=500, 
            detail=f"Prediction failed: {str(e)}"
        )

# Health check endpoint
@app.get("/health")
async def health_check():
    return {"status": "healthy"}

# For local testing
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)