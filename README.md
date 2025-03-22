# Named Entity Recognition (NER) API

## Project Overview
A robust Named Entity Recognition (NER) service built using FastAPI and transformer-based models, capable of extracting named entities from text inputs.

## Features
- Transformer-based NER model : `Jean-Baptiste/roberta-large-ner-english`
- FastAPI Backend
- Comprehensive entity recognition
- Multiple model support
- Error handling and logging

## Prerequisites
- Anaconda/Miniconda
- Python 3.9+

## Installation & Setup

### 1. Clone the Repository
```bash
git clone https://github.com/ratnapriyalal/ner-entity-recognition.git
cd ner-entity-recognition
```

### 2. Create conda environment
```bash
# Create new conda environment
conda create -n ner_project python=3.9 -y

# Activate the environment
conda activate ner_project
```

### 3. Install dependencies
```bash
# Install dependencies
conda install -c conda-forge numpy=1.23.5 pytorch=1.13.1 transformers=4.30.2 -y

# Install additional requirements
pip install fastapi uvicorn spacy
```

### 4. Download SpaCy model
```bash
python -m spacy download en_core_web_sm
```

### 5. Running the Application
```bash
# Run FastAPI application in local development server
uvicorn app:app --reload
```

### 6. Run the test script
```bash
# Open a new terminal (keep the FastAPI server running) and ensure you're in the same environment
conda activate ner_project

# Run the test script
python test_api.py
```

#### API Endpoints:
- Predict: Endpoint - `/predict`, Method - POST
- Health Check: Endpoint - `\health`, Method - GET
