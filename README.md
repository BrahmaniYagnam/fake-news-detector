# Fake News Detection using BERT Transformer with FastAPI

A production-ready end-to-end NLP application to classify news articles as **Fake** or **Real** using a fine-tuned BERT model and a FastAPI backend.

## Project Overview

This project includes data processing, model training, explainability, and a REST API. The backend exposes prediction, health, and model information endpoints, with automatic Swagger documentation.

## Features

- Fake news classification using `bert-base-uncased`
- FastAPI backend with structured logging and environment configuration
- Explainable predictions with token importance
- Model export and inference without retraining
- Docker support and deployment instructions
- Unit tests for core functionality

## Folder Structure

- `backend/` - FastAPI application code
- `dataset/` - dataset files and preprocessing helpers
- `notebooks/` - exploratory analysis and modeling notebooks
- `training/` - training scripts and model utilities
- `saved_model/` - exported model, tokenizer, and metadata
- `tests/` - unit tests
- `docs/` - documentation and deployment guide

## Getting Started

1. Clone the repository.
2. Create a Python virtual environment:
   ```bash
   python -m venv .venv
   source .venv/Scripts/activate
   ```
3. Install backend dependencies:
   ```bash
   pip install -r requirements.txt
   ```
4. Install frontend dependencies:
   ```bash
   cd frontend
   npm install
   ```
5. Start the API:
   ```bash
   uvicorn backend.main:app --reload
   ```
6. Start the frontend app:
   ```bash
   cd frontend
     ```
7. Open the app at `http://127.0.0.1:5173`.

## API Endpoints

- `GET /` - application info
- `GET /health` - health status
- `GET /model-info` - model metadata
- `POST /predict` - predict fake or real news

## Deployment

Use Docker:

```bash
docker compose up --build
```

## License

MIT License.
npm run dev
 