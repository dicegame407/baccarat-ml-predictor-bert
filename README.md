# Baccarat ML Predictor - Python Backend

Fast API backend service that implements the actual N-HiTS neural network model for baccarat prediction.

## Features

- Real N-HiTS neural network implementation
- Continuous learning from outcomes
- Brain file persistence
- Expert game theory analysis
- Uncertainty quantification

## Setup

```bash
pip install -r requirements.txt
```

## Run

```bash
uvicorn main:app --host 0.0.0.0 --port 8000
```

## Deploy

### Railway
```bash
railway up
```

### Render
Create a new Web Service and connect this repository.

## Endpoints

- `POST /predict` - Generate prediction
- `POST /record-outcome` - Record actual outcome and learn
- `GET /statistics` - Get statistics and brain state
- `POST /initialize` - Initialize brain system
