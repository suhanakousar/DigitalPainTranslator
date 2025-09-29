# Digital Pain Translator

A research prototype system for automated pain assessment using facial analysis and caregiver inputs. This enhanced version includes both local-only processing and optional server-assisted inference with machine learning capabilities.

## ⚠️ Important Notice

**This system is a research prototype and should not be used for clinical diagnosis or medical decision-making without proper validation and regulatory approval.**

## Overview

The Digital Pain Translator analyzes facial micro-expressions and caregiver assessments to estimate pain levels on a 0-10 scale. The system operates in two modes:

- **Local-only mode**: All processing happens in the browser using deterministic algorithms
- **Server-assisted mode**: Enhanced ML inference using a Python FastAPI backend

## Project Structure

```
digital-pain-translator/
├── client/                     # React frontend application
│   ├── src/
│   │   ├── components/         # UI components including ServerToggle
│   │   ├── services/           # API and WebSocket clients
│   │   └── ...
│   └── ...
├── backend/                    # Python FastAPI backend
│   ├── app/
│   │   ├── main.py            # FastAPI application
│   │   ├── routers/           # REST and WebSocket endpoints
│   │   ├── models/            # PyTorch model implementations
│   │   ├── inference/         # Prediction and feature extraction
│   │   └── schemas.py         # API request/response schemas
│   ├── tests/                 # Unit tests with synthetic data
│   ├── demo_data/             # Placeholder for user training data
│   ├── train.py               # Model training CLI
│   ├── eval.py                # Model evaluation and export CLI
│   ├── Dockerfile
│   └── docker-compose.yml
└── README.md                  # This file
```

## Quick Start

### Frontend Only (Local Mode)

1. **Install dependencies:**
   ```bash
   cd client
   npm install
   ```

2. **Start development server:**
   ```bash
   npm run dev
   ```

3. **Access the application:**
   - Open http://localhost:5000
   - Grant camera permissions when prompted
   - Use local-only processing (default mode)

### Full Stack Development

1. **Start backend services:**
   ```bash
   cd backend
   docker-compose up --build
   ```

2. **Start frontend:**
   ```bash
   cd client
   npm run dev
   ```

3. **Enable server-assisted mode:**
   - Toggle "Server-assisted Mode" in the application
   - Grant explicit consent for server processing
   - Backend provides enhanced ML inference

## Backend Setup

### Manual Installation

1. **Install Python dependencies:**
   ```bash
   cd backend
   pip install -r requirements.txt
   ```

2. **Run development server:**
   ```bash
   uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
   ```

### Docker Installation

```bash
cd backend
docker-compose up --build
```

The backend will be available at http://localhost:8000 with API documentation at http://localhost:8000/docs.

## API Documentation

### REST Endpoints

- **POST /api/infer** - Perform pain assessment inference
- **POST /api/infer/batch** - Batch inference for multiple requests
- **GET /api/infer/status** - Get inference service status
- **GET/POST /api/records** - Manage assessment records
- **GET /api/health** - Service health check
- **POST /api/model/reload** - Reload trained models

### WebSocket Endpoint

- **WS /ws/infer** - Real-time inference with low latency

### API Request Format

All inference requests must include:

```json
{
  "caregiverInputs": {
    "grimace": 0-5,
    "breathing": 0-5, 
    "restlessness": 0-5,
    "gestures": ["clench", "point", "shake"]
  }
}
```

Plus either:

```json
{
  "features": {
    "mouthOpen": 0.0-1.0,
    "eyeClosureAvg": 0.0-1.0,
    "browFurrowAvg": 0.0-1.0,
    "headTiltVar": 0.0-1.0,
    "microMovementVar": 0.0-1.0
  }
}
```

Or:

```json
{
  "landmarks": [
    [{"x": float, "y": float, "z": float}, ...]
  ]
}
```

## Training Custom Models

### Data Preparation

1. **Prepare training data** according to format in `backend/demo_data/README.md`

2. **Place data files:**
   ```bash
   backend/demo_data/training_data.json
   ```

3. **Validate data format:**
   ```bash
   python -c "
   import json
   with open('backend/demo_data/training_data.json') as f:
       data = json.load(f)
       print(f'Samples: {len(data[\"features\"])}')
       print(f'Features per sample: {len(data[\"features\"][0])}')
   "
   ```

### Training Commands

```bash
cd backend

# Basic training
python train.py --data-path demo_data/training_data.json

# Advanced training with custom parameters
python train.py \
  --data-path demo_data/training_data.json \
  --model-type lstm \
  --hidden-size 128 \
  --epochs 200 \
  --batch-size 64 \
  --learning-rate 0.001 \
  --output-dir models/custom
```

### Model Evaluation and Export

```bash
# Evaluate trained model
python eval.py \
  --model-path models/pain_assessment_model_lstm.pt \
  --data-path demo_data/test_data.json

# Export to ONNX for deployment
python eval.py \
  --model-path models/pain_assessment_model_lstm.pt \
  --export-onnx \
  --export-torchscript \
  --output-dir exports/
```

## Testing

### Backend Tests

```bash
cd backend
pytest tests/ -v
```

Tests use programmatically generated synthetic data and cover:
- Feature extraction from facial landmarks
- Baseline and ML model predictions
- API endpoint validation and responses
- WebSocket communication

### Frontend Tests

```bash
cd client
npm test
```

## Privacy and Security

### Local-Only Mode (Default)

- All processing happens in the browser
- No data transmitted to external servers
- Camera access used only for local analysis
- Assessment results stored locally only

### Server-Assisted Mode (Opt-in)

- Requires explicit user consent
- Facial features (not raw video) sent to backend
- Real-time WebSocket communication available
- Optional assessment history storage
- User can revoke consent at any time

### Security Recommendations

- **Development**: HTTP acceptable for localhost testing
- **Production**: Require HTTPS/WSS for data transmission
- **Data Storage**: Encrypt sensitive assessment records
- **Access Control**: Implement authentication for production deployments

## Model Information

### Baseline Model

- Deterministic weighted linear combination
- Explicit feature contributions for explainability
- No training data required
- Consistent, interpretable results

### Machine Learning Models

- **LSTM**: Sequence model for temporal facial features
- **1D CNN**: Convolutional model for pattern detection
- **Training**: Requires user-supplied datasets
- **Explainability**: Gradient-based feature attribution

## Deployment

### Development Deployment

```bash
docker-compose up --build
```

### Production Considerations

- Configure TLS certificates for HTTPS
- Set up proper database for record storage
- Implement authentication and authorization
- Monitor model performance and drift
- Regular security audits and updates

## Contributing

### Data Contribution

To contribute training data:

1. Follow format specifications in `backend/demo_data/README.md`
2. Ensure IRB approval and participant consent
3. De-identify all data before submission
4. Document data collection methodology

### Code Contribution

1. Follow existing code style and patterns
2. Add unit tests for new functionality
3. Update documentation for API changes
4. Test both local and server-assisted modes

## Ethical Guidelines

### Research Use Only

- This system is for research purposes only
- Not approved for clinical diagnosis or treatment decisions
- Requires validation before any healthcare application
- Users must understand limitations and appropriate use

### Bias and Fairness

- Train models on diverse populations
- Monitor for demographic bias in predictions
- Validate across different healthcare settings
- Document limitations and failure cases

### Transparency

- Provide explainable predictions when possible
- Document model training data and methodology
- Make limitations clear to end users
- Enable audit trails for assessment decisions

## Support and Troubleshooting

### Common Issues

1. **Camera not working**: Check browser permissions and HTTPS requirement
2. **Server connection failed**: Verify backend is running and accessible
3. **Model loading errors**: Check model file paths and formats
4. **Training data errors**: Validate JSON format and feature dimensions

### Getting Help

1. Check API documentation at `/docs` endpoint
2. Review configuration in `backend/app/config.py`
3. Examine training logs for model issues
4. Test with provided synthetic data first

## License

This project is provided as a research prototype. Check specific license terms for your use case.

## Acknowledgments

This system is built for research in automated pain assessment and should be used responsibly with appropriate ethical oversight and validation.