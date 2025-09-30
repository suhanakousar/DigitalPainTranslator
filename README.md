# Digital Pain Translator

[![Research Prototype](https://img.shields.io/badge/status-research%20prototype-orange)](https://github.com/yourusername/digital-pain-translator)
[![License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/)
[![React](https://img.shields.io/badge/react-18+-blue.svg)](https://reactjs.org/)

A research prototype system for automated pain assessment using facial analysis and caregiver inputs. Combines local browser processing with optional server-assisted ML inference.

## ⚠️ Important Notice

**This is a research prototype only. Not for clinical diagnosis or medical decision-making without proper validation and regulatory approval.**

## Table of Contents

- [Features](#features)
- [Quick Start](#quick-start)
- [Installation](#installation)
- [Usage](#usage)
- [API](#api)
- [Model Training](#model-training)
- [Testing](#testing)
- [Contributing](#contributing)
- [License](#license)

## Features

- **Dual Processing Modes**: Local browser-based or server-assisted ML inference
- **Facial Analysis**: Real-time micro-expression detection using MediaPipe
- **Caregiver Integration**: Combined assessment with professional observations
- **Privacy-First**: Explicit consent required, local processing by default
- **Explainable AI**: Feature attribution and confidence scoring
- **Real-time Communication**: WebSocket support for live inference
- **Research Tools**: Model training and evaluation utilities

## Quick Start

### Local Mode (Browser Only)

```bash
cd client
npm install
npm run dev
```

Open http://localhost:5000 and grant camera permissions.

### Full Stack (with ML Backend)

```bash
# Start backend
cd backend
docker-compose up --build

# Start frontend (in new terminal)
cd client
npm run dev
```

## Installation

### Prerequisites

- Node.js 18+
- Python 3.8+ (for backend)
- Docker (optional, for easy backend setup)

### Frontend Setup

```bash
cd client
npm install
npm run dev
```

### Backend Setup

#### Using Docker (Recommended)

```bash
cd backend
docker-compose up --build
```

#### Manual Setup

```bash
cd backend
pip install -r requirements.txt
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

## Usage

1. **Grant Camera Access**: Allow browser camera permissions for facial analysis
2. **Caregiver Input**: Enter observations (grimace, breathing, restlessness, gestures)
3. **Assessment**: View pain score, confidence, and contributing factors
4. **History**: Save and review assessment records locally

### Modes

- **Local Mode**: All processing in browser, no data transmission
- **Server Mode**: Enhanced ML inference (requires consent)

## API

### REST Endpoints

- `POST /api/infer` - Single pain assessment
- `POST /api/infer/batch` - Batch processing
- `GET /api/health` - Service health check
- `GET/POST /api/records` - Assessment records

### WebSocket

- `WS /ws/infer` - Real-time inference

### Request Format

```json
{
  "caregiverInputs": {
    "grimace": 3,
    "breathing": 2,
    "restlessness": 1,
    "gestures": ["clench"]
  },
  "landmarks": [
    [{"x": 0.5, "y": 0.5, "z": 0.0}, ...]
  ]
}
```

## Model Training

### Prepare Data

Place training data in `backend/demo_data/training_data.json`

### Train Model

```bash
cd backend
python train.py --data-path demo_data/training_data.json
```

### Evaluate Model

```bash
python eval.py --model-path models/pain_model.pt --data-path demo_data/test_data.json
```

## Testing

### Backend Tests

```bash
cd backend
pytest tests/ -v
```

### Frontend Tests

```bash
cd client
npm test
```

## Project Structure

```
digital-pain-translator/
├── client/                 # React frontend
│   ├── src/
│   │   ├── components/     # UI components
│   │   ├── services/       # API clients
│   │   └── utils/          # Helper functions
│   └── ...
├── backend/                # Python FastAPI backend
│   ├── app/
│   │   ├── routers/        # API endpoints
│   │   ├── models/         # PyTorch models
│   │   ├── inference/      # Prediction logic
│   │   └── schemas.py      # Data models
│   ├── tests/              # Unit tests
│   └── ...
├── server/                 # Express server
├── shared/                 # Common schemas
└── README.md
```

## Contributing

1. Fork the repository
2. Create a feature branch
3. Add tests for new functionality
4. Ensure all tests pass
5. Submit a pull request

### Data Contribution

- Follow ethical guidelines for data collection
- Ensure participant consent and IRB approval
- De-identify all personal information

## Ethical Guidelines

- **Research Use Only**: Not for clinical decision-making
- **Bias Awareness**: Monitor for demographic bias
- **Transparency**: Document limitations and methodology
- **Privacy**: Minimal data collection with explicit consent

## License

MIT License - see [LICENSE](LICENSE) file for details.

## Acknowledgments

Built for research in automated pain assessment. Use responsibly with appropriate ethical oversight.
