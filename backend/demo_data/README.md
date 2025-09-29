# Training Data Format

This directory is for user-supplied training data. **No example files are included** - you must provide your own data according to the format specifications below.

## Data Format Requirements

### JSON Structure

Training data must be provided as a JSON file with the following structure:

```json
{
  "features": [
    [feature_vector_1],
    [feature_vector_2],
    ...
  ],
  "labels": [
    pain_score_1,
    pain_score_2,
    ...
  ]
}
```

### Feature Vector Format

Each feature vector must contain exactly 8 values in this order:

1. **mouthOpen** (float, 0.0-1.0): Mouth openness measurement
2. **eyeClosureAvg** (float, 0.0-1.0): Average eye closure measurement  
3. **browFurrowAvg** (float, 0.0-1.0): Average eyebrow furrow intensity
4. **headTiltVar** (float, 0.0-1.0): Head tilt variation measurement
5. **microMovementVar** (float, 0.0-1.0): Micro-movement variation
6. **grimace** (float, 0.0-1.0): Caregiver grimace assessment (normalized from 0-5 scale)
7. **breathing** (float, 0.0-1.0): Caregiver breathing assessment (normalized from 0-5 scale) 
8. **restlessness** (float, 0.0-1.0): Caregiver restlessness assessment (normalized from 0-5 scale)

### Label Format

Pain scores must be floating point values between 0.0 and 10.0, representing the pain assessment on a standard 0-10 pain scale.

## Data Collection Guidelines

### Facial Feature Extraction

Facial features should be extracted from video sequences using facial landmark detection (e.g., MediaPipe Face Mesh). The extraction should:

- Normalize all measurements to [0,1] range
- Calculate temporal variations for movement-based features
- Ensure consistent landmark detection quality across samples

### Caregiver Assessments

Caregiver inputs should be collected using standardized assessment protocols:

- **Grimace**: Observable facial expressions of discomfort (0-5 scale)
- **Breathing**: Breathing pattern changes indicating distress (0-5 scale)  
- **Restlessness**: Physical restlessness or agitation (0-5 scale)

Convert to 0.0-1.0 range by dividing by 5.0.

### Pain Score Labels

Ground truth pain scores should be obtained through:

- Self-report when possible (for patients who can communicate)
- Clinical assessment by trained healthcare professionals
- Validated pain assessment tools appropriate for the patient population
- Multiple assessor consensus for objective scoring

## Training Data Requirements

### Minimum Dataset Size

- **Minimum**: 1,000 samples for baseline training
- **Recommended**: 10,000+ samples for robust ML model training
- **Optimal**: 50,000+ samples for production-quality models

### Data Distribution

Ensure balanced representation across:

- Pain score ranges (0-10 scale)
- Patient demographics (age, condition)
- Assessment conditions (lighting, camera angle)
- Caregiver assessors (to reduce bias)

### Data Quality

- Remove samples with poor facial landmark detection
- Exclude corrupted or incomplete assessments
- Validate pain score consistency across assessors
- Check for temporal consistency in sequential assessments

## File Placement

Place your training data files in this directory:

```
demo_data/
├── training_data.json          # Primary training dataset
├── validation_data.json        # Optional separate validation set
├── test_data.json             # Optional separate test set  
└── README.md                  # This file
```

## Training Commands

Once you have prepared your data files, use the training script:

```bash
# Basic training
python train.py --data-path demo_data/training_data.json

# Advanced training with custom parameters
python train.py \
  --data-path demo_data/training_data.json \
  --model-type lstm \
  --hidden-size 128 \
  --epochs 200 \
  --batch-size 64 \
  --learning-rate 0.001
```

## Data Privacy and Ethics

### Privacy Requirements

- Ensure all data collection follows institutional review board (IRB) approval
- Obtain informed consent from all participants
- De-identify all data before training
- Implement secure data handling protocols

### Ethical Considerations

- This system is for research purposes only - not for clinical diagnosis
- Validate model performance across diverse populations
- Monitor for algorithmic bias in pain assessment
- Ensure transparency in model decision-making

### Data Retention

- Follow institutional data retention policies
- Implement secure data destruction procedures
- Document data lineage and provenance
- Maintain audit trails for compliance

## Troubleshooting

### Common Issues

1. **JSON Format Errors**: Validate JSON structure using online validators
2. **Feature Dimension Mismatch**: Ensure exactly 8 features per sample
3. **Label Range Errors**: Verify pain scores are 0.0-10.0
4. **Missing Values**: Replace or interpolate missing data points

### Validation Commands

```bash
# Validate data format
python -c "
import json
with open('demo_data/training_data.json') as f:
    data = json.load(f)
    print(f'Samples: {len(data[\"features\"])}')
    print(f'Feature dim: {len(data[\"features\"][0])}')
    print(f'Label range: {min(data[\"labels\"])}-{max(data[\"labels\"])}')
"
```

## Support

For questions about data format or training procedures:

1. Check the main README.md for general guidance
2. Review the training script help: `python train.py --help`
3. Examine the model configuration in `app/config.py`

Remember: This is a research prototype and should not be used for clinical decision-making without proper validation and regulatory approval.