# Human Activity Recognition Using Hidden Markov Models

## Project Description

This project implements a Hidden Markov Model (HMM) to classify human physical activities using smartphone sensor data. The system uses accelerometer and gyroscope measurements to identify four distinct activities: being still, jumping, standing, and walking.

## Dataset

### Data Collection
- **Total Samples:** 22,508 rows of sensor data
- **Sensors Used:** 3-axis accelerometer and 3-axis gyroscope
- **Activities:** Being Still, Jumping, Standing, Walking
- **Devices:** iPhone 12 and Google Pixel 4a
- **Sampling Rate:** Approximately 2946 Hz (device maximum)

### Data Structure
- **Training Data:** 16 feature windows from short activity clips (5-10 seconds each)
- **Test Data:** 27 feature windows from unseen test dataset
- **Features:** 36 raw features reduced to 6 principal components via PCA

## Features

### Feature Extraction
- **Window Size:** 1.0 second sliding window
- **Overlap:** 75%
- **Time-Domain Features:**
  - Mean, Standard Deviation, Variance
  - Signal Magnitude Area (SMA)
  - Magnitude Mean
  - Axis Correlation
- **Frequency-Domain Features:**
  - Dominant Frequency
  - Spectral Energy (via FFT)

### Preprocessing
1. Z-score normalization using StandardScaler
2. Principal Component Analysis (PCA) for dimensionality reduction
3. 95% variance retention with 6 components

## Model Architecture

### HMM Configuration
- **Type:** Gaussian Hidden Markov Model
- **Hidden States:** 4 (one per activity)
- **Observation Space:** 6-dimensional PCA features
- **Covariance Type:** Diagonal
- **Library:** hmmlearn

### Training
- **Algorithm:** Baum-Welch (Expectation-Maximization)
- **Iterations:** 300
- **Learned Parameters:** Emission probabilities (means and covariances)
- **Fixed Parameters:** Transition matrix and initial state probabilities

### Inference
- **Algorithm:** Viterbi decoding
- **Purpose:** Find most likely sequence of hidden states

## Results

### Model Performance
- **Overall Accuracy:** 59.3%

| Activity | Samples | Sensitivity (Recall) | Specificity |
|----------|---------|---------------------|-------------|
| Being Still | 9 | 0.667 | 0.889 |
| Jumping | 9 | 0.333 | 1.000 |
| Standing | 5 | 0.600 | 0.864 |
| Walking | 4 | 1.000 | 0.739 |

### Key Findings
- **Best Performance:** Walking (100% recall)
- **Worst Performance:** Jumping (33.3% recall)
- **Main Limitation:** Fragmented training data led to unrealistic transition probabilities

## Installation

### Requirements
```
python >= 3.7
numpy
pandas
scikit-learn
hmmlearn
matplotlib
seaborn
```

### Setup
```bash
pip install numpy pandas scikit-learn hmmlearn matplotlib seaborn
```

## Usage

### Data Structure
Ensure your data is organized as follows:
```
project/
├── traindata/
│   └── train_data.csv
├── testdata/
│   └── test_data.csv
└── hmm_model.py
```

### Running the Model
```python
# Import required libraries
from hmmlearn import hmm
import numpy as np
import pandas as pd

# Load and preprocess data
# (See implementation code for details)

# Initialize and train HMM
model = hmm.GaussianHMM(n_components=4, covariance_type="diag", n_iter=300)
model.fit(training_features)

# Decode test sequence
predicted_states = model.predict(test_features)
```

## Project Structure

```
Hidden-Markov-Models/
├── README.md
├── markov_model.ipynb                    # Main HMM implementation notebook
├── cleaning.ipynb                        # Data cleaning and preprocessing
├── new_data_test.ipynb                   # Testing on new data
├── cleaned_data.csv                      # Preprocessed sensor data
├── features.csv                          # Extracted feature vectors
├── being_still/                          # Raw data for being still activity
├── jumping/                              # Raw data for jumping activity
├── standing/                             # Raw data for standing activity
├── walking/                              # Raw data for walking activity
├── testdata/                             # Test dataset folder
├── artifacts/                            # Generated artifacts and outputs
├── transition_matrix_heatmap.png         # Visualization of transition probabilities
├── transition_flow.png                   # Activity transition flow diagram
└── viterbi_decoded_sequence.png          # Viterbi algorithm output visualization
```

## Limitations and Future Work

### Current Limitations
1. **Fragmented Training Data:** Short, non-sequential clips resulted in unrealistic transition probabilities
2. **Limited Training Samples:** Only 16 training windows
3. **Static Activity Confusion:** Being Still and Standing have overlapping features

### Proposed Improvements
1. **Continuous Data Collection:** Collect several minutes of continuously performed activities
2. **Enhanced Model Complexity:** Implement Gaussian Mixture Model HMM (GMM-HMM)
3. **Larger Dataset:** Increase training samples across all activities
4. **Cross-Device Validation:** Test generalization across different smartphone models

## Contributors

| Name | Contribution | Percentage |
|------|-------------|-----------|
| Branis Sumba | Data Collection (Walking/Standing), Report Writing (Background/Discussion), Code Review | 50% |
| Christophe Gakwaya | Data Collection (Being Still/Jumping), Feature Extraction, HMM Implementation, Report Writing (Results/HMM Setup) | 50% |

## License

This project was completed as part of an academic assignment.

## References

- Rabiner, L. R. (1989). A tutorial on hidden Markov models and selected applications in speech recognition.
- hmmlearn documentation: https://hmmlearn.readthedocs.io/
- Scikit-learn documentation: https://scikit-learn.org/

## Contact

For questions or issues, please open an issue in this repository.

---

**Last Updated:** October 2025