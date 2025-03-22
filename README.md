# Retail Sales Forecasting Project

## Overview

This project provides a flexible sales forecasting solution for retail businesses, offering multiple methods for predicting sales across different stores and items. The primary goal is to help retailers optimize inventory management and sales strategies through data-driven insights. Training dataset and final results were uploaded to the [Kaggle Challenge](https://www.kaggle.com/competitions/ml-zoomcamp-2024-competition)

## Features

- **Multiple Forecasting Methods**
  - Average-based forecasting
  - Extensible architecture for adding new prediction methods
  - Store-level and item-level sales analysis
  
- **Logging and Monitoring**
  - Detailed logging for tracking script execution

## Prerequisites

- Python 3.8+
- Required libraries:
  - pandas
  - numpy
  - scikit-learn (for future method implementations)

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/your-username/retail-forecasting.git
   cd retail-forecasting
   ```

2. Create a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

### Running the Forecasting Script

Basic usage:
```bash
python main.py --data ./data --method avg --output ./submissions/forecast.csv
```

### Command-line Arguments

- `--data`: Path to the data directory (default: 'data')
- `--method`: Forecasting method to use (currently supports 'avg')
- `--output`: Path to save the submission file

### Supported Methods

- `avg`: Average-based sales prediction
- More methods can be added by extending the base method class

## Project Structure

```
retail-forecasting/
│
├── main.py              # Main script for running forecasts
├── method               # Method classes
├── util/
│   └── preprocess.py    # Data preprocessing utilities
├── data/                # Raw data directory
└── submissions/         # Output forecasts
```


# Kaggle Submission History

| Date       |Submission Description                  | Score      |  User     |
|------------|----------------------------------------|------------|-----------|
| 2024-12-19 |Feature engineering and tuning          | 0.13567    |           | 
| 2024-12-20 | Ensemble with XGBoost and LightGBM     | 0.14567    |           |

## License

Distributed under the MIT License. See `LICENSE` for more information.
