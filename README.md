# Stock Market Trend Prediction (Next 5-Candles Trend)

This project predicts the trend of the next 5 candles (Uptrend, Downtrend, Neutral) in stock market data using an LSTM (Long Short-Term Memory) model.

---

## Table of Contents

1. [Overview](#overview)
2. [Dependencies](#dependencies)
3. [Dataset](#dataset)
4. [How to Run](#how-to-run)
5. [Assumptions](#assumptions)
6. [Non-Standard Techniques](#non-standard-techniques)
7. [Insights and Challenges](#insights-and-challenges)
8. [Suggestions for Improvement](#suggestions-for-improvement)

---

## Overview

The model takes historical stock data (OHLC, volume, and technical indicators) and predicts the trend for the next 5 candles:

* **0**: Uptrend
* **1**: Downtrend
* **2**: Neutral

The project uses an LSTM model to capture sequential dependencies in time-series data.

---

## Dependencies

Make sure you have the following Python packages installed:

```bash
pip install pandas numpy scikit-learn matplotlib seaborn tensorflow
```

Optional for Google Colab:

```python
from google.colab import drive
```

---

## Dataset

* CSV file containing stock market data with columns like `Open`, `High`, `Low`, `Close`, `Volume`, and various technical indicators (SMA, EMA, RSI, MACD, Bollinger Bands, Alligator indicators, ATR, WMA).
* Ensure the CSV file is accessible from your system or Google Drive.

---

## How to Run

1. **Mount Google Drive** (if using Colab):

```python
from google.colab import drive
drive.mount("/content/drive")
```

2. **Load Dataset**:

```python
import pandas as pd
df = pd.read_csv('/content/drive/MyDrive/Assignment/large_32 (2).csv')
```

3. **Preprocess Dataset**:

   * Convert `Date Time` to datetime object
   * Sort data chronologically
   * Fill missing values (`ffill`)

4. **Generate Labels** using the next 5-candle trend.

5. **Select Features and Scale** using `StandardScaler`.

6. **Create Sequences** for LSTM with a default `time_steps = 20`.

7. **Train-Test Split** (80%-20%).

8. **Build and Train LSTM Model**:

```python
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
```

* Use `EarlyStopping` to avoid overfitting.

9. **Evaluate Model**:

   * Classification report
   * Confusion matrix
   * Plot training history (loss and accuracy)

10. **Predict New Data**:

* Use the last `time_steps` of data to predict the trend for the next 5 candles.

---

## Assumptions

* Labels are generated based on **all 5 future candles being bullish or bearish**.
* If fewer than 5 candles are available at the end, the trend is assumed **Neutral**.
* Dataset is assumed **chronologically ordered**.
* The model assumes **numeric features only**; non-numeric columns are excluded.

---

## Non-Standard Techniques

* **Custom Label Generation** for next 5-candle trends.
* **Sequential LSTM input**: sequences of 20 time-steps for better trend prediction.
* **EarlyStopping** callback to restore best model weights automatically.
* No data augmentation is applied (can be considered in future work).

---

## Insights and Challenges

* The dataset is **highly imbalanced**, with most sequences labeled as Neutral.
* The model struggles to predict Uptrend or Downtrend due to this imbalance.
* Sequential LSTM captures temporal dependencies better than standard classifiers.

---

## Suggestions for Improvement

1. **Data Balancing**:

   * Oversample minority classes (Uptrend, Downtrend) using SMOTE or custom techniques.

2. **Feature Engineering**:

   * Add more derived technical indicators or volatility features.

3. **Model Enhancement**:

   * Experiment with Bidirectional LSTM or attention mechanisms.
   * Hyperparameter tuning for LSTM layers, dropout, and batch size.

4. **Prediction Horizon**:

   * Try predicting for fewer or more than 5 candles to see the effect on accuracy.

---

**Author:** Prasad Kabade
**Date:** 2025-10-16
