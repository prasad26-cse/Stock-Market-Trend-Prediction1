# Next-5-Candles Trend Prediction

This repository contains a complete end-to-end pipeline for predicting the trend of the next 5 candlesticks in stock market data using **TensorFlow/Keras**. The model uses historical OHLCV data along with technical indicators to classify short-term trends.

## Table of Contents

* [Dependencies](#dependencies)
* [Dataset](#dataset)
* [Usage](#usage)
* [Assumptions](#assumptions)
* [Techniques](#techniques)
* [Insights & Challenges](#insights--challenges)
* [Suggestions for Improvement](#suggestions-for-improvement)

---

## Dependencies

Ensure you have the following Python packages installed. Recommended Python version: **3.10+**

```bash
pip install pandas numpy scikit-learn matplotlib seaborn tensorflow
```

Optional for Jupyter Notebook:

```bash
pip install notebook jupyterlab
```

---

## Dataset

* The dataset should be a CSV file containing at least the following columns:
  `Date Time`, `Open`, `High`, `Low`, `Close`, `Volume`
* Additional columns for indicators like RSI, EMA, MACD can be included.
* **Upload the dataset** in the same directory as the notebook/script or adjust the file path accordingly.

---

## Usage

1. **Load the dataset**
   The notebook/script reads the CSV file and parses the datetime column.

2. **Preprocess data**

   * Fill missing values if any.
   * Scale features using standardization or MinMax scaling.
   * Generate labels based on the trend of the next 5 candles:

     * `Up` if the closing price of the 5th candle > current close.
     * `Down` if the closing price of the 5th candle < current close.
     * `Neutral` if change is insignificant.

3. **Split data**

   * Training (70%)
   * Validation (15%)
   * Test (15%)

4. **Model Training**

   * Define a neural network with dense layers (or LSTM/CNN for sequence modeling).
   * Compile with `categorical_crossentropy` loss and an optimizer like `Adam`.
   * Fit model with training data and validate on the validation set.

5. **Evaluation**

   * Evaluate model accuracy on the test set.
   * Generate classification reports and confusion matrix for insights.

6. **Prediction**

   * Load new OHLCV data to generate next-5-candle trend predictions.

---

## Assumptions

* Only sequences with **at least 5 future candles** are labeled. Remaining trailing rows are ignored.
* Missing or NaN values are handled with **forward fill**.
* All features are numeric; categorical features (if any) are encoded before training.
* Trend thresholds for `Neutral` classification are configurable (default ±0.2%).

---

## Techniques Used

* **Data Augmentation for Time Series**: Sliding window approach to generate sequences.
* **Custom Labeling**: Next-5-candle trend classification.
* **Standardization**: Scaling features for neural network stability.
* Optional: Use of **custom loss functions** or **sequence models** (LSTM/GRU) to improve trend prediction.

---

## Insights & Challenges

* Stock data is noisy; minor price fluctuations can make `Neutral` trends dominant.
* Training deep learning models on small datasets can overfit; adding regularization and dropout helps.
* Feature engineering (technical indicators) significantly improves performance.

---

## Suggestions for Improvement

* Explore sequence models (LSTM, GRU, Transformer) for capturing temporal dependencies.
* Apply feature selection to reduce irrelevant indicators.
* Use rolling window normalization for better handling of non-stationary data.
* Incorporate market sentiment or news data for multi-modal learning.

---

## Execution Order

1. Upload dataset to working directory.
2. Run preprocessing notebook/script.
3. Train the model.
4. Evaluate on test data.
5. Run prediction module on new data.

