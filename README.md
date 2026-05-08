# AAPL Stock Price Prediction using LSTM

This project implements Deep Learning, time-series forecasting using **PyTorch**. The goal is to predict the daily closing price of Apple Inc. (AAPL) by analyzing historical patterns through a Long Short-Term Memory (LSTM) network.

## Project Highlights
*   **Data Source:** Real-time stock data via the `yfinance` API.
*   **Architecture:** A stacked LSTM model designed to capture long-term temporal dependencies.
*   **Hardware Acceleration:** Optimized for **Apple Silicon (M1/M2/M3)** using the `mps` (Metal Performance Shaders) backend.
*   **Interactive Visualization:** Comparison of actual vs. predicted prices using `Plotly`.

## The Pipeline
1.  **Preprocessing:** Data is normalized using `StandardScaler` to improve convergence during training.
2.  **Windowing:** The model uses a **30-day sliding window** (looking at the past month to predict the next day).
3.  **Modeling:** A 2-layer LSTM with 32 hidden units, followed by a fully connected layer.
4.  **Evaluation:** Performance is measured using **Root Mean Squared Error (RMSE)** in actual dollar amounts.

## Results
The final output includes an interactive chart showing the model's predictions against the actual market close, along with an error tracking subplot to identify periods of high volatility.
