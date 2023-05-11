import math
import yfinance as yf
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, TensorDataset
import tqdm
from datetime import datetime
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
import argparse
import sys
from models.lstm import LSTM, train_lstm, test_lstm

# filter common dates of two index
def get_common_dates(stock1, stock2):
    return sorted(list(set(stock1.index.tolist()).intersection(set(stock2.index.tolist()))))

def fit_linear_regression(price1, price2):
    logprice1, logprice2 = np.log(price1), np.log(price2)
    lin_reg = LinearRegression(fit_intercept=True)
    lin_reg.fit(logprice1.reshape(-1, 1), logprice2)
    beta, alpha = lin_reg.coef_[0], lin_reg.intercept_
    spread = logprice2 - beta * logprice1 - alpha
    print(f"Linear regression: log(stock2) = {beta} * log(stock1) + {alpha}")
    return beta, alpha, spread


def train_test_split(dates, data, test_ratio=0.2, window_size=30):
    assert len(dates) == len(data)
    test_size = int(len(dates) * test_ratio)
    train_size = len(dates) - test_size
    X_train, y_train, X_test, y_test = [], [], [], []
    for i in range(window_size, train_size):
        X_train.append(data[i - window_size:i])
        y_train.append(data[i])
    for i in range(train_size, train_size + test_size):
        X_test.append(data[i - window_size:i])
        y_test.append(data[i])
    print(f"Finishing splitting, training size: {len(X_train)}, testing size: {len(X_test)}")
    return X_train, y_train, X_test, y_test, dates[:train_size], dates[train_size:]

def to_dataloader(X_train, y_train, X_test, y_test, batch_size):
    train_set = TensorDataset(torch.Tensor(X_train), torch.Tensor(y_train))
    train_loader = DataLoader(train_set, batch_size=batch_size)
    test_set = TensorDataset(torch.Tensor(X_test), torch.Tensor(y_test))
    test_loader = DataLoader(test_set, batch_size=batch_size)
    return train_loader, test_loader

def simulate_standard_model(spread_pred, beta, mu, sigma, price_open_a, price_open_b, price_close_final_a, price_close_final_b):
    # Setup at t = 0
    if spread_pred[0] < mu: # logprice2 - beta * logprice1 < alpha, so long B
        invest_a, invest_b = -1, beta
    else:
        invest_a, invest_b = 1, -beta
    hold_a, hold_b = invest_a / price_open_a[0], invest_b / price_open_a[0]
    print(f'initial hold: {hold_a, hold_b}')
    init_invest = invest_a + invest_b
    invest = init_invest
    histories = []

    num_test = len(y_test)
    for t in range(num_test):
        if spread_pred[t] < mu - 1 * sigma and invest_a == 1: # switch from long A to long B
            invest_adjust = (-1 - price_open_a[t] * hold_a) + (beta - price_open_b[t] * hold_b)
            print(f'Enter long B, short A at t={t}, invest_adjustment: {invest_adjust}')
            invest += invest_adjust
            invest_a, invest_b = -1, beta
            histories.append((t, invest_adjust, 'B'))
            hold_a, hold_b = invest_a / price_open_a[t], invest_b / price_open_b[t]
            
        elif spread_pred[t] > mu + 1 * sigma and invest_a == -1: # switch from long B to long A
            invest_adjust = (1 - price_open_a[t] * hold_a) + (-beta - price_open_b[t] * hold_b)
            print(f'Enter long A, short B at t={t}, invest_adjustment: {invest_adjust}')
            invest += invest_adjust
            invest_a, invest_b = 1, -beta
            histories.append((t, invest_adjust, 'A'))
            hold_a, hold_b = invest_a / price_open_a[t], invest_b / price_open_b[t]
    
    # clear all investment
    print(f'final hold: {hold_a, hold_b}')
    invest -= (hold_a * price_close_final_a + hold_b * price_close_final_b)
    return init_invest, invest, histories
    
    # clear all investment
    print(f'final hold: {hold_a, hold_b}')
    invest -= (hold_a * price_close_final_a + hold_b * price_close_final_b)
    return init_invest, invest, histories
        
    
    # clear all investment
    invest += hold_a * price_close_final_a + hold_b * price_close_final_b
    return init_invest, invest, histories


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--years', type=int, default=3)      # option that takes a value
    parser.add_argument('--test_ratio', type=float, default=0.2)
    parser.add_argument('--model', type=str, default="lstm")
    parser.add_argument('--batch_size', type=int, default="8")

    args = parser.parse_args()

    sd = datetime(2023-args.years, 5, 6)
    ed = datetime(2023, 5, 5)
    config = {
        'test_ratio': args.test_ratio,
        'model': args.model,
        'batch_size': args.batch_size,
    }

    stock1 = yf.download(tickers='PEP', start=sd, end=ed, interval="1d") # TODO: use real pairs found by Wenqi
    stock2 = yf.download(tickers='MCD', start=sd, end=ed, interval="1d")
    common_dates = get_common_dates(stock1, stock2)
    stock1_common, stock2_common = stock1.loc[common_dates], stock2.loc[common_dates]
    price1, price2 = np.array(stock1_common['Adj Close']), np.array(stock2_common['Adj Close'])
    beta, alpha, spread = fit_linear_regression(price1, price2)
    scaler = MinMaxScaler(feature_range=(0,1))
    scaled_spread = scaler.fit_transform(spread.reshape(-1,1))
    X_train, y_train, X_test, y_test, train_dates, test_dates = train_test_split(common_dates, scaled_spread, test_ratio=config['test_ratio'])

    if config['model'] == "lstm":
        train_loader, test_loader = to_dataloader(X_train, y_train, X_test, y_test, batch_size=config['batch_size'])
        # Define the training function
        model = LSTM(input_dim=1, hidden_dim=10, output_dim=1, num_layers=2)
        train_lstm(model, train_loader, num_epochs=100)
        predictions, labels = test_lstm(model, test_loader, scaler)
    
        mu, sigma = np.mean(labels), np.std(labels)
        num_train, num_test = len(y_train), len(y_test)
        price_open_a, price_open_b = np.array(stock1_common['Open'])[num_train:], np.array(stock2_common['Open'])[num_train:]
        init_invest, invest, histories = simulate_standard_model(predictions, beta, mu, sigma, price_open_a, price_open_b, price1[-1], price2[-1])
        print(f'Initial investment: {init_invest}, Final value: {invest}')
        print(histories)
