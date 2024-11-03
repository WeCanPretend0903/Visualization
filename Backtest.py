import pandas as pd
import matplotlib.pyplot as plt
import glob
import os
import numpy as np


class DataLoader:
    def __init__(self, file_path):
        self.file_path = file_path

    def load_data(self):
        try:
            data = pd.read_csv(self.file_path)

            # Process columns only if they contain strings (for SPY data with $ signs)
            for col in ["Close/Last", "Open", "High", "Low"]:
                if col in data.columns and data[col].dtype == object:
                    data[col] = (
                        data[col]
                        .str.replace("$", "")
                        .str.replace(",", "")
                        .astype(float)
                    )

            # Convert 'Volume' to integer if it's not N/A
            if "Volume" in data.columns:
                data["Volume"] = (
                    pd.to_numeric(data["Volume"], errors="coerce").fillna(0).astype(int)
                )

            # Convert 'Date' to datetime and set it as the index
            if "Date" in data.columns:
                data["Date"] = pd.to_datetime(data["Date"])
                data.set_index("Date", inplace=True)

            return data
        except Exception as e:
            print(f"Error loading data: {e}")
            return None


class Strategy:
    def __init__(self):
        self.signals_df = pd.DataFrame(columns=["timestamp", "percentage change"])

    def generate_percentage_change(self, data):
        if "Close/Last" not in data.columns:
            raise ValueError("Data must contain a 'Close/Last' column.")
        self.signals_df = self.signals_df._append(
            {"timestamp": data.index[0], "percentage change": 0}, ignore_index=True
        )

        for i in range(1, len(data["Close/Last"])):
            previous_price = data["Close/Last"].iloc[i - 1]
            current_price = data["Close/Last"].iloc[i]

            percentage_change = (
                ((current_price - previous_price) / previous_price) * 100
                if previous_price != 0
                else 0
            )

            self.signals_df = self.signals_df._append(
                {"timestamp": data.index[i], "percentage change": percentage_change},
                ignore_index=True,
            )

        return self.signals_df

    def calculate_rsi(self, window=14):
        if len(self.signals_df) < window:
            raise ValueError("Not enough data points to calculate RSI.")

        # Calculate gains and losses
        self.signals_df["gain"] = self.signals_df["percentage change"].where(
            self.signals_df["percentage change"] > 0, 0
        )
        self.signals_df["loss"] = -self.signals_df["percentage change"].where(
            self.signals_df["percentage change"] < 0, 0
        )

        # Calculate rolling averages
        avg_gain = self.signals_df["gain"].rolling(window=window, min_periods=1).mean()
        avg_loss = self.signals_df["loss"].rolling(window=window, min_periods=1).mean()

        # Avoid division by zero errors by setting avg_loss to a very small value if it's zero
        avg_loss = avg_loss.replace(0, 1e-10)

        # Relative Strength (RS)
        rs = avg_gain / avg_loss

        # Relative Strength Index (RSI)
        self.signals_df["RSI"] = 100 - (100 / (1 + rs))
        return self.signals_df


class Backtest:
    def __init__(self, data, strategy, initial_balance, file_name):
        self.data = data
        self.strategy = strategy
        self.balance = initial_balance
        self.results = []
        self.position = None
        self.shares = 0
        self.net_worth_history = []
        self.buy_price = 0

    def handle_buy_order(self, rsi_value, current_price, timestamp, net_worth):
        # Buying logic
        self.position = "buy"
        self.shares = round(float(self.balance / current_price), 2)
        self.balance -= self.shares * current_price
        self.balance = round(self.balance, 2)
        self.buy_price = current_price
        self.results.append(
            (
                timestamp,
                "Buy",
                round(float(rsi_value), 2),
                float(current_price),
                float(self.shares),
                float(self.balance),
                round(float(net_worth), 2),
            )
        )

    def handle_sell_order(self, rsi_value, current_price, timestamp, net_worth):
        # Selling logic
        self.position = None
        self.balance += self.shares * current_price
        self.balance = round(self.balance, 2)
        self.shares = 0
        self.buy_price = 0
        self.results.append(
            (
                timestamp,
                "Sell",
                round(float(rsi_value), 2),
                float(current_price),
                float(self.shares),
                float(self.balance),
                round(float(net_worth), 2),
            )
        )

    def run(self):

        if len(self.strategy.signals_df) != len(self.data["Close/Last"]):
            raise ValueError("The length of signals_df and data must be the same.")

        print(
            "Timestamp, Buy/Sell, RSI Value, Current Price, Total Shares Owned, Current Balance, Net Portfolio"
        )
        x = 1
        for i in range(len(self.strategy.signals_df) - 1, -1, -1):
            rsi_value = self.strategy.signals_df["RSI"].iloc[i]
            current_price = self.data["Close/Last"].iloc[i]
            timestamp = self.strategy.signals_df["timestamp"].iloc[i]
            x = x + 1
            # Calculate the current balance based on shares held
            net_worth = self.balance + (self.shares * current_price)

            self.net_worth_history.append(net_worth)  # Store current balance

            if (
                rsi_value < 30
                and self.position is None
                and self.balance >= current_price
            ):
                self.handle_buy_order(rsi_value, current_price, timestamp, net_worth)

            elif rsi_value > 70 and self.position == "buy":
                # RSI Sell Signal
                self.handle_sell_order(rsi_value, current_price, timestamp, net_worth)

            elif current_price > 1.01 * self.buy_price and self.position == "buy":
                # take profit 2%
                self.handle_sell_order(rsi_value, current_price, timestamp, net_worth)
            elif current_price < 0.99 * self.buy_price and self.position == "buy":
                # stop loss 2%
                self.handle_sell_order(rsi_value, current_price, timestamp, net_worth)

        return self.results

    def plot_portfolio(self):
        if len(self.net_worth_history) > len(self.strategy.signals_df["timestamp"]):
            self.net_worth_history = self.net_worth_history[:-1]

        plotItem = self.net_worth_history
        plotItem.reverse()

        # Prepare lists for buy and sell points
        buy_points = []
        buy_timestamps = []
        sell_points = []
        sell_timestamps = []

        for result in self.results:
            timestamp, action, rsi_value, current_price, shares, balance, net_worth = (
                result
            )
            if action == "Buy":
                buy_points.append(net_worth)
                buy_timestamps.append(timestamp)
            elif action == "Sell":
                sell_points.append(net_worth)
                sell_timestamps.append(timestamp)

        plt.figure(figsize=(14, 8))
        plt.plot(
            self.strategy.signals_df["timestamp"],
            plotItem,
            label="Portfolio Balance",
            color="Black",
        )

        if buy_points and buy_timestamps:
            plt.scatter(
                buy_timestamps,
                buy_points,
                color="green",
                marker="^",
                s=100,
                label="Buy",
            )

        if sell_points and sell_timestamps:
            plt.scatter(
                sell_timestamps,
                sell_points,
                color="red",
                marker="v",
                s=100,
                label="Sell",
            )

        # Formatting the plot
        plt.xlabel("Date")
        plt.ylabel("Balance (USD)")
        plt.title(file_name)
        plt.xticks(rotation=45)
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        plt.show()


def plot_stock_prices(df):
    plt.figure(figsize=(14, 10))
    plt.plot(df.index, df["Close/Last"], label="Close Price", color="black")
    plt.xlabel("Date")
    plt.ylabel("Close Price (USD)")
    plt.title("Stock Prices")
    plt.xticks(rotation=45)
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()


# Main execution
file_name = "GE.csv"
data_loader = DataLoader(file_name)
data = data_loader.load_data()


if data is not None:
    strategy = Strategy()
    strategy.generate_percentage_change(data)
    strategy.calculate_rsi(window=14)

    backtest = Backtest(data, strategy, 10000, file_name)
    results = backtest.run()
    # Print results
    for result in results:
        print(result)

    backtest.plot_portfolio()

    # plot_stock_prices(data)  # Plot the stock prices
else:
    print("Data loading failed.")
