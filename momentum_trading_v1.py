import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from matplotlib import pyplot as plt
import math
import statistics
import time
from tqdm import tqdm

data = pd.read_csv("C:/Users/Advik/Documents/PythonProjects/data/BTC_min_jan-sep2020.csv", usecols=["close"])

def reg_angle(model):
    return np.arctan(model.coef_[0]) * 180 / math.pi

def reg_slope(model):
    return model.coef_[0]

def reg_model(a, b):
    x = np.array(list(range(a, b))).reshape((-1, 1))
    y = data.iloc[a:b].close.to_numpy(dtype = 'float32')
    return LinearRegression().fit(x, y)

def show_plot(x, y, model):
    plt.scatter(x, y,color='b')
    plt.plot(x, model.predict(x))
    plt.show()

def downtrend20 (a, b):
    return reg_angle(reg_model(a, b)) < -20

def uptrend30 (a, b):
    return reg_angle(reg_model(a, b)) > 30

class Position():
    def __init__(self, buy, val, takeprofit, stoploss):
        self.buy = buy
        self.val = val
        self.takeprofit = takeprofit
        self.stoploss = stoploss
    def __str__(self):
        return ("buy: " + str(self.buy)
                + "\nval: " + str(self.val)
                + "\ntake profit: " + str(self.takeprofit)
                + "\nstoploss: " + str(self.stoploss)) 

account = 10000.0
tradesize = 2000.0
positions = []
last = 0
buys = 0
gains = 0
losses = 0
breakevens = 0
sells = 0
trades = []
end = data.shape[0]
closes = []
margin = 0.001

def orders():
    global account
    global tradesize
    global positions
    global last
    global buys
    global gains
    global losses
    global breakevens
    global sells
    global trades
    global closes
    global margin
    for i in tqdm (range(100, end), desc="Progress"):
        currPrice = data.iloc[i].close
        if (i>last+100 and len(positions)<round(10000/tradesize) and reg_slope(reg_model(i-5, i)) > 5 and reg_angle(reg_model(i-50, i-5)) < -2):
            account-= tradesize*(1+margin)
            positions.append(Position(1.0*currPrice, tradesize, 1.05*currPrice, 0.90*currPrice))
            last = i
            buys+=1
        for pos in positions:
            if currPrice > pos.takeprofit:
                sell = round((pos.val * pos.takeprofit / pos.buy)*(1.0-margin))
                account += sell
                trades.append("buy: "+str(pos.buy)+"   sell: "+str(currPrice)+"   profit: "+str(round(sell-pos.val)))
                positions.remove(pos)
                gains += 1
            elif currPrice > ((pos.buy + pos.takeprofit)/2):
                pos.stoploss = pos.buy
                # profit = 0.3 * pos.val * pos.takeprofit / pos.buy
                # account += profit
                # trades.append("buy: "+str(pos.buy)+"   sell: "+str(currPrice)+"   profit: "+str(profit-1000))
            elif currPrice < pos.stoploss:
                sell = round((pos.val * pos.stoploss / pos.buy)*(1.0-margin), 2)
                account += sell
                trades.append("buy: "+str(pos.buy)+"   sell: "+str(currPrice)+"   loss: "+str(round(sell-pos.val)))
                if pos.stoploss < pos.buy:
                    losses += 1
                else:
                    breakevens += 1
                positions.remove(pos)
    for pos in positions:
        close = pos.val*(data.iloc[end-1].close/pos.buy)
        account += close
        closes.append(close-pos.val)

print("starting....")
orders()
for trade in trades:
    print(trade)
for close in closes:
    print(close)
print("account value: "+str(account))
print("gains: "+str(gains))
print("losses: "+str(losses))
print("breakevens: "+str(breakevens))
print("last: "+str(last))