import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
    
class PayOffCall():
    def __init__(self, _strike):
        self.strike = _strike
    def compute(self, spot):
        return np.maximum(spot - self.strike, 0)
    
class PayOffPut():
    def __init__(self, _strike):
        self.strike = _strike
    def compute(self, spot):
        return np.maximum(self.strike - spot, 0)
    
if __name__ == "__main__":
    pay_off_call = PayOffCall(1.5)
    spot = np.linspace(0, 2.0, num = 50, endpoint=True)
    print(spot)
    
    price = pay_off_call.compute(spot)
    
    print(price)
    plt.plot(spot, price)
    plt.show()