import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from payoff import PayOffCall, PayOffPut


class VanillaOption():
    def __init__(self, _strike, _r, _T, _sigma, _pay_off):
        self.strike = _strike
        self.r = _r
        self.T = _T
        self.sigma = _sigma
        self.payoff = _pay_off
    