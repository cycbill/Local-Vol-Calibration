import numpy as np 
from scipy.optimize import minimize, LinearConstraint
import matplotlib.pyplot as plt

from parameters import *


class parent():
    def __init__(self, _first_name, _country):
        self.first_name = _first_name
        self.country = _country
    def print_info(self):
        print(self.first_name, self.country)

class child(parent):
    def __init__(self, _first_name, _country, _age):
        self.first_name = _first_name
        self.country = _country
        self.age = _age


if __name__ == "__main__":
    c = child('Cui', 'China', 5)
    c.print_info()