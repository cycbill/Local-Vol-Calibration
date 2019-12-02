import numpy as np 
from scipy.interpolate import interp1d, interp2d
from scipy.integrate import quad, dblquad
from scipy.interpolate import CubicSpline
import matplotlib.pyplot as plt

from read_write_excel import read_from_excel

class CubicSpline_LinearExtrp():
    def __init__(self, _x_inputs, _value_inputs):
        self.x_inputs = _x_inputs
        self.x1 = self.x_inputs[0]
        self.xn = self.x_inputs[-1]
        self.value_inputs = _value_inputs
        self.y1 = self.value_inputs[0]
        self.yn = self.value_inputs[-1]

        self.cubic_spline_interpolate = CubicSpline(self.x_inputs, self.value_inputs)
        self.left_firstderiv = self.cubic_spline_interpolate(self.x1, 1)
        self.right_firstderiv = self.cubic_spline_interpolate(self.xn, 1)
        self.left_extrapolate = lambda x: self.left_firstderiv * (x - self.x1) + self.y1
        self.right_extrapolate = lambda x: self.right_firstderiv * (x - self.xn)  + self.yn
    
    def interpolate(self, x):
        result = np.zeros_like(x)
        for i, xi in enumerate(x):
            if (self.xn - xi) * (self.x1 - xi) <= 0:
                result[i] = self.cubic_spline_interpolate(xi)
            elif (self.xn - self.x1) * (self.x1 - xi) > 0:
                result[i] = self.left_extrapolate(xi)
            else:
                result[i] = self.right_extrapolate(xi)
        return result

    def integral(self, x0, x1):
        result, _ = quad(self.interpolate, x0, x1)
        return result

if __name__ == '__main__':
    k_prev, price_prev = read_from_excel()

    J = 98
    x_min = -0.09391736906882849
    x_max = 0.10109773583359967
    x_values = np.linspace(x_min, x_max, J, endpoint=True)

    cubic_spline = CubicSpline_LinearExtrp(k_prev, price_prev)
    result = cubic_spline.interpolate(x_values)

    plt.plot(k_prev, price_prev, 'r.', x_values, result, 'b-')
    plt.show()
