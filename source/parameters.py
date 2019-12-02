import numpy as np 
from scipy.interpolate import interp1d, interp2d
from scipy.integrate import quad, dblquad
from scipy.interpolate import CubicSpline

class ConstantParameter1D():
    def __init__(self, _constant):
        self.constant = _constant

    def interpolate(self, x):
        x_size = len(x)
        result = np.repeat(self.constant, x_size)
        return result

    def integral(self, x0, x1):
        return self.constant * (x1 - x0)


class ConstantParameter2D():
    def __init__(self, _constant):
        self.constant = _constant

    def interpolate(self, x, y):
        return self.constant

    def integral(self, x0, x1, y0, y1):
        return self.constant * (x1 - x0) * (y1 - y0)


class PiecewiseLinearParameter1D():
    def __init__(self, _x_inputs, _value_inputs):
        self.x_inputs = _x_inputs
        self.value_inputs = _value_inputs
        self.interp_function = interp1d(self.x_inputs, self.value_inputs, fill_value=(self.value_inputs[0], self.value_inputs[-1]), bounds_error=False)

    def interpolate(self, x):
        result = self.interp_function(x)
        return result

    def integral(self, x0, x1):
        result, _ = quad(self.interp_function, x0, x1)
        return result


class CubicSplineParameter1D():
    def __init__(self, _x_inputs, _value_inputs):
        self.x_inputs = _x_inputs
        self.value_inputs = _value_inputs
        self.interp_function = interp1d(self.x_inputs, self.value_inputs, fill_value=(self.value_inputs[0], self.value_inputs[-1]), kind='cubic', bounds_error=False)

    def interpolate(self, x):
        return self.interp_function(x)

    def integral(self, x0, x1):
        result, _ = quad(self.interp_function, x0, x1)
        return result


class CubicSplineLinearExtrpParameter1D():
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
    
    def interp_func(self, x):
        if (self.xn - x) * (self.x1 - x) <= 0:
            result = self.cubic_spline_interpolate(x)
        elif (self.xn - self.x1) * (self.x1 - x) > 0:
            result = self.left_extrapolate(x)
        else:
            result = self.right_extrapolate(x)
        return result

    def interpolate(self, xs):
        if (xs.size) == 1:
            result = self.interp_func(xs)
        else:
            result = np.zeros_like(xs)
            for i, x in enumerate(xs):
                result[i] = self.interp_func(x)
        return result

    def integral(self, x0, x1):
        result, _ = quad(self.interpolate, x0, x1)
        return result


class PiecewiseLinearParameter2D():
    def __init__(self, _x_inputs, _y_inputs, _value_inputs):
        self.x_inputs = _x_inputs
        self.y_inputs = _y_inputs
        self.value_inputs = _value_inputs
        self.interp_function = interp2d(self.x_inputs, self.y_inputs, self.value_inputs)

    def interpolate(self, t, x):
        return self.interp_function(t, x)
    
    def integral(self, t0, t1, x0, x1):
        result, _ = dblquad(self.interp_function, t0, t1, lambda t: x0, lambda t: x1)
        return result
    

if __name__ == '__main__':
    time_inputs = np.arange(0, 10)
    value_inputs = np.arange(0, 10)
    print(time_inputs, value_inputs)
    para = PiecewiseLinearParameter1D(time_inputs, value_inputs)
    print("interpolate: ", para.interpolate(3.5))
    print("integral: ", para.integral(1, 3.5))

    scale_inputs = np.arange(0, 10)
    value2_inputs = np.arange(0, 100)
    para2 = PiecewiseLinearParameter2D(time_inputs, scale_inputs, value2_inputs)
    print("interpolate: ", para2.interpolate(1.5, 1.5))
    print("integral: ", para2.integral(0, 1.5, 0, 1.5))