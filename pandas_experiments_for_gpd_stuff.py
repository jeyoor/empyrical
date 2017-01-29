#!/bin/python
import sys
import numpy as np
import scipy.optimize as op

#TODO: write wrapper methods
#Wrapper1 -> take ONLY NEGATIVE PRICE DATA.
#Wrapper2 -> Invert the sign of the negative price data
#Wrapper3 -> minimize both functions with piecewise hey yaz.

def create_gpd_loglikelihood_lambda(price_data):
    """
    Return a method to be maximized by scipy over the given dataset
    Requires an array of data to use for the loglikehood calculation

    Parameters
    ----------
    price_data : pd.Series or np.ndarray
    """
    #TODO: finish implementation here
    return lambda scale, shape : np.piecewise

def gpd_loglikelihood_scale_and_shape_factory(price_data):
    """
    Return a method that calculates the loglikelihood of the GPD given the data set, a scale parameter (\\sigma), and a shape parameter (\\xi)

    NOTE: This method is only meant to be used when the shape parameter (\\xi) is nonzero

    Parameters
    ----------
    scale : float
        This is the scale parameter to be used in the GPD calculation
    shape : float
        This is the shape parameter to be used in the GPD calculation
    price_data : pd.Series or np.ndarray
        This data is used for maximizing the loglikelihood
    """
    n = len(price_data)
    #minimize a function of two variables requires a list of params
    #we are expecting the lambda below to be called as follows:
    #parameters = [scale, shape]
    #the final outer negative is added because scipy only minimizes, not maximizes
    #TODO: simplify this lambda with helper methods
    return lambda params: -gpd_loglikelihood_scale_and_shape(n, params[0], params[1], price_data) 

def gpd_loglikelihood_scale_and_shape(n, scale, shape, price_data):
    """
    Helper for performing GPD calculations with only scale
    n : int
        number of data items
    scale : float
        scale parameter to be used in the GPD calculation
    price_data : ndarray
        array of price data
    """
    result = -1 * sys.float_info.max
    if (scale != 0):
        param_factor = shape / scale
        if (shape != 0 and param_factor >= 0 and scale >= 0):
            result = ((-n * np.log(scale))-(((1 / shape) + 1) * (np.log((shape / scale * price_data) + 1)).sum()))
    return result

def gpd_loglikelihood_scale_only_factory(price_data):
    """
    Return a method that calculates the loglikelihood of the GPD given the data set and a scale parameter (\\sigma)

    NOTE: This method is only meant to be used when the shape parameter (\\xi) is zero

    Parameters
    ----------
    scale : float
        This is the scale parameter to be used in the GPD calculation
    price_data : pd.Series or np.ndarray
        This data is used for maximizing the loglikelihood
    """
    n = len(price_data)
    data_sum = price_data.sum()
    #the negative is added because scipy only minimizes and we want to maximize
    return lambda scale: -gpd_loglikelihood_scale_only(n, scale, data_sum) 

def gpd_loglikelihood_scale_only(n, scale, data_sum):
    """
    Helper for performing GPD calculations with only scale
    n : int
        number of data items
    scale : float
        scale parameter to be used in the GPD calculation
    data_sum : float
        sum of data items
    """
    result = -1 * sys.float_info.max
    if (scale >= 0):
        result = ((-n*np.log(scale)) - (data_sum/scale))
    return result

test_data_array = [0.03, 0.04, 0.05, 0.07, 0.2, 0.21, 0.22, 0.3,0.35]
test_scale_only = gpd_loglikelihood_scale_only_factory(np.array(test_data_array))
test_scale_and_shape = gpd_loglikelihood_scale_and_shape_factory(np.array(test_data_array))

#TODO: learn how to avoid precision loss?
optimized_scale_only = op.minimize(test_scale_only, 1)
optimized_scale_and_shape = op.minimize(test_scale_and_shape, [1, 1])

print(optimized_scale_only)
print(optimized_scale_and_shape)
