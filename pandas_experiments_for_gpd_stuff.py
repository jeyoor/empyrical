#!/bin/python
import sys
import numpy as np
import pandas as pd
import scipy.optimize as op

# Positive and negative returns with max drawdown
mixed_returns = pd.Series(
    np.array([np.nan, 1., 10., -4., 2., 3., 2., 1., -10.]) / 100,
    index=pd.date_range('2000-1-30', periods=9, freq='D'))

def gpd_var_estimator_aligned(returns):
    DEFAULT_THRESHOLD = 0.2
    returns_array = pd.Series(returns).as_matrix()
    flipped_returns = -1*returns_array
    threshold = DEFAULT_THRESHOLD
    while threshold > 0.001: #TODO: put other flags here
        #TODO: implement stuff here, call gpd maxer, etc
        threshold = threshold / 2

def gpd_loglikelihood_minimizer_aligned(price_data):
    DEFAULT_SCALE_PARAM = 1
    DEFAULT_SHAPE_PARAM = 1
    gpd_loglikelihood_lambda = gpd_loglikelihood_factory(price_data)
    optimization_results = op.minimize(test_both, [DEFAULT_SCALE_PARAM, DEFAULT_SHAPE_PARAM], method='Nelder-Mead')
    if optimization_results.success:
        #TODO: handle success here
        resulting_params = optimization_results.x
        if len(resulting_params) == 2:
            scale_param = resulting_params[0]
            shape_param = resulting_params[1]
    else:
        #TODO: handle failure here


def gpd_loglikelihood_factory(price_data):
    """
    Return a method to be maximized by scipy over the given dataset
    Requires an array of data to use for the loglikehood calculation

    Parameters
    ----------
    price_data : pd.Series or np.ndarray
    """
    return lambda params: gpd_loglikelihood(params, price_data)

def gpd_loglikelihood(params, price_data):
    """
    Helper simplify the loglikelihood factory
    """
    if (params[1] != 0):
        return -gpd_loglikelihood_scale_and_shape(params[0], params[1], price_data)
    else:
        return -gpd_loglikelihood_scale_only(params[0], price_data)



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
    return lambda params: -gpd_loglikelihood_scale_and_shape( params[0], params[1], price_data) 

def gpd_loglikelihood_scale_and_shape(scale, shape, price_data):
    """
    Helper for performing GPD calculations with scale and shape
    n : int
        number of data items
    scale : float
        scale parameter to be used in the GPD calculation
    price_data : ndarray
        array of price data
    """
    n = len(price_data)
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
    #the negative is added because scipy only minimizes and we want to maximize
    return lambda scale: -gpd_loglikelihood_scale_only(scale, price_data) 

def gpd_loglikelihood_scale_only(scale, price_data):
    """
    Helper for performing GPD calculations with only scale
    n : int
        number of data items
    scale : float
        scale parameter to be used in the GPD calculation
    data_sum : float
        sum of data items
    """
    n = len(price_data)
    data_sum = price_data.sum()
    result = -1 * sys.float_info.max
    if (scale >= 0):
        result = ((-n*np.log(scale)) - (data_sum/scale))
    return result

test_data_array = [0.03, 0.04, 0.05, 0.07, 0.2, 0.21, 0.22, 0.3,0.35]
test_scale_only = gpd_loglikelihood_scale_only_factory(np.array(test_data_array))
test_scale_and_shape = gpd_loglikelihood_scale_and_shape_factory(np.array(test_data_array))
test_both = gpd_loglikelihood_factory(np.array(test_data_array))

#TODO: learn how to avoid precision loss?
optimized_scale_only = op.minimize(test_scale_only, 1, method='Nelder-Mead')
optimized_scale_and_shape = op.minimize(test_scale_and_shape, [1, 1], method='Nelder-Mead')
optimized_both = op.minimize(test_both, [1, 1], method='Nelder-Mead')

print(optimized_scale_only)
print(optimized_scale_and_shape)
print(optimized_both)
