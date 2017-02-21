#!/bin/python
import sys
import math
import numpy as np
import pandas as pd
import scipy.optimize as op

# Simple benchmark, no drawdown
simple_benchmark = pd.Series(
    np.array([0., 1., 0., 1., 0., 1., 0., 1., 0.]) / 100,
    index=pd.date_range('2000-1-30', periods=9, freq='D'))

# All positive returns, small variance
positive_returns = pd.Series(
    np.array([1., 2., 1., 1., 1., 1., 1., 1., 1.]) / 100,
    index=pd.date_range('2000-1-30', periods=9, freq='D'))

# All negative returns
negative_returns = pd.Series(
    np.array([0., -6., -7., -1., -9., -2., -6., -8., -5.]) / 100,
    index=pd.date_range('2000-1-30', periods=9, freq='D'))

# Positive and negative returns with max drawdown
mixed_returns = pd.Series(
    np.array([np.nan, 1., 10., -4., 2., 3., 2., 1., -10.]) / 100,
    index=pd.date_range('2000-1-30', periods=9, freq='D'))

# Flat line
flat_line_1 = pd.Series(
    np.array([1., 1., 1., 1., 1., 1., 1., 1., 1.]) / 100,
    index=pd.date_range('2000-1-30', periods=9, freq='D'))

# Weekly returns
weekly_returns = pd.Series(
    np.array([0., 1., 10., -4., 2., 3., 2., 1., -10.])/100,
    index=pd.date_range('2000-1-30', periods=9, freq='W'))

# Monthly returns
monthly_returns = pd.Series(
    np.array([0., 1., 10., -4., 2., 3., 2., 1., -10.])/100,
    index=pd.date_range('2000-1-30', periods=9, freq='M'))

# Series of length 1
one_return = pd.Series(
    np.array([1.])/100,
    index=pd.date_range('2000-1-30', periods=1, freq='D'))

# Empty series
empty_returns = pd.Series(
    np.array([])/100,
    index=pd.date_range('2000-1-30', periods=0, freq='D'))


def gpd_var_estimator_aligned(returns):
    DEFAULT_THRESHOLD = 0.2
    MINIMUM_THRESHOLD = 0.000000001
    VAR_P = 0.01
    returns_array = pd.Series(returns).as_matrix()
    flipped_returns = -1*returns_array
    filtered_returns = flipped_returns[flipped_returns>0]
    threshold = DEFAULT_THRESHOLD
    finished = False
    scale_param = 0
    shape_param = 0
    result = [0, 0, 0, 0]
    while not finished and threshold > MINIMUM_THRESHOLD:
        iteration_returns = filtered_returns[filtered_returns>=threshold]
        param_result = gpd_loglikelihood_minimizer_aligned(iteration_returns)
        if (param_result[0] != False and param_result[1] != False):
            scale_param = param_result[0]
            shape_param = param_result[1]
            #non-negative shape parameter is required for fat tails
            if (shape_param > 0):
                finished = True
        threshold = threshold / 2
        #DEBUG
        #print('threshold:{} iteration_returns:{} param_result:{} scale_param:{} shape_param:{} finished:{}'.format(threshold, iteration_returns, param_result, scale_param, shape_param, finished))
    if (finished):
        var_estimate = gpd_var_calculator(threshold, scale_param, shape_param, VAR_P, len(returns_array), len(iteration_returns)) 
        es_estimate = gpd_es_calculator(var_estimate, threshold, scale_param, shape_param)
        result = [threshold, scale_param, shape_param, var_estimate, es_estimate]
    return result

def gpd_es_calculator(var_estimate, threshold, scale_param, shape_param):
    result = 0
    if ((1 - shape_param) != 0):
        result = (var_estimate/(1-shape_param))+((scale_param-(shape_param*threshold))/(1-shape_param))
    return result

def gpd_var_calculator(threshold, scale_param, shape_param, probability, total_n, exceedance_n):
    result = 0
    if (exceedance_n > 0 and shape_param > 0):
        result = threshold+((scale_param/shape_param)*(math.pow((total_n/exceedance_n)*probability, -shape_param)-1))
    return result

def gpd_loglikelihood_minimizer_aligned(price_data):
    result = [False, False]
    DEFAULT_SCALE_PARAM = 1
    DEFAULT_SHAPE_PARAM = 1
    if (len(price_data) > 0):
        gpd_loglikelihood_lambda = gpd_loglikelihood_factory(price_data)
        optimization_results = op.minimize(gpd_loglikelihood_lambda, [DEFAULT_SCALE_PARAM, DEFAULT_SHAPE_PARAM], method='Nelder-Mead')
        if optimization_results.success:
            #TODO: handle success here
            resulting_params = optimization_results.x
            if len(resulting_params) == 2:
                result[0] = resulting_params[0]
                result[1] = resulting_params[1]
    return result

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

print('Simple:',gpd_var_estimator_aligned(simple_benchmark))
print('Positive:',gpd_var_estimator_aligned(positive_returns))
print('Negative:',gpd_var_estimator_aligned(negative_returns))
print('Mixed:',gpd_var_estimator_aligned(mixed_returns))
print('Flat Line 1:',gpd_var_estimator_aligned(flat_line_1))
print('Weekly:',gpd_var_estimator_aligned(weekly_returns))
print('Monthly:',gpd_var_estimator_aligned(monthly_returns))
print('One return:',gpd_var_estimator_aligned(one_return))
print('Empty returns:',gpd_var_estimator_aligned(empty_returns))



#>>> print('Simple:',gpd_var_estimator_aligned(simple_benchmark))
#Simple: [0, 0, 0, 0]
#>>> print('Positive:',gpd_var_estimator_aligned(positive_returns))
#Positive: [0, 0, 0, 0]
#>>> print('Negative:',gpd_var_estimator_aligned(negative_returns))
#Negative: [0.025, 0.068353586736348199, 9.4304947982121171e-07, 0.31206547376799765, 0.38041939568242211]
#>>> print('Mixed:',gpd_var_estimator_aligned(mixed_returns))
#Mixed: [0.05, 0.10001255835838491, 1.5657360018514067e-06, 0.29082525469237713, 0.39083834671363232]
#>>> print('Flat Line 1:',gpd_var_estimator_aligned(flat_line_1))
#Flat Line 1: [0, 0, 0, 0]
#>>> print('Weekly:',gpd_var_estimator_aligned(weekly_returns))
#Weekly: [0.05, 0.10001255835838491, 1.5657360018514067e-06, 0.29082525469237713, 0.39083834671363232]
#>>> print('Monthly:',gpd_var_estimator_aligned(monthly_returns))
#Monthly: [0.05, 0.10001255835838491, 1.5657360018514067e-06, 0.29082525469237713, 0.39083834671363232]
#>>> print('One return:',gpd_var_estimator_aligned(one_return))
#One return: [0, 0, 0, 0]
#>>> print('Empty returns:',gpd_var_estimator_aligned(empty_returns))
#Empty returns: [0, 0, 0, 0]
