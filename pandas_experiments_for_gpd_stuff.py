#!/bin/python
import numpy as np
import scipy.optimize as op


def gpd_loglikelihood_scale_and_shape(scale, shape, price_data):
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
    #the final outer negative is added because scipy only minimizes, not maximizes
    #TODO: simplify this lambda with helper methods
    return lambda scale, shape: - (-(n * np.log(scale))-(((1 / shape) + 1) * (np.log((shape / scale * price_data) + 1)).sum()))


def gpd_loglikelihood_scale_only(scale, price_data):
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
    #the final outer negative is added because scipy only minimizes, not maximizes
    return lambda scale: - (- (n * np.log(scale)) - (data_sum / scale))

def create_gpd_loglikelihood_lambda(price_data):
    """
    Return a method to be maximized by scipy over the given dataset
    Requires an array of data to use for the loglikehood calculation

    Parameters
    ----------
    price_data : pd.Series or np.ndarray
    """
    return lambda scale, shape : np.piecewise

test_scale_only = gpd_loglikelihood_scale_only(3, np.array([-0.05,-0.2,-0.22,-0.3,-0.35]))
test_scale_and_shape = gpd_loglikelihood_scale_and_shape(2, 3, np.array([-0.05,-0.2,-0.22,-0.3,-0.35]))

#TODO: learn how to avoid precision loss?
optimized_scale_only = op.minimize(test_scale_only, 10)
#TODO: learn how to minimize two-valued function!!!
optmized_scale_and_shape = op.minimize(test_scale_and_shape, [10, 5])
