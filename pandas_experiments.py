import pandas as pd
import numpy as np

# Positive and negative returns with max drawdown
positive_returns = pd.Series(
    np.array([1., 2., 1., 1., 1., 1., 1., 1., 1.]) / 100,
    index=pd.date_range('2000-1-30', periods=9, freq='D'))

mixed_returns = pd.Series(
    np.array([np.nan, 1., 10., -4., 2., 3., 2., 1., -10.]) / 100,
    index=pd.date_range('2000-1-30', periods=9, freq='D'))

# Simple benchmark, no drawdown
simple_benchmark = pd.Series(
    np.array([0., 1., 0., 1., 0., 1., 0., 1., 0.]) / 100,
    index=pd.date_range('2000-1-30', periods=9, freq='D'))

# Flat line
flat_line_1 = pd.Series(
    np.array([1., 1., 1., 1., 1., 1., 1., 1., 1.]) / 100,
    index=pd.date_range('2000-1-30', periods=9, freq='D'))

returns_series = positive_returns
factor_returns_series = simple_benchmark
#combine returns and factor returns into pairs
pairs = pd.concat([returns_series, factor_returns_series], axis=1)
pairs.columns = ['returns', 'factor_returns']

#exclude any rows where returns are nan 
pairs = pairs.dropna()
#sort by beta
pairs = pairs.sort_values(by='factor_returns')

#find the three vectors, using median of 3
start_index = 0
mid_index = int(round(len(pairs) / 2, 0))
end_index = len(pairs) - 1

(start_returns, start_factor_returns) = pairs.iloc[start_index]
(mid_returns, mid_factor_returns) = pairs.iloc[mid_index]
(end_returns, end_factor_returns) = pairs.iloc[end_index]

factor_returns_range = (end_factor_returns - start_factor_returns)
start_returns_weight = 0.5
end_returns_weight = 0.5
 
#find weights for the start and end returns using a convex combination
if not factor_returns_range == 0:
    start_returns_weight =  (mid_factor_returns - start_factor_returns) / factor_returns_range
    end_returns_weight = (end_factor_returns - mid_factor_returns) / factor_returns_range

#calculate fragility heuristic
heuristic =  (start_returns_weight*start_returns) + (end_returns_weight*end_returns) - mid_returns

#returns 0.0
