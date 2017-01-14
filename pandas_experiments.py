import pandas as pd

#combine returns and factor returns into pairs
returns_series = pd.Series([30, 80, -10], index=['A', 'B', 'C'], name='unwashed_returns')
factor_returns_series = pd.Series([5, 4, -1], index=['A', 'B', 'C'], name='unwashed_factor_returns')
pairs = pd.concat([returns_series, factor_returns_series], axis=1)
pairs.columns = ['returns', 'factor_returns']

#sort by beta
pairs.sort_values(by='factor_returns')

#find the three vectors, using median of 3
start_index = 0
mid_index = int(round(len(pairs) / 2, 0))
end_index = len(pairs) - 1

(start_returns, start_factor_returns) = pairs.iloc[start_index]
(mid_returns, mid_factor_returns) = pairs.iloc[mid_index]
(end_returns, end_factor_returns) = pairs.iloc[end_index]

#find weights for the start and end returns using a convex combination
start_returns_weight =  (mid_factor_returns - start_factor_returns) / (end_factor_returns - start_factor_returns)
end_returns_weight = (end_factor_returns - mid_factor_returns) / (end_factor_returns - start_factor_returns)

#calculate fragility heuristic
heuristic =  (start_returns_weight*start_returns) + (end_returns_weight*end_returns) - mid_returns

#returns 40.0 as the result


#combine returns and factor returns into pairs
returns_series = pd.Series([30, 15, 80, -10, -900], index=['A', 'B', 'C', 'D', 'E'], name='unwashed_returns')
factor_returns_series = pd.Series([5, 2, 4, -1, -3], index=['A', 'B', 'C', 'D', 'E'], name='unwashed_factor_returns')
pairs = pd.concat([returns_series, factor_returns_series], axis=1)
pairs.columns = ['returns', 'factor_returns']

#sort by beta
pairs.sort_values(by='factor_returns')

#find the three vectors, using median of 3
start_index = 0
mid_index = int(round(len(pairs) / 2, 0))
end_index = len(pairs) - 1

(start_returns, start_factor_returns) = pairs.iloc[start_index]
(mid_returns, mid_factor_returns) = pairs.iloc[mid_index]
(end_returns, end_factor_returns) = pairs.iloc[end_index]

#find weights for the start and end returns using a convex combination
start_returns_weight =  (mid_factor_returns - start_factor_returns) / (end_factor_returns - start_factor_returns)
end_returns_weight = (end_factor_returns - mid_factor_returns) / (end_factor_returns - start_factor_returns)

#calculate fragility heuristic
heuristic =  (start_returns_weight*start_returns) + (end_returns_weight*end_returns) - mid_returns

#returns -863.75 as the result


#combine returns and factor returns into pairs
returns_series = pd.Series([.30, .15, .80, -.10, -.30], index=['A', 'B', 'C', 'D', 'E'], name='unwashed_returns')
factor_returns_series = pd.Series([.05, .02, .04, -.01, -.03], index=['A', 'B', 'C', 'D', 'E'], name='unwashed_factor_returns')
pairs = pd.concat([returns_series, factor_returns_series], axis=1)
pairs.columns = ['returns', 'factor_returns']

#sort by beta
pairs.sort_values(by='factor_returns')

#find the three vectors, using median of 3
start_index = 0
mid_index = int(round(len(pairs) / 2, 0))
end_index = len(pairs) - 1

(start_returns, start_factor_returns) = pairs.iloc[start_index]
(mid_returns, mid_factor_returns) = pairs.iloc[mid_index]
(end_returns, end_factor_returns) = pairs.iloc[end_index]

#find weights for the start and end returns using a convex combination
start_returns_weight =  (mid_factor_returns - start_factor_returns) / (end_factor_returns - start_factor_returns)
end_returns_weight = (end_factor_returns - mid_factor_returns) / (end_factor_returns - start_factor_returns)

#calculate fragility heuristic
heuristic =  (start_returns_weight*start_returns) + (end_returns_weight*end_returns) - mid_returns

#returns -1.0250000000000001 as result as the result


#combine returns and factor returns into pairs
returns_series = pd.Series([.30, .80, -.10], index=['A', 'B', 'C'], name='unwashed_returns')
factor_returns_series = pd.Series([.05, .04, -.01], index=['A', 'B', 'C'], name='unwashed_factor_returns')
pairs = pd.concat([returns_series, factor_returns_series], axis=1)
pairs.columns = ['returns', 'factor_returns']

#sort by beta
pairs.sort_values(by='factor_returns')

#find the three vectors, using median of 3
start_index = 0
mid_index = int(round(len(pairs) / 2, 0))
end_index = len(pairs) - 1

(start_returns, start_factor_returns) = pairs.iloc[start_index]
(mid_returns, mid_factor_returns) = pairs.iloc[mid_index]
(end_returns, end_factor_returns) = pairs.iloc[end_index]

#find weights for the start and end returns using a convex combination
start_returns_weight =  (mid_factor_returns - start_factor_returns) / (end_factor_returns - start_factor_returns)
end_returns_weight = (end_factor_returns - mid_factor_returns) / (end_factor_returns - start_factor_returns)

#calculate fragility heuristic
heuristic =  (start_returns_weight*start_returns) + (end_returns_weight*end_returns) - mid_returns

#returns 0.40000000000000002 as result



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
factor_returns_series = mixed_returns
pairs = pd.concat([returns_series, factor_returns_series], axis=1)
pairs.columns = ['returns', 'factor_returns']

#sort by beta
pairs.sort_values(by='factor_returns')

#find the three vectors, using median of 3
start_index = 0
mid_index = int(round(len(pairs) / 2, 0))
end_index = len(pairs) - 1

(start_returns, start_factor_returns) = pairs.iloc[start_index]
(mid_returns, mid_factor_returns) = pairs.iloc[mid_index]
(end_returns, end_factor_returns) = pairs.iloc[end_index]

#find weights for the start and end returns using a convex combination
start_returns_weight =  (mid_factor_returns - start_factor_returns) / (end_factor_returns - start_factor_returns)
end_returns_weight = (end_factor_returns - mid_factor_returns) / (end_factor_returns - start_factor_returns)

#calculate fragility heuristic
heuristic =  (start_returns_weight*start_returns) + (end_returns_weight*end_returns) - mid_returns

#returns 0.40000000000000002 as result
