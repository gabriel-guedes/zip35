from scipy import stats
import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
import zipline
from zipline.api import symbol, order_target_percent, set_commission, set_slippage, schedule_function, time_rules, date_rules
from zipline.finance.commission import PerDollar
from zipline.finance.slippage import VolumeShareSlippage, FixedSlippage
import pyfolio as pf

# Model Settings
initial_portfolio = 100000
momentum_window = 125
minimum_momentum = 40
portfolio_size = 30
vola_window = 20

# Comission and Slippage Settings
enable_commission = True
commission_pct = 0.001
enable_slippage = True
slippage_volume_limit = 0.025
slippage_impact = 0.05

# Momentum Score Function
def momentum_score(ts: pd.Series):
    """
    Input: price time series
    Output: Annualized exponential regression slope multiplied by the R2
    """

    # Make a list of consecutive numbers
    x = np.arange(len(ts))

    # Get logs
    log_ts = np.log(ts)

    # Calculate regression values
    slope, intercept, r_value, p_value, std_err = stats.linregress(x, log_ts)

    # Annualize percent
    annualized_slope = (np.power(np.exp(slope), 252) -1) * 100

    # Adjust for fitness
    score = annualized_slope * (r_value ** 2)
    
    return score

# Volatility Function
def volatility(ts: pd.Series):
    return ts.pct_change().rolling(vola_window).std().iloc[-1]

# Output progress function
def output_progress(context):
    """
    Output some performance numbers during backtest run.
    This code just prints out the past month's performance 
    so that we have something to look at while the backtest runs.
    """

    # Get today's date
    today = zipline.api.get_datetime().date()

    # Calculate percent difference since last month
    perf_pct = (context.portfolio.portfolio_value / context.last_month) - 1

    # Print performance, format as percent with 2 decimals
    print('{} - Last Month Result: {:.2%}'.format(today, perf_pct))

    # Remember today's portfolio value for next month's calculation
    context.last_month = context.portfolio.portfolio_value

def initialize(context):
    # Set comission and slippage
    if enable_commission:
        comm_model = PerDollar(cost=commission_pct)
    else:
        comm_model = PerDollar(cost=0.0)
    
    set_commission(comm_model)

    if enable_slippage:
        slippage_model = VolumeShareSlippage(volume_limit=slippage_volume_limit, price_impact=slippage_impact)
    else:
        slippage_model = FixedSlippage(spread=0.0)

    # Used only for progress output
    context.last_month = initial_portfolio

    # Store index membership
    context.index_members = pd.read_csv('./data/sp500_members.csv', index_col=0, parse_dates=[0])

    # Schedule rebalance monthly
    schedule_function(func=rebalance, date_rule=date_rules.month_start(), time_rule=time_rules.market_open())

def rebalance(context, data):
    # Write some progress output during the backtest
    output_progress(context)

    #First get today's date (simulation's "today")
    today = zipline.api.get_datetime()

    # Second, get the index makeup for all days prior to today
    all_prior = context.index_members[context.index_members.index < today]

    # Now let's snag the first column of the last row, i.e. latest entry
    latest_day = all_prior.iloc[-1,0]