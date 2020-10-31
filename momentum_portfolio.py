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

'''
Helper functions
'''
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

def volatility(ts: pd.Series):
    return ts.pct_change().rolling(vola_window).std().iloc[-1]

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
    """
    Initialization and trading logic
    """
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
    """
    Rebalance portfolio
    """
    # Write some progress output during the backtest
    output_progress(context)

    #First get today's date (simulation's "today")
    today = zipline.api.get_datetime()

    # Second, get the index makeup for all days prior to today
    all_prior = context.index_members[context.index_members.index < today]

    # Now let's snag the first column of the last row, i.e. latest entry
    latest_day = all_prior.iloc[-1,0]

    # Split text string with tickers into a list 
    list_of_tickers = latest_day.split(',')

    # Finally, get the Zipline symbols for the tickers
    todays_universe = [symbol(ticker) for ticker in list_of_tickers]

    # Get today's universe historical data
    hist = data.history(todays_universe, 'close', momentum_window, '1d')

    # Make momentum ranking table
    ranking_table = hist.apply(momentum_score).sort_values(ascending=False)

    # Sell logic
    # First we check if any existing position should be sold.
    # * Sell if stock is no longer part of the index
    # * Sell if stock has too low momentum value
    kept_positions = list(context.portfolio.positions.keys())

    for security in context.portfolio.positions:
        if security not in todays_universe:
            order_target_percent(security, 0.0)
            kept_positions.remove(security)
        elif ranking_table[security] < minimum_momentum:
            order_target_percent(security, 0.0)
            kept_positions.remove(security)

    # Stock selection logic
    # Check how many stocks we are keeping from last month
    # Fill from top of the ranking list, until we reach the desired total number of portfolio holdings
    replacement_stocks = portfolio_size - len(kept_positions)
    
    buy_list = ranking_table.loc[~ranking_table.index.isin(kept_positions)][:replacement_stocks]

    new_portfolio = pd.concat(buy_list, ranking_table.loc[ranking_table.index.isin(kept_positions)])

    # Calculate inverse volatility for stocks and make target position wheights
    vola_table = hist[new_portfolio.index].apply(volatility)
    inv_vola_table = 1 / vola_table
    sum_inv_vola = np.sum(inv_vola_table)
    vola_target_weights = inv_vola_table / sum_inv_vola

    for security, rank in new_portfolio.iteritems():
        weight = vola_target_weights[security]
        if security in kept_positions:
            order_target_percent(security, weight)
        else:
            if ranking_table[security] > minimum_momentum:
                order_target_percent(security, weight)

def analyze(context, perf):
    perf['max'] = perf.portfolio_value.cummax()
    perf['dd'] = (perf.portfolio_value / perf['max']) - 1
    max_dd = perf['dd'].min()

    portfolio_start_value = perf['portfolio_value'].iloc[0]
    portfolio_end_value = perf['portfolio_value'].iloc[-1]

    ann_ret = (np.power(portfolio_end_value/portfolio_start_value), (252/len(perf)))

    print('Annualized Return: {:.2%} Max Drawdown: {:.2%}'.format(ann_ret, max_dd))

start_date = pd.Timestamp('1997-1-1', tz='utc')
end_date = pd.Timestamp('2018-12-31', tz='utc')

perf = zipline.run_algorithm(
    start=start_date, 
    end=end_date, 
    initialize=initialize, 
    analyze=analyze, 
    capital_base=initial_portfolio,
    data_frequency='daily',
    bundle='quandl')

