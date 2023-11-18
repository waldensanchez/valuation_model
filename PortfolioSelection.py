import pandas as pd
import yfinance as yf
import AssetAllocation as AA
from datetime import datetime, timedelta

# Asset Picker Functions
def verify_assets(data, portfolio, fiscal_date):
    portfolio = portfolio[portfolio.Asset != 'Rf']
    predictor = data[(data.fiscalDateEnding == fiscal_date) & (data.Yhat == 0)].merge(portfolio, how='inner', left_on='Stock', right_on='Asset')
    assets_portfolio = list(predictor.Asset)
    return assets_portfolio

def fill_portfolio(data, assets_portfolio, fiscal_date):
    if len(assets_portfolio) == 5:
        return assets_portfolio
    else:
        predictor = data[(data.fiscalDateEnding == fiscal_date) & (data.Yhat == 0)]
        predictions_available = len(predictor)
        already_selected = len(assets_portfolio)
        missing_assets = 5 - already_selected
        if predictions_available >= missing_assets:
            additional_assets = predictor.sample(n = missing_assets).Stock.values
            additional_assets = list(additional_assets)
            assets_list = assets_portfolio + additional_assets
            return assets_list
        elif predictions_available > 0:
            additional_assets = predictor.Stock.values
            additional_assets = list(additional_assets)
            assets_list = assets_portfolio + additional_assets
            return assets_list
        else:
            return ['Rf']
        
        

# Asset Picker
def asset_picker(data, portfolio, fiscal_date):
    assets_portfolio = verify_assets(data, portfolio, fiscal_date)
    assets_list = fill_portfolio(data, assets_portfolio, fiscal_date)
    return assets_list

# Asset Allocation Functions
def validate_operation(assets_list):
        return 'Rf' not in assets_list

def safe_yf(tickers, start_date, end_date):
    try:
        prices = yf.download(tickers, start=start_date, end=end_date, progress=False)['Adj Close']
    except:
        try:
            prices = yf.download(tickers, start=start_date, end=end_date, progress=False)['Adj Close']
        except:
            try:
                prices = yf.download(tickers, start=start_date, end=end_date, progress=False)['Adj Close']
            except:
                prices = yf.download(tickers, start=start_date, end=end_date, progress=False)['Adj Close']
    return prices

def omega(data, assets_list, fiscal_date, mkt_idx: str = '^GSPC'):
    rf_data = data[data.fiscalDateEnding == fiscal_date]
    rf_rate = rf_data.rf.values[0] / 100
    if len(assets_list) > 0:
        tickers = assets_list.copy()
        tickers.append(mkt_idx)
        end_date = pd.to_datetime(fiscal_date)
        start_date = end_date + timedelta(days = -365)
        prices = safe_yf(tickers, start_date, end_date)
        data_stocks = prices[prices.columns[:-1]]
        data_benchmark = prices[prices.columns[-1]].to_frame()
        omega = AA.asset_allocation(data_stocks=data_stocks, data_benchmark=data_benchmark, rf=rf_rate)
        weights = omega.omega(n_port=1000)
        weights = pd.DataFrame(weights, index=assets_list, columns=['Weight'])
        return weights
    
# Asset Allocation
def asset_allocation(data, assets_list, fiscal_date):
    if validate_operation(assets_list):
        weights = omega(data, assets_list, fiscal_date)
    else:
        weights = pd.DataFrame([1], index=['Rf'], columns=['Weight'])
    return weights

# Valuate Portfolio
def valuate_portfolio(data, portfolio, fiscal_date):
    prices = data[data.fiscalDateEnding == fiscal_date].merge(portfolio, how='right', left_on='Stock', right_on='Asset')
    portfolio_value = prices[['Amount','Adj Close']]
    portfolio_value = portfolio_value['Amount'] * portfolio_value['Adj Close']
    portfolio_value = portfolio_value.sum()
    return portfolio_value