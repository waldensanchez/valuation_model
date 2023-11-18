import pandas as pd
import numpy as np

# Trade Functions
def close_position(portfolio: pd.DataFrame, asset: str):
    if asset == 'Rf':
        portfolio = portfolio[portfolio.Asset != asset]
        return portfolio
    else:
        # Capitals
        pass

# Sell Risk Free Functions
def get_cash(portfolio):
    rf_position = portfolio[portfolio.Asset == 'Rf']
    rf = rf_position.Price.iloc[0] / 100
    cash = rf_position.Amount.iloc[0] * ( 1 + rf *  3/12)
    return cash

## Sell Risk Free
def sell_risk_free(portfolio, verbose):
    if 'Rf' in portfolio.Asset.values:
        cash = get_cash(portfolio)
        portfolio = close_position(portfolio, 'Rf')
    else:
        cash = 0
    if verbose:
        print('Cash from RF: ', cash)
    return cash, portfolio

## Calculate Capital's Positions
def calculate_positions(data, portfolio, weights, portfolio_value, cash, fiscal_date):
    # Add Money from RF to capitals value
    portfolio_value += cash
    print('Portfolio Value: ', portfolio_value)
    # Old Positions
    previous_positions = portfolio[['Asset','Amount']] 
    previous_positions['Amount'] = previous_positions['Amount'] * -1
    previous_positions = previous_positions.rename(columns = {'Amount':'Old'})
    # New Positions
    new_positions = weights 
    print('Weights')
    print(weights)
    new_positions['Weight'] = new_positions['Weight'] * portfolio_value
    new_positions = new_positions.reset_index()
    new_positions.columns = ['Asset','CashAmt']
    prices = data[data.fiscalDateEnding == fiscal_date][['Stock','Adj Close']]
    new_positions = new_positions.merge(prices, how='left', left_on='Asset', right_on='Stock')
    new_positions['New'] = new_positions['CashAmt'] / new_positions['Adj Close']
    new_positions['New'] = np.floor(new_positions['New'])
    new_positions = new_positions[['Asset','New']]
    # Generate Trade Order
    trade_order = previous_positions.merge(new_positions, how='outer', on='Asset')
    trade_order = trade_order.fillna(0)
    trade_order['TradeOrder'] = trade_order['New'] + trade_order['Old']
    print('TRADE ORDER')
    print(trade_order)
    trade_order = trade_order[['Asset','TradeOrder']]
    return trade_order

# Portfolio adjustments functions
# No comission considered
def sell_capitals(data, trade_order, portfolio, cash, fiscal_date, verbose):
    # Current prices
    prices = data[data.fiscalDateEnding == fiscal_date][['Stock','Adj Close']]
    # Sell positions
    sell_orders = trade_order[trade_order.TradeOrder < 0]
    print('Sell Order')
    print(sell_orders)
    for asset in sell_orders.Asset:
        available_amount = portfolio[portfolio.Asset == asset].Amount.values[0]
        amount_to_sell = sell_orders[sell_orders.Asset == asset].TradeOrder.values[0]
        sell_price = prices[prices.Stock == asset]['Adj Close'].values[0]
        # Sell whole position
        if available_amount <= abs(amount_to_sell):
            # Position update
            portfolio = close_position(portfolio, asset)
            # Cash update
            cash -= available_amount * sell_price
        # Sell part of the position
        else:
            # Position update
            portfolio.loc[(portfolio.Asset == asset), 'Amount'] += amount_to_sell
            # Cash update
            cash -= amount_to_sell * sell_price
        if verbose:
            print('Cash from capitals: ', -amount_to_sell * sell_price, ' Remaining: ', cash)
    return portfolio, cash
    
def buy_capitals(data, trade_order, portfolio, cash, fiscal_date, verbose):
    # Current prices
    prices = data[data.fiscalDateEnding == fiscal_date][['Stock','Adj Close']]
    # Buy positions
    buy_orders = trade_order[trade_order.TradeOrder > 0]
    for asset in buy_orders.Asset:
        buy_price = prices[prices.Stock == asset]['Adj Close'].values[0]
        # Adjust Existing Position
        if asset in portfolio.Asset.values:
            existing_position = portfolio[portfolio.Asset == asset].Amount.values[0]
            amount_to_buy = buy_orders[buy_orders.Asset == asset].TradeOrder.values[0]
            new_position = existing_position + amount_to_buy
            initial_price = portfolio[portfolio.Asset == asset].Price.values[0]
            updated_price = initial_price * (existing_position / new_position) + buy_price * (amount_to_buy / new_position) 
            # Position update 
            portfolio.loc[(portfolio.Asset == asset), 'Date'] = fiscal_date
            portfolio.loc[(portfolio.Asset == asset), 'Amount'] = new_position
            portfolio.loc[(portfolio.Asset == asset), 'Amount'] = updated_price # Ponderate prices
            # Cash update
            cash -= new_position * buy_price
        # Sell whole position
        else:
            amount_to_buy = buy_orders[buy_orders.Asset == asset].TradeOrder.values[0]
            # Position update 
            open_position = pd.DataFrame([fiscal_date,asset,amount_to_buy, buy_price], index = ['Date','Asset','Amount','Price']).T
            portfolio = pd.concat([portfolio, open_position], axis=0, ignore_index=True)
            # Cash update
            cash -= amount_to_buy * buy_price
        if verbose:
            print('Cash spent in capitals: ', amount_to_buy * buy_price, ' Remaining: ', cash)
    return portfolio, cash

## Portfolio Adjustments
def portfolio_adjustments(data, trade_order, portfolio, cash, fiscal_date, verbose):
    portfolio, cash = sell_capitals(data, trade_order, portfolio, cash, fiscal_date, verbose)
    portfolio, cash = buy_capitals(data, trade_order, portfolio, cash, fiscal_date, verbose)
    return portfolio, cash

## Buy Risk Free
def buy_risk_free(data, portfolio, cash, fiscal_date, verbose):
    risk_free_rate = data[data.fiscalDateEnding == fiscal_date].rf.iloc[0]
    asset = 'Rf'
    open_position = pd.DataFrame([fiscal_date, asset, cash, risk_free_rate], index = ['Date','Asset','Amount','Price']).T
    portfolio = pd.concat([portfolio, open_position], axis=0, ignore_index=True)
    if verbose:
        print('Rf bought:', cash)
    return portfolio

### Trade
def trade(data, portfolio, weights, portfolio_value, assets_list, fiscal_date, initial_trade, initial_capital, verbose):
    if initial_trade:
        cash = initial_capital
    else:
        cash, portfolio = sell_risk_free(portfolio, verbose)
        if 'Rf' in assets_list:
            #assets_list.remove('Rf')
            pass
    if 'Rf' not in assets_list:
        trade_order = calculate_positions(data, portfolio, weights, portfolio_value, cash, fiscal_date)
        portfolio, cash = portfolio_adjustments(data, trade_order, portfolio, cash, fiscal_date, verbose)
    portfolio = buy_risk_free(data, portfolio, cash, fiscal_date, verbose)
    return portfolio, cash