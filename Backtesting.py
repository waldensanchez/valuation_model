import pandas as pd
class Backtesting():
    def __init__(self, initial_capital: int = 10**6):
        self.cash = initial_capital
        self.current_open_positions = pd.DataFrame(columns=['Date','Asset','Position'])
        self.operations = pd.DataFrame(columns=['Date','Stock','X','Price','Position','Type'])

    def Trade(self):
        sell_rf(self)

def open_positions(current_open_positions: pd.DataFrame, consult_asset: str):
    return consult_asset in  current_open_positions.Assets.values

def valuate_position(current_open_positions: pd.DataFrame, consult_asset: str):
    rf_open_position = current_open_positions[current_open_positions['Asset'] == consult_asset]['Position'].values[0] # Verify Work
    position_date = current_open_positions[current_open_positions['Asset'] == consult_asset]['Date'].values[0]        # Verify Work
    return   rf_open_position, position_date

def close_position(current_open_positions: pd.DataFrame):
    current_open_positions = current_open_positions.drop(['Rf'], axis = 0)
    return current_open_positions

def report_sale(operations: pd.DataFrame, rf_open_position: float, position_date: str, fiscal_date: str):
    previous_rf = operations[(operations['Date'] == position_date) & (operations['Stock'] == 'Rf') & (operations['Type'] == 'Buy')]['Price'].values[0]
    income_risk_free =  rf_open_position * ( 1 + previous_rf *  3/12)
    sell_operation = pd.DataFrame([fiscal_date,'Rf',income_risk_free,previous_rf,-income_risk_free,'Sell'], index = ['Date','Stock','X','Price','Position','Type']).T
    operations = pd.concat([operations,sell_operation], axis = 1, ignore_index = True)
    return operations, income_risk_free

def update_funds(income_risk_free: float):
    return income_risk_free
    
def sell_rf(current_open_positions: pd.DataFrame, operations: pd.DataFrame, fiscal_date: str):
    if open_positions(current_open_positions, 'Rf'):
        rf_open_position, position_date = valuate_position(current_open_positions, 'Rf')
        current_open_positions = close_position(current_open_positions)
        operations, income_risk_free = report_sale(operations, rf_open_position, position_date, fiscal_date)
        cash = update_funds(cash, income_risk_free)
    else:
        pass
    return current_open_positions, operations, cash