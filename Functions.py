from datetime import datetime
import pandas as pd
import numpy as np
import requests
import time

def join_fundamentals(stock_list: np.array, Alphavantage_key: str, excel_name: str, error_safe = True, financial_statement = "INCOME_STATEMENT"):

    """ Stock_list: cargar excel sp500.xslx con pandas y seleccionar columna de tickers, debe ser un array de numpy.
        Alphavantage_key: Poner clave alphavantage.
        Excel_name: nombre del output sin .xlsx
        Output: DF con fundamentales.
        Se guarda un Excel automaticamente 
        Error_safe: True / False. Prompts not only the resulting DF but the tickers at which an error was encountered.
        If False it will only prompt the resulting DF
        Financial_statement: BALANCE_SHEET, INCOME_STATEMENT or CASH_FLOW"""

    raw_data = []
    wrong_tickers = []
    i = 0
    for ticker in stock_list:
        i = i + 1
        url = f'https://www.alphavantage.co/query?function={financial_statement}&symbol={ticker}&apikey={Alphavantage_key}'
        r = requests.get(url)
        income_statement = r.json()
        raw_data.append(income_statement)
        if i ==5:
            time.sleep(60)
            i = 0
            print(f"{int(round(list(stock_list).index(ticker)/len(stock_list)*100,0))}% complete")
    for j in range(len(stock_list)):
        try:
            if j == 0:
                idx = pd.MultiIndex.from_tuples([(stock_list[j],pd.DataFrame(raw_data[j]["quarterlyReports"]).columns[i]) 
                    for i in range(len(pd.DataFrame(raw_data[j]["quarterlyReports"]).columns))])
                df = pd.DataFrame(raw_data[j]["quarterlyReports"])
                df.columns = idx
            else:
                idx = pd.MultiIndex.from_tuples([(stock_list[j],pd.DataFrame(raw_data[j]["quarterlyReports"]).columns[i]) 
                    for i in range(len(pd.DataFrame(raw_data[j]["quarterlyReports"]).columns))])
                df2 = pd.DataFrame(raw_data[j]["quarterlyReports"])
                df2.columns = idx
                df = pd.concat([df,df2], axis = 1)
        except:
            print('Problem encountered')
            wrong_tickers.append(ticker)
            continue

    df.transpose().to_excel(f'{excel_name}.xlsx')
    print(f"Finished! Go look at your file: {excel_name}.xlsx saved next to this jupyter file.")
    if error_safe == True:
        return df, wrong_tickers
    else:
        return df

def load_partial_excel(df: pd.DataFrame):
    """
    For partial Excel files
    """
    df[df.columns[0]] = df[df.columns[0]].ffill()
    df = df.rename( 
        columns = {df.columns[0]:'Ticker', df.columns[1]:'Fundamental'}
    )
    df = df.set_index(
        pd.MultiIndex.from_frame( df[df.columns[:2]] )
        )[df.columns[2:]].transpose()
    return df

def missing_tickers(downloaded_assets:list, full_asset_list:list):
    compare = pd.Series([value in downloaded_assets for value in full_asset_list]) 
    compare = compare[compare == False].index.values
    return list(pd.Series(full_asset_list).iloc[list(compare)].values)

def in_both_lists(list_1:list, list_2:list):
    """" If one list is bigger assing it to list_2 """
    compare = pd.Series([value in list_1 for value in list_2]) 
    compare = compare[compare == True].index.values
    return list(pd.Series(list_2).iloc[list(compare)].values)

def load_full_excel(path: str):
    """" 
    For final Excel Files with all assets
    Path/Name.xlsx """
    df = pd.read_excel(path)
    df[df.columns[0]] = df[df.columns[0]].ffill()
    df = df.rename( 
        columns = {df.columns[0]:'Ticker', df.columns[1]:'Fundamental'}
    )
    df = df.set_index(
        pd.MultiIndex.from_frame( df[df.columns[:2]] )
        )[df.columns[2:]].transpose()
    return df

def quarters(dates_data):
    data = pd.Series(dates_data).dropna().values
    dates = [datetime.strptime(value,'%Y-%m-%d') for value in data]
    return np.array([date - pd.tseries.offsets.DateOffset(days=1) + pd.tseries.offsets.QuarterEnd() for date in  dates])

def assets(income_statement: pd.DataFrame):
    """
    income_statement: DataFrame with all the Income Statements. Functions.load_full_excel('Data/Income/Income_Statement.xlsx')
    """
    tickers = list(np.unique(np.array([income_statement.columns[i][0] for i in range(len(income_statement.columns))])))
    return tickers

def prices_date(balance_statement: pd.DataFrame, prices: pd.DataFrame, sp500: list):
    """
    Function that cleans dates in prices in order for them to match with the last fiscal date, for prices in weekends last prices available is taken.

    balance_statement: DataFrame with all the balance_statements. Functions.load_full_excel('Data/Balance/Balance_Statement.xlsx')
    prices: Precio de los activos. yf.download(tickers=sp500, start='2018-09-01', progress=False)['Adj Close']
    sp500: Lista de activos. Usar assets formula.

    """
    fiscal_endings = quarters(balance_statement[sp500[0]]['fiscalDateEnding'].values)
    # Filtrado Precios
    comparisson_list = prices.index.values
    dates_test = pd.to_datetime( [date if date in comparisson_list else comparisson_list[comparisson_list < date][-1] for date in fiscal_endings] )
    prices_filtered = prices.loc[dates_test] # Con fechas más cercanas al día fiscal
    prices_fiscal = prices_filtered.copy()
    prices_fiscal['fiscalDateEnding'] = fiscal_endings
    prices_fiscal = prices_fiscal.set_index('fiscalDateEnding')
    prices_fiscal.columns = pd.MultiIndex.from_tuples( [(value,'Adj Close') for value in prices_fiscal.columns.values] )
    return prices_fiscal


def clean_df(balance_statement: pd.DataFrame, income_statement: pd.DataFrame, sp500: list, prices_fiscal: list):
    """"
    Return a Clean DataFrame with selected variables ready to calculate finantial ratios.

    balance_statement: DataFrame with all the balance_statements. Functions.load_full_excel('Data/Balance/Balance_Statement.xlsx')
    income_statement: DataFrame with all the income statements. Functions.load_full_excel('Data/Income/Income_Statement.xlsx')
    sp500: Lista de activos. Usar assets formula.
    prices_fiscal: Precio de los activos despues de usar función prices_date.
    """
    # Datos necesarios de los Estados Financieros
    balance_cols = ['fiscalDateEnding','currentDebt','inventory','totalAssets','totalCurrentAssets','currentAccountsPayable','currentNetReceivables','commonStockSharesOutstanding']
    income_cols = ['totalRevenue','costofGoodsAndServicesSold','costOfRevenue','netIncome']
    # Formatos de Fecha homogeneos
    balance = balance_statement[sp500[0]][balance_cols]
    balance['fiscalDateEnding'] = quarters(balance['fiscalDateEnding'].values)
    balance = balance.set_index('fiscalDateEnding')
    income = income_statement[sp500[0]][income_cols]
    income['fiscalDateEnding'] = balance.index.values
    income = income.set_index('fiscalDateEnding')
    # Unir columnas
    company = pd.concat([balance,income,prices_fiscal[sp500[0]]], axis = 1)
    company.columns = pd.MultiIndex.from_tuples( [(sp500[0],value) for value in company.columns.values] )
    companies = company.copy()
    for ticker in sp500[1:]:
        # Formatos de Fecha homogeneos
        balance = balance_statement[sp500[0]][balance_cols]
        balance['fiscalDateEnding'] = quarters(balance['fiscalDateEnding'].values)
        balance = balance.set_index('fiscalDateEnding')
        income = income_statement[ticker][income_cols]
        income['fiscalDateEnding'] = balance.index.values
        income = income.set_index('fiscalDateEnding')
        # Unir columnas
        company = pd.concat([balance,income,prices_fiscal[ticker]], axis = 1)
        company.columns = pd.MultiIndex.from_tuples( [(ticker,value) for value in company.columns.values] )
        companies = pd.concat([companies,company], axis=1)
    return companies

def tabular_df(financial_info: pd.DataFrame, sp500: list):
    """
    financial_info: DataFrame resultant from using function clean_df.
    sp500: Lista de activos. Usar assets formula.
    """
    table = []
    for ticker in sp500:
        partial = financial_info[ticker].reset_index()
        partial['Stock'] = ticker
        table.append(partial)
    table = pd.concat(table, axis=0)
    order = ['Stock'] + list(table.columns.values[:-1])
    table = table.reindex(columns=order)
    table = table.fillna(0)
    for column in table.columns[2:-1]:
        table[column] = table[column].apply(float)
    return table

def PER(data_table: pd.DataFrame):
    """"
    data_table: DataFrame resultant from using tabular_df function.
    """
    ratio = data_table['Adj Close'] / (data_table['netIncome'] / data_table['commonStockSharesOutstanding'])
    return ratio


def PBV(data_table: pd.DataFrame):
    """"
    data_table: DataFrame resultant from using tabular_df function.
    """
    ratio = ( data_table['commonStockSharesOutstanding'] *  data_table['Adj Close'] ) / data_table['totalAssets']
    return ratio

def acid(data_table: pd.DataFrame):
    """"
    data_table: DataFrame resultant from using tabular_df function.
    """
    ratio = ( data_table['totalCurrentAssets'] *  data_table['inventory'] ) / data_table['currentDebt']
    return ratio

def AR(data_table: pd.DataFrame):
    """"
    data_table: DataFrame resultant from using tabular_df function.
    """
    ratio = ( data_table['totalCurrentAssets'] *  data_table['inventory'] ) / data_table['currentDebt']
    return ratio

def CCC(data_table: pd.DataFrame):
    """"
    data_table: DataFrame resultant from using tabular_df function.
    """ 
    days_sales_of_inventory = 365 / ( data_table['costOfRevenue'] / data_table['inventory'] )
    days_sales_outstanding = 365 / ( data_table['totalRevenue'] / data_table['currentNetReceivables'] )
    days_payables_outstanding = 365 / ( data_table['costOfRevenue'] / data_table['currentAccountsPayable'] )
    return days_sales_of_inventory + days_sales_outstanding - days_payables_outstanding

def financial_ratios(data_table: pd.DataFrame):
    """"
    data_table: DataFrame resultant from using tabular_df function.
    """ 
    financials = data_table.copy()
    financials['PER'] = PER(data_table)
    financials['PBV'] = PBV(data_table)
    financials['Acid'] = acid(data_table)
    financials['AR'] = AR(data_table)
    financials['CCC'] = CCC(data_table)
    return financials[['Stock','fiscalDateEnding','PER','PBV','Acid','AR','CCC']]


