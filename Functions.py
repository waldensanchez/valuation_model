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