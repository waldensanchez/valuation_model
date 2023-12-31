from sklearn.impute import KNNImputer
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


def clean_df(balance_statement: pd.DataFrame, income_statement: pd.DataFrame, sp500: list, prices_fiscal: pd.DataFrame):
    """"
    Return a Clean DataFrame with selected variables ready to calculate finantial ratios.

    balance_statement: DataFrame with all the balance_statements. Functions.load_full_excel('Data/Balance/Balance_Statement.xlsx')
    income_statement: DataFrame with all the income statements. Functions.load_full_excel('Data/Income/Income_Statement.xlsx')
    sp500: Lista de activos. Usar assets formula.
    prices_fiscal: Precio de los activos despues de usar función prices_date.
    """
    # Datos necesarios de los Estados Financieros
    balance_cols = ['fiscalDateEnding','currentDebt','inventory','totalAssets','totalCurrentAssets',
                    'currentAccountsPayable','currentNetReceivables','commonStockSharesOutstanding','totalLiabilities','totalShareholderEquity']
    income_cols = ['totalRevenue','costofGoodsAndServicesSold','costOfRevenue','netIncome']

    # Get Returns
    returns = prices_fiscal.pct_change()
    returns.columns = pd.MultiIndex.from_tuples([( returns.columns[i][0],'Return') for i in range(len(returns.columns))])

    # Formatos de Fecha homogeneos
    balance = balance_statement[sp500[0]][balance_cols]
    balance['fiscalDateEnding'] = quarters(balance['fiscalDateEnding'].values)
    balance = balance.set_index('fiscalDateEnding')
    income = income_statement[sp500[0]][income_cols]
    income['fiscalDateEnding'] = balance.index.values
    income = income.set_index('fiscalDateEnding')
    # Unir columnas
    company = pd.concat([balance,income,prices_fiscal[sp500[0]],returns[sp500[0]]], axis = 1)
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
        company = pd.concat([balance,income,prices_fiscal[ticker],returns[ticker]], axis = 1)
        company.columns = pd.MultiIndex.from_tuples( [(ticker,value) for value in company.columns.values] )
        companies = pd.concat([companies,company], axis=1)
    companies = companies.iloc[1:]
    for ticker in sp500:
        companies[(ticker,'Return')] = companies[(ticker,'Return')].apply(lambda x: 1 if x > 0 else 0)
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

def Acid_test(data_table: pd.DataFrame):
    """"
    data_table: DataFrame resultant from using tabular_df function.
    """
    ratio = ( data_table['totalCurrentAssets'] -  data_table['inventory'] ) / data_table['currentDebt']
    return ratio

def ATR(data_table: pd.DataFrame):
    """"
    data_table: DataFrame resultant from using tabular_df function.
    """
    ratio = data_table['totalRevenue'] / data_table['totalAssets']
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

def ROA(data_table: pd.DataFrame):
    """"
    data_table: DataFrame resultant from using tabular_df function.
    """ 
    ratio = data_table['netIncome'] / data_table['totalAssets']
    return ratio

def DER(data_table: pd.DataFrame):
    """"
    data_table: DataFrame resultant from using tabular_df function.
    """ 
    ratio = data_table['totalLiabilities'] / data_table['totalShareholderEquity']
    return ratio

def NPM(data_table: pd.DataFrame):
    """"
    data_table: DataFrame resultant from using tabular_df function.
    """ 
    ratio = data_table['netIncome'] / data_table['totalRevenue']
    return ratio

def EM(data_table: pd.DataFrame):
    """"
    data_table: DataFrame resultant from using tabular_df function.
    """ 
    ratio = data_table['totalAssets'] / data_table['totalShareholderEquity']
    return ratio

def financial_ratios(data_table: pd.DataFrame):
    """"
    data_table: DataFrame resultant from using tabular_df function.
    """ 
    financials = data_table.copy()
    financials['PER'] = PER(data_table)
    financials['PBV'] = PBV(data_table)
    financials['Acid_test'] = Acid_test(data_table)
    financials['ATR'] = ATR(data_table)
    financials['CCC'] = CCC(data_table)
    financials['ROA'] = ROA(data_table)
    financials['DER'] = DER(data_table)
    financials['NPM'] = NPM(data_table)
    financials['EM'] = EM(data_table)
    return financials[['Stock','fiscalDateEnding','PER','PBV','Acid_test','ATR','CCC','ROA','DER','NPM','EM','Return']]

def dqr(data):
    # Lista de variables de la base de datos
    columns = pd.DataFrame(list(data.columns.values), columns=['Nombres'], index=list(data.columns.values))

    # Lista de tipos de datos
    data_types = pd.DataFrame(data.dtypes, columns=['Data_Type'])

    #  Lista de valores perdidos (NaN)
    missing_values = pd.DataFrame(data.isnull().sum(), columns=['Missing_Values'])

    # Lista de valores presentes
    present_values = pd.DataFrame(data.count(), columns=['Present_Values'])

    # Número de valores únicos para cada variable
    unique_values = pd.DataFrame(columns=['Num_Unique_Values'])
    for col in list(data.columns.values):
        unique_values.loc[col] = [data[col].nunique()]

    # Lista de valores mínimos para cada variable
    min_values = pd.DataFrame(columns=['Min'])
    for col in list(data.columns.values):
        try:
            min_values.loc[col] = [data[col].min()]
        except:
            pass

    # Lista de valores máximos para cada variable
    max_values = pd.DataFrame(columns=['Max'])
    for col in list(data.columns.values):
        try:
            max_values.loc[col] = [data[col].max()]
        except:
            pass
    # Columna 'Categórica' que obtenga un valor booleano True cuando mi columna es una variable
    # Categórica; y False, cuando sea Numérica
    categorical = pd.DataFrame(columns=['Categorical'])
    for col in list(data.columns.values):
        if data[col].dtype == 'object':
            categorical.loc[col] = True
        else:
            categorical.loc[col] = False

    # Si es categórica no mayor a 20 elementos únicos, anexar sus valores
    cat_values = pd.DataFrame(columns=['Categories'])
    for col in list(data.columns.values):
        if data[col].dtype == 'object' and data[col].nunique() < 21:
            cat_values.loc[col] = [data[col].unique()]
        elif data[col].dtype == 'object' and data[col].nunique() > 20:
            cat_values.loc[col] = 'Category too large'
        else:
            cat_values.loc[col] = 'Not categorical'

    # Unión de tablas / DataFrames
    return columns.join(data_types).join(missing_values).join(present_values).join(unique_values).join(min_values).join(max_values).join(categorical).join(cat_values)

def clean_ratios_function(df: pd.DataFrame):
    # Conservar solo ratios y reemplazo de infinito (y -infinito) por NaN
    ratios_only = df.drop(['Stock','fiscalDateEnding','Return'],axis=1).replace(np.inf,np.nan).replace(-np.inf,np.nan)
    # Imputación de valores faltantes 
    imputer = KNNImputer(n_neighbors=5)
    imputed_ratios = imputer.fit_transform(ratios_only)
    df[['PER','PBV','Acid_test','ATR','CCC','ROA','DER','NPM','EM']] = imputed_ratios
    return df