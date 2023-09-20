# Importar librerías 
import pandas as pd
import numpy as np
import Functions
import requests
import time
import os

# SP 500 List
sp500 = list(pd.read_excel('sp500.xlsx')['Symbol'].values)

# Income Statement Directory
income_path = 'Data/Income/Parts'
income_dir = os.listdir(income_path)

# Balance Statement Directory
balance_path = 'Data/Balance/Parts'
balance_dir = os.listdir(balance_path)

# Join Income Statement Information
all_income = [ Functions.load_partial_excel(pd.read_excel(income_path+'/'+income_dir[i])).transpose() for i in range(len(income_dir)) ]
income_statement = pd.concat(all_income, ignore_index=False, axis=0).transpose()

# Join Balance Statement Information
all_balance = [ Functions.load_partial_excel(pd.read_excel(balance_path+'/'+balance_dir[i])).transpose() for i in range(len(balance_dir)) ]
balance_statement = pd.concat(all_balance, ignore_index=False, axis=0).transpose()

downloaded_tickers_income = pd.Series([income_statement.columns.values[i][0] for i in range(len(income_statement.columns.values))]).unique()
downloaded_tickers_balance = pd.Series([balance_statement.columns.values[i][0] for i in range(len(balance_statement.columns.values))]).unique()

tickers_in_data = Functions.in_both_lists(list_1=downloaded_tickers_balance,list_2=downloaded_tickers_income)

balance_statement[tickers_in_data].transpose().to_excel('Data/Balance/Balance_Statement.xlsx')

income_statement[tickers_in_data].transpose().to_excel('Data/Income/Income_Statement.xlsx')