import pandas as pd
import Functions

class Companies():
    def __init__(self, df) -> None:
        self.df = df

    def get_ratios(self):
        self.ratios = Functions.financial_ratios(self.df)
        return self.ratios
    
    def clean_ratios(self):
        self.clean = Functions.clean_ratios_function(self.ratios)
        return self.clean