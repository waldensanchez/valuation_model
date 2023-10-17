import pandas as pd
import Functions

class Companies():
    def __init__(self, df) -> None:
        self.df = df

    def get_ratios(self):
        self.ratios = Functions.financial_ratios(self.df)
        return self.ratios