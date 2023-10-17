import DataFunctions
import ModelFunctions

class Companies():
    def __init__(self, df) -> None:
        self.df = df

    def get_ratios(self):
        self.ratios = DataFunctions.financial_ratios(self.df)
        return self.ratios
    
    def clean_ratios(self):
        self.clean = DataFunctions.clean_ratios_function(self.ratios)
        return self.clean

    def model_input(self):
        self.input = DataFunctions.final_step_cleansing(self.clean)
    
class Models():
    def __init__(self, df) -> None:
        self.df = df

    def load_all(self, all_variables):
        self.all_variables = all_variables

    def split_data(self):
        self.X_train, self.X_test, self.y_train, self.y_test = ModelFunctions.split(self.df)

    def Logistic_Regression(self):
        self.Logistic_Regression = ModelFunctions.Logistic(self.X_train, self.X_test, self.y_train, self.y_test)
    
    def OLS_Regression(self):
        self.OLS_Regression = ModelFunctions.OLS_Reg(self.df)
    
    def transformations(self):
        """ Returns Standarized train and test X. It also returns MinMax scaled train and test X. """
        self.X_train_ss, self.X_test_ss = ModelFunctions.std_scaler(self.X_train, self.X_test)
        self.X_train_mm, self.X_test_mm = ModelFunctions.minmax(self.X_train, self.X_test)

    def transformed_models_individual(self):
        """ Models with Scalers. """
        self.MLP_T, self.SGD_T, self.KNC_u_T, self.KNC_d_T, self.CNB_T, self.DTC_T, self.RFC_T, self.LR_T, self.RC_T, self.LDA_T, self.GBC_T, self.SV_T, self.HGBC_T, self.XGB_T = ModelFunctions.std_models_individual(self.X_train_ss, self.X_train_mm, self.y_train, self.X_test_ss, self.X_test_mm, self.y_test)

    def models_individual(self):
        """ Models with direct data. """
        self.MLP, self.SGD, self.KNC_u, self.KNC_d, self.CNB, self.DTC, self.RFC, self.LR, self.RC, self.LDA, self.GBC, self.SV, self.HGBC, self.XGB = ModelFunctions.models_individual(self.X_train, self.y_train, self.X_test, self.y_test)

    def Stacking_automatic(self):
        self.Stacking_automatic = ModelFunctions.stacking_auto(self.X_train, self.y_train, self.X_test, self.y_test)
    
    def Stacking_manual(self):
        self.Stacking_manual = ModelFunctions.stacking_manual(self.MLP, self.SGD, self.KNC_u, self.KNC_d, self.DTC, self.RFC, self.LR, self.RC, self.LDA, self.GBC, self.SV, self.HGBC, self.XGB, self.X_train, self.y_train, self.X_test, self.y_test)

    def All_Variables_Models(self):
        self.X_train_all, self.X_test_all, self.y_train_all, self.y_test_all, self.GBC_All = ModelFunctions.full_variables_model(self.all_variables)

    def GBC_Bagging(self, All_Vars: bool = True):
        if All_Vars:
            print('Bagging with all the variables')
            self.GBC_Bagging = ModelFunctions.GBCWithBagging(self.X_train_all, self.y_train_all , self.X_test_all , self.y_test_all)
        else:
            print('Bagging with selected variables')
            self.GBC_Bagging = ModelFunctions.GBCWithBagging(self.X_train, self.y_train , self.X_test , self.y_test)

    def ConfusionMatrix(self, model, training: bool = True):
        """ Model is the selected model to which create the confusion matrix """
        if training:
            print('Confusion Matrix with Training Data')
            ModelFunctions.ConfusionTrain(model, self.X_train_all, self.y_train_all)
        else:
            print('Confusion Matrix with Predicted Values')
            y_pred = model.predict(self.X_test_all)
            ModelFunctions.ConfusionPred(self.y_test_all, y_pred)

    def ClassificationReport(self, model, training: bool = True):
        if training:
            print('Classification Report with Training')
            ModelFunctions.ClassificationRep(model, self.X_train_all, self.y_train_all)
        else:
            print('Classification Report with Predicted Values')
            ModelFunctions.ClassificationRep(model, self.X_test_all, self.y_test_all)

    def save(self, model, name):
        ModelFunctions.SaveModel(model, name)
