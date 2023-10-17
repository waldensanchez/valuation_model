import joblib
import xgboost
import numpy as np
import pandas as pd
import statsmodels.api as sm
from sklearn.svm import SVC
from scipy.optimize import minimize
from sklearn.metrics import f1_score
import matplotlib.pyplot as plt
from sklearn.ensemble import BaggingClassifier
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.naive_bayes import CategoricalNB
from sklearn.ensemble import StackingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import roc_auc_score, jaccard_score
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.linear_model import SGDClassifier, LogisticRegression, RidgeClassifierCV

def split(df: pd.DataFrame):
    X_train, X_test, y_train, y_test = train_test_split(df.drop(['Return'], axis = 1),df['Return'], test_size=0.2, random_state=42)
    return X_train, X_test, y_train, y_test

def Logistic(X_train: pd.DataFrame, X_test: pd.DataFrame, y_train: pd.DataFrame, y_test: pd.DataFrame):
    # Create a Logistic Regression model
    model = LogisticRegression()

    # Train the model on the training data
    model.fit(X_train, y_train)

    # Make predictions on the test data
    y_pred = model.predict(X_test)

    # Evaluate the model
    accuracy = accuracy_score(y_test, y_pred)
    confusion = confusion_matrix(y_test, y_pred)
    report = classification_report(y_test, y_pred)

    # Print the results
    print("Accuracy:", accuracy)
    print("Confusion Matrix:\n", confusion)
    print("Classification Report:\n",report)

def OLS_Reg(df: pd.DataFrame):
    X_sm = df.drop(['Return'], axis = 1)
    y_sm = df['Return']
    X = sm.add_constant(X_sm)
    model = sm.OLS(y_sm, X) 
    result = model.fit()
    summary = result.summary()
    return summary

def std_scaler(X_train: pd.DataFrame, X_test: pd.DataFrame):
    scaler = StandardScaler()
    X_train_ss = scaler.fit_transform(X_train)
    X_test_ss = scaler.transform(X_test)
    return X_train_ss, X_test_ss

def minmax(X_train: pd.DataFrame, X_test: pd.DataFrame):
    mm_scaler = MinMaxScaler()
    X_train_mm = mm_scaler.fit_transform(X_train)
    X_test_mm = mm_scaler.transform(X_test)
    return X_train_mm, X_test_mm 

def std_models_individual(X_train_ss: pd.DataFrame, X_train_mm: pd.DataFrame, y_train: pd.DataFrame, X_test_ss: pd.DataFrame, X_test_mm: pd.DataFrame, y_test: pd.DataFrame):
    MLP = MLPClassifier(activation='logistic', max_iter=1000, random_state=0).fit(X_train_ss, y_train)
    SGD = SGDClassifier(penalty="l2", max_iter=1000).fit(X_train_ss, y_train)
    KNC_u = KNeighborsClassifier(weights='uniform').fit(X_train_ss, y_train)
    KNC_d = KNeighborsClassifier(weights='distance').fit(X_train_ss, y_train)
    CNB = CategoricalNB().fit(X_train_mm, y_train)
    DTC = DecisionTreeClassifier(random_state=0).fit(X_train_ss, y_train)
    RFC = RandomForestClassifier(random_state=0).fit(X_train_ss, y_train)
    LR = LogisticRegression(max_iter=1000, random_state=0).fit(X_train_ss, y_train)
    RC = RidgeClassifierCV().fit(X_train_ss, y_train)
    LDA = LinearDiscriminantAnalysis().fit(X_train_ss, y_train)
    GBC = GradientBoostingClassifier().fit(X_train_ss, y_train)
    SV = SVC().fit(X_train_ss, y_train)
    HGBC = HistGradientBoostingClassifier().fit(X_train_ss, y_train)
    XGB = xgboost.XGBClassifier().fit(X_train_ss, y_train)
    
    models = {'Multilayer Perceptron':[MLP,'ss'], 'Stochastic Gradient Descent':[SGD,'ss'],\
          'K-Nearest Neighbors (uniformly weighted)':[KNC_u,'ss'],\
          'K-Nearest Neighbors (weighted by distance)':[KNC_d,'ss'],\
          'Categorical Naive-Bayes':[CNB,'mm'], 'Decision Tree Classifier':[DTC,'ss'],\
          'Random Forest Classifier':[RFC,'ss'], 'Logistic Regression':[LR,'ss'],\
          'Ridge Classifier CV':[RC,'ss'], 'Linear Discriminant Analysis':[LDA,'ss'],\
          'Gradient Boosting Classifier':[GBC,'ss'], 'Support Vector Classifier':[SV,'ss'],\
          'HGBC':[HGBC,'ss'], 'XGBoost':[XGB,'ss']
         }

    # Show train and test accuracy for each model sorted in a descending order by the former
    test_accuracies = sorted(models.items(), key=lambda x: \
                            x[1][0].score(X_test_ss if x[1][1] == 'ss' else X_test_mm, y_test), reverse=True)

    for model_name, (model, scaling_type) in test_accuracies:
        print(f'-- {model_name} --')
        accuracy_train = np.round(100 * model.score(X_train_ss if scaling_type == 'ss' else X_train_mm, y_train), 2)
        accuracy_test = np.round(100 * model.score(X_test_ss if scaling_type == 'ss' else X_test_mm, y_test), 2)
        print(f'Accuracy (train): {accuracy_train}%')
        print(f'Accuracy (test): {accuracy_test}%')

    return MLP, SGD, KNC_u, KNC_d, CNB, DTC, RFC, LR, RC, LDA, GBC, SV, HGBC, XGB

def models_individual(X_train: pd.DataFrame, y_train: pd.DataFrame, X_test: pd.DataFrame, y_test: pd.DataFrame):
    MLP = MLPClassifier(activation='logistic', max_iter=1000, random_state=0).fit(X_train, y_train)
    SGD = SGDClassifier(penalty="l2", max_iter=1000).fit(X_train, y_train)
    KNC_u = KNeighborsClassifier(weights='uniform').fit(X_train, y_train)
    KNC_d = KNeighborsClassifier(weights='distance').fit(X_train, y_train)
    CNB = CategoricalNB().fit(X_train, y_train)
    DTC = DecisionTreeClassifier(random_state=0).fit(X_train, y_train)
    RFC = RandomForestClassifier(random_state=0).fit(X_train, y_train)
    LR = LogisticRegression(max_iter=1000, random_state=0).fit(X_train, y_train)
    RC = RidgeClassifierCV().fit(X_train, y_train)
    LDA = LinearDiscriminantAnalysis().fit(X_train, y_train)
    GBC = GradientBoostingClassifier().fit(X_train, y_train)
    SV = SVC().fit(X_train, y_train)
    HGBC = HistGradientBoostingClassifier().fit(X_train, y_train)
    XGB = xgboost.XGBClassifier().fit(X_train, y_train)

    # Name models and specs
    models = {'Multilayer Perceptron':MLP, 'Stochastic Gradient Descent':SGD,\
            'K-Nearest Neighbors (uniformly weighted)':KNC_u,\
            'K-Nearest Neighbors (weighted by distance)':KNC_d,\
            'Decision Tree Classifier':DTC,\
            'Random Forest Classifier':RFC, 'Logistic Regression':LR,\
            'Ridge Classifier CV':RC, 'Linear Discriminant Analysis':LDA,\
            'Gradient Boosting Classifier':GBC, 'Support Vector Classifier':SV,\
            'HGBC':HGBC, 'XGBoost':XGB
            }
    # Crear diccionarios para almacenar los resultados de train y test
    train_accuracies = {}
    test_accuracies = {}

    # Calcular el accuracy para cada modelo en los conjuntos de train y test
    for model_name, model in models.items():
        y_train_pred = model.predict(X_train)
        y_test_pred = model.predict(X_test)
        
        train_accuracy = accuracy_score(y_train, y_train_pred)
        test_accuracy = accuracy_score(y_test, y_test_pred)
        
        train_accuracies[model_name] = train_accuracy
        test_accuracies[model_name] = test_accuracy

    # Ordenar los modelos por accuracy en el conjunto de prueba en orden descendente
    sorted_models_test = {k: v for k, v in sorted(test_accuracies.items(), key=lambda item: item[1], reverse=True)}

    # Imprimir los resultados ordenados
    for model_name, test_accuracy in sorted_models_test.items():
        train_accuracy = train_accuracies[model_name]
        print(f'Modelo: {model_name}')
        print(f'Accuracy en Train: {train_accuracy:.4f}')
        print(f'Accuracy en Test: {test_accuracy:.4f}')
        print('-' * 40)

    return MLP, SGD, KNC_u, KNC_d, CNB, DTC, RFC, LR, RC, LDA, GBC, SV, HGBC, XGB

def stacking_auto(X_train: pd.DataFrame, y_train: pd.DataFrame, X_test: pd.DataFrame, y_test:pd.DataFrame):
    #SKLEARN STACKINGCLASSIFIER
    base_models = [
        ('MLP', MLPClassifier(activation='logistic', max_iter=1000, random_state=0)),
        ('SGD', SGDClassifier(penalty="l2", max_iter=1000)),
        ('KNC_u', KNeighborsClassifier(weights='uniform')),
        ('KNC_d', KNeighborsClassifier(weights='distance')),
        ('CNB', CategoricalNB()),
        ('DTC', DecisionTreeClassifier(random_state=0)),
        ('RFC', RandomForestClassifier(random_state=0)),
        ('LR', LogisticRegression(max_iter=10000, random_state=0)),
        ('RC', RidgeClassifierCV()),
        ('LDA', LinearDiscriminantAnalysis()),
        ('GBC', GradientBoostingClassifier()),
        ('SV', SVC()),
        ('HGBC', HistGradientBoostingClassifier())
    ]

    # Create a StackingClassifier
    stacked_model = StackingClassifier(estimators=base_models)

    # Fit the StackingClassifier on the training data
    stacked_model.fit(X_train, y_train)

    # Evaluate the stacked model
    accuracy_train = np.round(100 * stacked_model.score(X_train, y_train), 2)
    accuracy_test = np.round(100 * stacked_model.score(X_test, y_test), 2)

    print(f'Accuracy (train): {accuracy_train}%')
    print(f'Accuracy (test): {accuracy_test}%')

    return stacked_model

def stacking_manual(MLP, SGD, KNC_u, KNC_d, DTC, RFC, LR, RC, LDA, GBC, SV, HGBC, XGB, X_train: pd.DataFrame, y_train: pd.DataFrame, X_test: pd.DataFrame, y_test: pd.DataFrame):
    # 1. Realizar predicciones de los modelos base en los datos de entrenamiento y prueba
    models = [MLP, SGD, KNC_u, KNC_d, DTC, RFC, LR, RC, LDA, GBC, SV, HGBC, XGB]
    train_predictions = []
    test_predictions = []

    for model in models:
        train_pred = model.predict(X_train)
        test_pred = model.predict(X_test)
        train_predictions.append(train_pred)
        test_predictions.append(test_pred)

    # 2. Apilar las predicciones de los modelos base en un nuevo conjunto de datos
    stacked_train = np.column_stack(train_predictions)
    stacked_test = np.column_stack(test_predictions)

    # 3. Entrenar un modelo final (meta-modelo)
    meta_model = LogisticRegression(max_iter=1000, random_state=0)
    meta_model.fit(stacked_train, y_train)

    # Calcular el accuracy en los datos de entrenamiento y prueba
    train_accuracy = accuracy_score(y_train, meta_model.predict(stacked_train))
    test_accuracy = accuracy_score(y_test, meta_model.predict(stacked_test))

    print("Accuracy en datos de entrenamiento:", train_accuracy)
    print("Accuracy en datos de prueba:", test_accuracy)

    return meta_model

def full_variables_model(df: pd.DataFrame):
    X_train, X_test, y_train, y_test = train_test_split(df.drop(['Stock','fiscalDateEnding','Return'], axis = 1), df['Return'], test_size=0.2, random_state=42)

    MLP = MLPClassifier(activation='logistic', max_iter=1000, random_state=0).fit(X_train, y_train)
    SGD = SGDClassifier(penalty="l2", max_iter=1000).fit(X_train, y_train)
    KNC_u = KNeighborsClassifier(weights='uniform').fit(X_train, y_train)
    KNC_d = KNeighborsClassifier(weights='distance').fit(X_train, y_train)
    DTC = DecisionTreeClassifier(random_state=0).fit(X_train, y_train)
    RFC = RandomForestClassifier(random_state=0).fit(X_train, y_train)
    LR = LogisticRegression(max_iter=1000, random_state=0).fit(X_train, y_train)
    RC = RidgeClassifierCV().fit(X_train, y_train)
    LDA = LinearDiscriminantAnalysis().fit(X_train, y_train)
    GBC = GradientBoostingClassifier().fit(X_train, y_train)
    SV = SVC().fit(X_train, y_train)
    HGBC = HistGradientBoostingClassifier().fit(X_train, y_train)
    XGB = xgboost.XGBClassifier().fit(X_train, y_train)

    # Name models and specs
    models = {'Multilayer Perceptron':MLP, 'Stochastic Gradient Descent':SGD,\
            'K-Nearest Neighbors (uniformly weighted)':KNC_u,\
            'K-Nearest Neighbors (weighted by distance)':KNC_d,\
            'Decision Tree Classifier':DTC,\
            'Random Forest Classifier':RFC, 'Logistic Regression':LR,\
            'Ridge Classifier CV':RC, 'Linear Discriminant Analysis':LDA,\
            'Gradient Boosting Classifier':GBC, 'Support Vector Classifier':SV,\
            'HGBC':HGBC, 'XGBoost':XGB
            }

    # Crear diccionarios para almacenar los resultados de train y test
    train_accuracies = {}
    test_accuracies = {}

    # Calcular el accuracy para cada modelo en los conjuntos de train y test
    for model_name, model in models.items():
        y_train_pred = model.predict(X_train)
        y_test_pred = model.predict(X_test)
        
        train_accuracy = accuracy_score(y_train, y_train_pred)
        test_accuracy = accuracy_score(y_test, y_test_pred)
        
        train_accuracies[model_name] = train_accuracy
        test_accuracies[model_name] = test_accuracy

    # Ordenar los modelos por accuracy en el conjunto de prueba en orden descendente
    sorted_models_test = {k: v for k, v in sorted(test_accuracies.items(), key=lambda item: item[1], reverse=True)}

    # Imprimir los resultados ordenados
    for model_name, test_accuracy in sorted_models_test.items():
        train_accuracy = train_accuracies[model_name]
        print(f'Modelo: {model_name}')
        print(f'Accuracy en Train: {train_accuracy:.4f}')
        print(f'Accuracy en Test: {test_accuracy:.4f}')
        print('-' * 40)
    
    return X_train, X_test, y_train, y_test, GBC

def GBCWithBagging(X_train: pd.DataFrame, y_train: pd.DataFrame, X_test: pd.DataFrame, y_test: pd.DataFrame):
    GBC_bag = BaggingClassifier(GradientBoostingClassifier()).fit(X_train, y_train)
    y_pred_bag = GBC_bag.predict(X_test)
    print(classification_report(y_test, y_pred_bag))
    return GBC_bag

def ConfusionPred(y_test: pd.DataFrame, y_pred: pd.DataFrame):
    ConfusionMatrixDisplay.from_predictions(y_test, y_pred, cmap=plt.cm.Blues)
    plt.show()

def ConfusionTrain(model, X_train: pd.DataFrame, y_train: pd.DataFrame):
    ConfusionMatrixDisplay.from_estimator(model, X_train, y_train, cmap=plt.cm.Blues)
    plt.show()

def ClassificationRep(model, X: pd.DataFrame, y:pd.DataFrame):
    y_pred = model.predict(X)
    print(classification_report(y, y_pred))

def SaveModel(model, name: str):
    joblib.dump(model, 'GBC_bagging_model.pkl')