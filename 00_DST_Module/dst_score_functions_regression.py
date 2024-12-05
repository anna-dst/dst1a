#
# Gütemaße für die Regression
#
# Autoren: Anna, Max (i3-Versicherung)
# Webseite: Data Science Training https://data-science.training
# Datum: 23.03.2023
#
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score, root_mean_squared_error, mean_absolute_error, mean_absolute_percentage_error
from sklearn.model_selection import cross_validate

# Berechnung der wichtigsten Gütemaße zur Regression
# mit Hilfe des Modells, der (Trainings-)Daten (X, y) und der Kreuzvalidierung (Cross Validation cv)
# in Phase 4 (Modeling) der Datenanalyse nach CRISP-DM
def dst_scores_regression(model, X, y, cv):

    # Anzahl der Validierungen
    n = cv.n_splits
    
    # Initialisierung der Gütemaße zur Regression
    scores = { 'r2': np.zeros(n), 'rmse': np.zeros(n), 'mae': np.zeros(n), 'mape': np.zeros(n) }
    
    # Schleife über alle Validierungen der Kreuzvalidierung
    index = 0
    for train_index, test_index in cv.split(X, y):
    
        # Bilde die Trainings- und Validierungsmengen
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]
    
        # Modell mit den Trainigsdaten trainieren
        model.fit(X_train, y_train)
        
        # Prognose bestimmen
        y_pred = model.predict(X_test)

        # Gütemaße bestimmen
        scores['r2']  [index] = r2_score(                      y_test, y_pred)
        scores['rmse'][index] = root_mean_squared_error(       y_test, y_pred)
        scores['mae'] [index] = mean_absolute_error(           y_test, y_pred)
        scores['mape'][index] = mean_absolute_percentage_error(y_test, y_pred)

        index += 1
    
    # Bestimmung der Mittelwerte der Gütemaße
    r2   = np.mean(scores['r2'])
    rmse = np.mean(scores['rmse'])
    mae  = np.mean(scores['mae'])
    mape = np.mean(scores['mape'])
    
    # Ergebnis zurückgeben
    return {'r2': r2, 'rmse': rmse, 'mae': mae, 'mape': mape }


# Berechnung der wichtigsten Gütemaße zur Regression
# mit Hilfe des Modells, der (Trainings-)Daten (X, y) und der Kreuzvalidierung (Cross Validation cv)
# in Phase 4 (Modeling) der Datenanalyse nach CRISP-DM
def dst_scores_regression_v2(model, X, y, cv):

    # Anzahl der Validierungen
    n = cv.n_splits
    
    # Initialisierung der Gütemaße
    scores = { 'r2': np.zeros(n), 'rmse': np.zeros(n), 'mae': np.zeros(n), 'mape': np.zeros(n) }
    
    # Dictionary mit den Gütemaßen
    scoring = { 'r2': 'r2', 'rmse': 'neg_root_mean_squared_error', 'mae': 'neg_mean_absolute_error', \
                'mape' : 'neg_mean_absolute_percentage_error' }
    
    # Berechnung der Gütemaße mittels Kreuzvalidierung
    scores = cross_validate(model, X, y, cv=cv, scoring=scoring)
    
    # Bestimmung der Mittelwerte der Gütemaße
    r2   =   scores['test_r2'].mean()
    rmse =  -scores['test_rmse'].mean()
    mae  =  -scores['test_mae'].mean()
    mape =  -scores['test_mape'].mean()
 
    # Ergebnis zurückgeben
    return {'r2': r2, 'rmse': rmse, 'mae': mae, 'mape': mape }


# Berechnung der wichtigsten Gütemaße zur Regression
# mit Hilfe des Modells, der Testdaten (X, y)
# in Phase 5 (Evaluation) der Datenanalyse nach CRISP-DM
def dst_scores_regression_trained_model(model, X_test, y_test):

    # Initialisierung der Gütemaße
    scores = { 'r2': 0, 'rmse': 0, 'mae': 0, 'mape': 0 }
    
    # Prognose bestimmen
    y_pred = model.predict(X_test)

    # Gütemaße bestimmen
    scores['r2']   = r2_score(                      y_test, y_pred)
    scores['rmse'] = root_mean_squared_error(       y_test, y_pred)
    scores['mae']  = mean_absolute_error(           y_test, y_pred)
    scores['mape'] = mean_absolute_percentage_error(y_test, y_pred)

    # Ergebnis zurückgeben
    return scores


# Ausgabe der wichtigsten Gütemaße zur Regression
def dst_print_scores_regression(scores, decimals=4):
    print()
    print('Gütemaße')
    print()
    print('* Bestimmtheitsmaß (R^2)                : ', round(    scores['r2'],  decimals)     )
    print('* Root Mean Squared Error (RMSE)        : ', round(    scores['rmse'],decimals)     )
    print('* Mean Absolute Error (MAE)             : ', round(    scores['mae'], decimals)     )
    print('* Mean Absolute Percentage Error (MAPE) : ', round(100*scores['mape'],decimals), '%')

    
# Anzeige der wichtigsten Gütemaße zur Regression
def dst_display_scores_regression(scores, decimals=4):
    print()
    print('Gütemaße')
    print()
    print('| Gütemaß                        | ()     | Wert      |')
    print('| -------------------------------| -------| ----------|')
    print('| Bestimmtheitsmaß               | (R^2)  | ', round(    scores['r2'],  decimals),   '|')
    print('| Root Mean Squared Error        | (RMSE) | ', round(    scores['rmse'],decimals),   '|')
    print('| Mean Absolute Error            | (MAE)  | ', round(    scores['mae'], decimals),   '|')
    print('| Mean Absolute Percentage Error | (MAPE) | ', round(100*scores['mape'],decimals), '% |')