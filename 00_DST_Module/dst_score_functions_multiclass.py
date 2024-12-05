#
# Gütemaße für die Mehrklassen-Klassifikation
#
# Autoren: Anna, Max (i3-Versicherung)
# Webseite: Data Science Training https://data-science.training
# Datum: 23.03.2023
#
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.metrics import accuracy_score, roc_auc_score, log_loss, precision_recall_fscore_support
from sklearn.model_selection import cross_validate, cross_val_predict, cross_val_score 

# Berechnung der wichtigsten Gütemaße zur Multiklassen-Klassifikation
# mit Hilfe des Modells, der (Trainings-)Daten (X, y) und der Kreuzvalidierung (Cross Validation cv)
# in Phase 4 (Modeling) der Datenanalyse nach CRISP-DM
def dst_scores_multi_simple(model, X, y, cv, labels=None):
    
    # Berechnung der Prognosewerte mittels Kreuzvalidierung
    y_pred = cross_val_predict(model, X, y, cv=cv)
    # Berechnung der Konfusionsmatrix
    conf_matrix = confusion_matrix(y, y_pred, labels=labels)
    
    # Dictionary mit den Gütemaßen
    scoring = {'accuracy': 'accuracy', 'aurc': 'roc_auc_ovr', 'log_loss': 'neg_log_loss'}
    # Berechnung der Gütemaße mittels Kreuzvalidierung
    scores = cross_validate(model, X, y, cv=cv, scoring=scoring)
    
    # Bestimmung der Mittelwerte der Gütemaße
    accuracy =  scores['test_accuracy'].mean()
    aurc     =  scores['test_aurc'].mean()
    logloss  = -scores['test_log_loss'].mean()
    
    # Rückgabe der Gütemaße mittels Dictionary (Schlüssel-Wert-Paare)
    return {'CM': conf_matrix, 'Accuracy': accuracy, 'AURC': aurc, 'LogLoss': logloss}


# Berechnung der wichtigsten Gütemaße zur Multiklassen-Klassifikation
# mit Hilfe des Modells, der (Trainings-)Daten (X, y) und der Kreuzvalidierung (Cross Validation cv)
# in Phase 4 (Modeling) der Datenanalyse nach CRISP-DM
def dst_scores_multi(model, X, y, cv, labels=None):
    
    # Dimensionen
    n = cv.n_splits # Anzahl der Validierungen
    m = len(labels) # Anzahl der Klassen
    
    # Initialisierung
    scores = { 'CM': np.zeros((m,m)), \
               'Precision': np.zeros(m), 'Recall': np.zeros(m), 'F-Score': np.zeros(m), \
               'Accuracy': 0, 'AURC': 0, 'LogLoss': 0, \
               'Macro-Precision': 0, 'Macro-Recall': 0, 'Macro-F-Score': 0 }
    
    # Schleife über alle Validierungen der Kreuzvalidierung
    index = 0
    for train_index, test_index in cv.split(X, y):
    
        # Bilde die Trainings- und Validierungsmengen
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]
    
        # Modell mit den Trainigsdaten trainieren
        model.fit(X_train, y_train)
        
        # Prognose und Wahrscheinlichkeiten bestimmen
        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)
        
        # Konfusionsmatrix berechnen
        conf_matrix = confusion_matrix(y_test, y_pred, labels=labels)
    
        # Gütemaße für jede einzelne Klasse bestimmen
        precision, recall, f_score, support = precision_recall_fscore_support(y_test, y_pred, labels=labels, \
                                                                              average=None, zero_division=0)
        
        # Weitere Gütemaße mit Hilfe der Scikit-Learn-Funktionen bestimmen
        accuracy = accuracy_score (y_test, y_pred)
        aurc     = roc_auc_score  (y_test, y_prob, multi_class='ovr')
        logloss  = log_loss       (y_test, y_prob, labels=labels)
        
        # Werte der Gütemaße aktualisieren
        scores['Accuracy'] += accuracy
        scores['AURC']     += aurc
        scores['LogLoss']  += logloss
        for k in range(0, m):        
            scores['Precision'][k] += precision[k]
            scores['Recall']   [k] += recall   [k]
            scores['F-Score']  [k] += f_score  [k]
            for i in range(0, m):
                scores['CM'][i][k] += conf_matrix[i][k]
        
        index += 1
           
    # Bestimmung der Mittelwerte der Gütemaße
    scores['Accuracy'] /= n
    scores['AURC']     /= n
    scores['LogLoss']  /= n
    for k in range(0, m):
        scores['Precision'][k] /= n
        scores['Recall']   [k] /= n
        scores['F-Score']  [k] /= n
        scores['Macro-Precision'] += scores['Precision'][k]
        scores['Macro-Recall']    += scores['Recall']   [k]
        scores['Macro-F-Score']   += scores['F-Score']  [k]
    
    scores['Macro-Precision'] /= m
    scores['Macro-Recall']    /= m
    scores['Macro-F-Score']   /= m
        
    # Ergebnis zurückgeben    
    return scores


# Berechnung der wichtigsten Gütemaße zur Multiklassen-Klassifikation
# mit Hilfe des trainierten Modells und der Testdaten (X_test, y_test)
# in Phase 5 (Evaluation) der Datenanalyse nach CRISP-DM
def dst_scores_multi_trained_model(model, X_test, y_test, labels=None):

    # Prognosen für Testdaten berechnen
    y_pred = model.predict(X_test)
    # Wahrscheinlichkeiten für Testdaten berechnen
    y_prob = model.predict_proba(X_test)
    # Konfusionsmatrix berechnen
    conf_matrix = confusion_matrix(y_test, y_pred, labels=labels)
    
    # Gütemaße für jede einzelne Klasse bestimmen
    precision, recall, f_score, support = precision_recall_fscore_support(y_test, y_pred, labels=labels, average=None)
    
    # Gütemaße mit Makro-Mittelwertbildung bestimmen
    m_precision, m_recall, m_f_score, m_support = precision_recall_fscore_support(y_test, y_pred, labels=labels, average='macro')
        
    # Weitere Gütemaße mit Hilfe der Scikit-Learn-Funktionen bestimmen
    accuracy = accuracy_score (y_test, y_pred)
    aurc     = roc_auc_score  (y_test, y_prob, multi_class='ovr')
    logloss  = log_loss       (y_test, y_prob, labels=labels)
    
    # Rückgabe der Gütemaße mittels Dictionary (Schlüssel-Wert-Paare)
    return {'CM': conf_matrix, 'Accuracy': accuracy, \
            'Precision': precision, 'Recall': recall, \
            'F-Score': f_score, 'AURC': aurc, 'LogLoss': logloss, \
            'Macro-Precision': m_precision, 'Macro-Recall': m_recall, \
            'Macro-F-Score': m_f_score}


# Ausgabe der Gütemaße zur Multiklassen-Klassifikation
def dst_print_scores(scores, decimals=4):
    print()
    print('Gütemaße')
    print()
    print('* Genauigkeit (Accuracy) : ', round(100 * scores['Accuracy'],      decimals), ' %')
    print('* Makro-F-Maß (F-Score)  : ', round(100 * scores['Macro-F-Score'], decimals), ' %')
    print('* AURC        (AURC)     : ', round(100 * scores['AURC'],          decimals), ' %')
    print('* LogLoss     (LogLoss)  : ', round(      scores['LogLoss'],       decimals)      )
    print()
    
# Anzeige der Gütemaße zur Multiklassen-Klassifikation
def dst_display_scores(scores, decimals=4):
    print()
    print('Gütemaße')
    print()
    print('| Gütemaß      | (Metrics)     | Wert      |')
    print('| -------------| --------------| ----------|')
    print('| Genauigkeit  | (Accuracy)    |', round(100 * scores['Accuracy'],      decimals), '% |')
    print('| Makro-F-Maß  | (F-Score)     |', round(100 * scores['Macro-F-Score'], decimals), '% |')
    print('| AURC         | (AURC)        |', round(100 * scores['AURC'],          decimals), '% |')
    print('| LogLoss      | (LogLoss)     |', round(      scores['LogLoss'],       decimals), '  |')
    print()

# Anzeige der Klassenabhängigen Gütemaße zur Multiklassen-Klassifikation
def dst_display_class_scores(scores, decimals=4, labels=None):
    precision = 0
    recall    = 0
    f_score   = 0
    print()
    print('Klassenabhängige Gütemaße')
    print()
    print('| Klasse | Präzision | Recall | F-Maß |')
    print('|--------|-----------|--------|-------|')
    index = 0
    for label in labels:
        precision += scores['Precision'][index]
        recall    += scores['Recall']   [index]
        f_score   += scores['F-Score']  [index]
        print('|', label, \
              '|', round(100*scores['Precision'][index],decimals), \
              '|', round(100*scores['Recall']   [index],decimals), \
              '|', round(100*scores['F-Score']  [index],decimals), \
              '|')
        index += 1
    precision /= index
    recall    /= index
    f_score   /= index   
    print('|--------|-----------|--------|-------|')    
    print('| Makro ', \
          '|', round(100*precision,decimals), \
          '|', round(100*recall,   decimals), \
          '|', round(100*f_score,  decimals), \
          '|')
    print()    

# Anzeige der Konfusionsmatrix zur Multiklassen-Klassifikation
def dst_display_confusion_matrix(confusion_matrix, labels=None):
    print()
    print('Confusion Matrix')
    print()
    fig, ax = plt.subplots(figsize=(8, 8))
    cmd = ConfusionMatrixDisplay(confusion_matrix, display_labels=labels)
    cmd.plot(cmap=plt.cm.Blues, ax=ax, values_format='.0f')
    plt.xlabel('Prognosen')
    plt.ylabel('Wahrheit')
    plt.show()
    print()
    
# Ausgabe der wichtigsten Gütemaße zur Multiklassen-Klassifikation
def dst_print_scores_multi(scores, decimals=4, labels=None):
    # Gütemaße
    dst_print_scores(scores, decimals)
    # Confusion Matrix
    dst_display_confusion_matrix(scores['CM'], labels=labels)

# Ausgabe der wichtigsten Gütemaße zur Multiklassen-Klassifikation
def dst_display_scores_multi(scores, decimals=4, labels=None):
    # Gütemaße
    dst_display_scores(scores, decimals)
    # Klassenabhängige Gütemaße
    dst_display_class_scores(scores, decimals, labels=labels)
    # Confusion Matrix
    dst_display_confusion_matrix(scores['CM'], labels=labels)