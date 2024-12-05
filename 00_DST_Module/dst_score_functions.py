#
# Gütemaße für die binäre Klassifikation
#  0: negative Klasse (N)
#  1: positive Klasse (P)
#
# Autoren: Anna, Max (i3-Versicherung)
# Webseite: Data Science Training https://data-science.training
# Datum: 23.03.2023
#
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, roc_curve, auc, roc_auc_score, log_loss 
import locale
locale.setlocale(locale.LC_ALL, 'de')

# Genauigkeit (Accuracy)
def dst_accuracy(TP, TN, FP, FN):
    accuracy = 0
    denominator = TP + TN + FP + FN
    if denominator > 0:
        accuracy = (TP + TN) / denominator
    return accuracy

# Spezifität (Specificity)
def dst_specificity(TN, FP):
    specificity = 0
    denominator = TN + FP
    if denominator > 0:
        specificity = TN / denominator
    return specificity

# Sensitivität (Sensitivity)
def dst_sensitivity(TP, FN):
    sensitivity = 0
    denominator = TP + FN
    if denominator > 0:
        sensitivity = TP / denominator
    return sensitivity

# Präzision (Precision)
def dst_precision(TP, FP):
    precision = 0
    denominator = TP + FP
    if denominator > 0:
        precision = TP / denominator
    return precision

# Recall = Sensitivity
def dst_recall(TP, FN):
    return dst_sensitivity(TP, FN)

# Hilfsfunktion: Harmonisches Mittel
def dst_harmonic_mean(a, b):
    hm = 0
    denominator = a + b
    if denominator > 0:
        hm = 2 * (a * b) / denominator
    return hm

# F-Maß (F Score)
def dst_f_score(TP, FP, FN):
    return dst_harmonic_mean(dst_precision(TP, FP), dst_recall(TP, FN))

# Prognosen aus den Klassen-Wahrscheinlichkeiten mittels Schwellenwert berechnen
def dst_predictions(y_prob, threshold=0.5):
    m = len(y_prob[:,1])
    y_pred = np.zeros(m)
    for k in range(0, m):
        if y_prob[k][1] > threshold:
            y_pred[k] = 1
    return y_pred


# Berechnung der wichtigsten Gütemaße zur binären Klassifikation
# mit Hilfe des Modells, der (Trainings-)Daten (X, y) und der Kreuzvalidierung (Cross Validation cv)
# hauptsächlich aus der Konfusionsmatrix (Confusion Matrix)
# für einen gegebenen Schwellenwert (der normalerweise bei 0,5 liegt)
# in Phase 4 (Modeling) der Datenanalyse nach CRISP-DM
def dst_scores_threshold(model, X, y, cv, threshold=0.5):

    # Leere Konfusionsmatrix erstellen
    cmatrix = np.array([[0, 0],[0, 0]])

    # Dimension der Kreuzvalidierung
    n = cv.n_splits
    
    # Initialisierung der Gütemaße    
    scores = { 'accuracy': np.zeros(n), 'specificity': np.zeros(n), 'sensitivity': np.zeros(n), \
               'precision': np.zeros(n), 'recall': np.zeros(n), 'f_score': np.zeros(n), \
               'aurc': np.zeros(n), 'log_loss': np.zeros(n) }
    
    # Schleife über alle Validierungen der Kreuzvalidierung
    index = 0
    for train_index, test_index in cv.split(X, y):
    
        # Bilde die Trainings- und Validierungsmengen
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]
    
        # Modell mit den Trainigsdaten trainieren
        model.fit(X_train, y_train)
        
        # Klassen-Wahrscheinlichkeiten berechnen
        y_prob = model.predict_proba(X_test)
        
        # Prognosen aus den Klassen-Wahrscheinlichkeiten mittels Schwellenwert berechnen
        y_pred = dst_predictions(y_prob, threshold)
        
        # Einzelne Konfusionsmatrix berechnen
        cm = confusion_matrix(y_test, y_pred)
        
        # Exraktion der vier Werte der Konfusionsmatrix
        #  Für die Indizes einer binären Klassifikation siehe:
        #  https://scikit-learn.org/stable/modules/generated/sklearn.metrics.confusion_matrix.html
        TN = cm[0][0]
        FP = cm[0][1]
        FN = cm[1][0]
        TP = cm[1][1]

        # Werte der Gesamt-Konfusionsmatrix aktualisieren
        cmatrix[0][0] += TN
        cmatrix[0][1] += FP
        cmatrix[1][0] += FN
        cmatrix[1][1] += TP
        
        # Werte der Gütemaße-Arrays aktualisieren - Teil 1 (aus Konfusionsmatrix)
        scores['accuracy']   [index] = dst_accuracy(TP, TN, FP, FN)
        scores['specificity'][index] = dst_specificity(TN, FP)
        scores['sensitivity'][index] = dst_sensitivity(TP, FN)
        scores['precision']  [index] = dst_precision(TP, FP)
        scores['recall']     [index] = dst_recall(TP, FN)
        scores['f_score']    [index] = dst_f_score(TP, FP, FN)
        # Werte der Gütemaße-Arrays aktualisieren - Teil 2 (spezielle Funktionen)
        #  y_prob[:,1] sind die Wahrscheinlichkeiten der positiven Klasse
        scores['aurc']       [index] = roc_auc_score(y_test, y_prob[:,1])
        scores['log_loss']   [index] = log_loss(y_test, y_prob)
        
        # Index inkrementieren
        index += 1
    
    # Bestimmung der Mittelwerte der Gütemaße
    accuracy    = np.mean(scores['accuracy'])
    specificity = np.mean(scores['specificity'])
    sensitivity = np.mean(scores['sensitivity'])
    precision   = np.mean(scores['precision'])
    recall      = np.mean(scores['recall'])
    f_score     = np.mean(scores['f_score'])
    aurc        = np.mean(scores['aurc'])
    logloss     = np.mean(scores['log_loss'])
    
    # Ergebnis zurückgeben    
    return {'TN': cmatrix[0][0], 'FP': cmatrix[0][1], 'FN': cmatrix[1][0], 'TP': cmatrix[1][1], \
            'Accuracy': accuracy, 'Specificity': specificity, 'Sensitivity': sensitivity, \
            'Precision': precision, 'Recall': recall, 'F-Score': f_score, \
            'AURC': aurc, 'LogLoss': logloss}


# Berechnung der wichtigsten Gütemaße zur binären Klassifikation
# mit Hilfe des Modells, der (Trainings-)Daten (X, y) und der Kreuzvalidierung (Cross Validation cv)
# hauptsächlich aus der Konfusionsmatrix (Confusion Matrix)
# in Phase 4 (Modeling) der Datenanalyse nach CRISP-DM
def dst_scores(model, X, y, cv):
    return dst_scores_threshold(model, X, y, cv)
    
    
# Ausgabe der Receiver Operation Characteristic (ROC)
# mit Hilfe des Modells, der (Trainings-)Daten (X, y) und der Kreuzvalidierung (Cross Validation cv)
# in Phase 4 (Modeling) der Datenanalyse nach CRISP-DM
#
# Quelle: https://scikit-learn.org/stable/auto_examples/model_selection/plot_roc_crossval.html
#
def dst_roc(model, X, y, cv):
    
    # Initialisierungen
    tprs = []
    aucs = []
    mean_fpr = np.linspace(0, 1, 100)
    fig, ax = plt.subplots()
    
    # Schleife über alle Validierungsmengen
    for train_index, test_index in cv.split(X, y):
        
        # Bilde die Trainings- und Validierungsmengen
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]
    
        # Modell mit Trainigsdaten trainieren
        model.fit(X_train, y_train)
    
        # False Positive Rate (FPR) und True Positive Rate (TPR) aus Validierungsdaten berechnen
        y_prob = model.predict_proba(X_test)
        y_pred = y_prob[:,1]
        fpr, tpr, threshold = roc_curve(y_test, y_pred)
    
        # ROC zeichnen (Dünne Kurven)
        ax.plot(fpr, tpr, color='blue', alpha=0.3, lw=1)
    
        # Linreare Interpolation
        interp_tpr = np.interp(mean_fpr, fpr, tpr)
        interp_tpr[0] = 0.0
        tprs.append(interp_tpr)
        aucs.append(auc(fpr, tpr))
        
    # Mittelwert (Dicke Kurve)
    mean_tpr = np.mean(tprs, axis=0)
    mean_tpr[-1] = 1.0
    mean_auc = 100 * auc(mean_fpr, mean_tpr)
    std_auc  = 100 * np.std(aucs)
    ax.plot(mean_fpr, mean_tpr, color='blue', lw=2, alpha=0.8, label='ROC [AURC %0.2f $\pm$ %0.2f]' % (mean_auc, std_auc))
    
    # Korridor (Standardabweichungen)
    std_tpr = np.std(tprs, axis=0)
    tprs_upper = np.minimum(mean_tpr + 2 * std_tpr, 1)
    tprs_lower = np.maximum(mean_tpr - 2 * std_tpr, 0)
    ax.fill_between(mean_fpr, tprs_lower, tprs_upper, color='gray', alpha=0.2, label='$\pm$ 2 STD')

    # Zufall (Diagonale)
    ax.plot([0, 1], [0, 1], linestyle="--", color='red', lw=2, alpha=0.8, label='Zufall')

    # Beschriftungen
    ax.set(xlim=[-0.05, 1.05], ylim=[-0.05, 1.05], title='Receiver Operating Characteristic (ROC)', \
           xlabel='False Positive Rate (FPR)', ylabel='True Positive Rate (TPR)')
    ax.legend(loc='lower right')
    
    # Grafik ausgeben
    plt.show()
    
    
# Ausgabe der wichtigsten Gütemaße (zur binären Klassifikation)
def dst_print_scores(scores, critical='Precision'):
    
    # Confusion Matrix
    print()      
    print('Confusion Matrix')
    print()      
    print('* TN (True  Negatives) :', scores['TN'])
    print('* TP (True  Positives) :', scores['TP'])
    print('* FN (False Negatives) :', scores['FN'])
    print('* FP (False Positives) :', scores['FP'])
    
    # Gütemaße
    print()      
    print('Gütemaße')
    print()
    print('* Genauigkeit  (Accuracy)    : ', locale.format_string('%6.2f', 100 * scores['Accuracy'],  ), '%')
    print('* Spezifität   (Specificity) : ', locale.format_string('%6.2f', 100 * scores['Specificity']), '%')
    print('* Sensitivität (Sensitivity) : ', locale.format_string('%6.2f', 100 * scores['Sensitivity']), '%')
    print('* Präzision    (Precision)   : ', locale.format_string('%6.2f', 100 * scores['Precision']  ), '%')
    print('* Recall       (Recall)      : ', locale.format_string('%6.2f', 100 * scores['Recall']     ), '%')
    print('* F-Maß        (F Score)     : ', locale.format_string('%6.2f', 100 * scores['F-Score']    ), '%')
    print('* AURC         (AURC)        : ', locale.format_string('%6.2f', 100 * scores['AURC']       ), '%')
    print('* LogLoss      (LogLoss)     : ', locale.format_string('%7.3f',       scores['LogLoss']    )     )
    
    # Mittelwert zur vier spezielle Gütemaße
    print()
    print('Mittelwert')
    print()
    mean = 100 * (scores['Accuracy'] + scores[critical] + scores['F-Score'] + scores['AURC']) / 4
    print('* Mittelwert (Accuracy, F Score, AURC, %s): %s %%' % (critical, locale.format_string('%6.2f', mean))) 
    
# Anzeige der wichtigsten Gütemaße (zur binären Klassifikation)
def dst_display_scores(scores, critical='Precision'):
    
    # Confusion Matrix
    print()      
    print('Confusion Matrix')
    print()      
    print('|          | Prognose | Prognose |')
    print('| ---------| ---------| ---------|')
    print('|          | Positive | Negative |')
    print('| Positive | TP =', scores['TP'], '|', 'FN =', scores['FN'], '|')
    print('| Negative | FP =', scores['FP'], '|', 'TN =', scores['TN'], '|')
    
    # Gütemaße
    print()
    print('Gütemaße')
    print()
    print('| Gütemaß      | (Metrics)     | Wert      |')
    print('| -------------| --------------| ----------|')
    print('| Genauigkeit  | (Accuracy)    |', locale.format_string('%6.2f', 100 * scores['Accuracy']   ), '%  |')   
    print('| Spezifität   | (Specificity) |', locale.format_string('%6.2f', 100 * scores['Specificity']), '%  |')
    print('| Sensitivität | (Sensitivity) |', locale.format_string('%6.2f', 100 * scores['Sensitivity']), '%  |')
    print('| Präzision    | (Precision)   |', locale.format_string('%6.2f', 100 * scores['Precision']  ), '%  |')
    print('| Recall       | (Recall)      |', locale.format_string('%6.2f', 100 * scores['Recall']     ), '%  |')
    print('| F-Maß        | (F Score)     |', locale.format_string('%6.2f', 100 * scores['F-Score']    ), '%  |')
    print('| AURC         | (AURC)        |', locale.format_string('%6.2f', 100 * scores['AURC']       ), '%  |')
    print('| LogLoss      | (LogLoss)     |', locale.format_string('%7.3f',       scores['LogLoss']    ),  '  |')
    print()   
    # Mittelwert zur vier spezielle Gütemaße
    mean = 100 * (scores['Accuracy'] + scores[critical] + scores['F-Score'] + scores['AURC']) / 4
    print('* Mittelwert (Accuracy, F Score, AURC, %s): %s %%' % (critical, locale.format_string('%6.2f', mean)))
    
# Berechnung der wichtigsten Gütemaße zur binären Klassifikation
# mit Hilfe des trainierten Modells und der Testdaten (X_test, y_test)
# hauptsächlich aus der Konfusionsmatrix (Confusion Matrix)
# in Phase 5 (Evaluation) der Datenanalyse nach CRISP-DM
def dst_scores_trained_model(model, X_test, y_test):
    return dst_scores_trained_model_threshold(model, X_test, y_test)

# Berechnung der wichtigsten Gütemaße zur binären Klassifikation
# mit Hilfe des trainierten Modells und der Testdaten (X_test, y_test)
# hauptsächlich aus der Konfusionsmatrix (Confusion Matrix)
# für einen gegebenen Schwellenwert (der normalerweise bei 0,5 liegt)
# in Phase 5 (Evaluation) der Datenanalyse nach CRISP-DM
def dst_scores_trained_model_threshold(model, X_test, y_test, threshold=0.5):

    # Klassen-Wahrscheinlichkeiten für Testdaten berechnen
    y_prob = model.predict_proba(X_test)

    # Prognosen aus den Klassen-Wahrscheinlichkeiten mittels Schwellenwert berechnen
    y_pred = dst_predictions(y_prob, threshold)

    # Berechnung und Rückgabe der Gütemaße
    return dst_scores_predictions(y_test, y_prob, y_pred)

# Berechnung der wichtigsten Gütemaße zur binären Klassifikation
# mit Hilfe der der Lösung (y_test), der Klassen-Wahrscheinlichkeiten (y_prob) und der Prognosen (y_pred) 
# hauptsächlich aus der Konfusionsmatrix (Confusion Matrix)
# in Phase 4 (Modeling) bzw. 5 (Evaluation) der Datenanalyse nach CRISP-DM
def dst_scores_predictions(y_test, y_prob, y_pred):

    # Berechnung der Konfusionsmatrix
    conf_matrix = confusion_matrix(y_test, y_pred)

    # Exraktion der vier Werte der Konfusionsmatrix
    #  Für die Indizes einer binären Klassifikation siehe:
    #  https://scikit-learn.org/stable/modules/generated/sklearn.metrics.confusion_matrix.html
    TN = conf_matrix[0][0]
    FN = conf_matrix[1][0]
    FP = conf_matrix[0][1]
    TP = conf_matrix[1][1]
    
    # Bestimmung der Gütemaße aus den Werten der Konfusionsmatrix 
    accuracy    = dst_accuracy(TP, TN, FP, FN)
    specificity = dst_specificity(TN, FP)
    sensitivity = dst_sensitivity(TP, FN)
    precision   = dst_precision(TP, FP)
    recall      = dst_recall(TP, FN)
    f_score     = dst_f_score(TP, FP, FN)
    # Bestimmung der Gütemaße mit speziellen Funktionen
    aurc        = roc_auc_score(y_test, y_prob[:,1])
    logloss     = log_loss(y_test, y_prob)
  
    # Rückgabe der Gütemaße mittels Dictionary (Schlüssel-Wert-Paare)
    return {'TN': TN,'TP': TP, 'FN': FN, 'FP': FP, 'Accuracy': accuracy, \
            'Specificity': specificity, 'Sensitivity': sensitivity, \
            'Precision': precision, 'Recall': recall, 'F-Score': f_score, \
            'AURC': aurc, 'LogLoss': logloss}

# Ausgabe der Receiver Operation Characteristic (ROC)
# mit Hilfe des trainierten Modells und der Testdaten (X_test, y_test)
# in Phase 5 (Evaluation) der Datenanalyse nach CRISP-DM
def dst_roc_trained_model(model, X_test, y_test):

    # Klassen-Wahrscheinlichkeiten für Testdaten berechnen
    y_prob = model.predict_proba(X_test)
    
    # Prognosen = Wahrscheinlichkeiten der positiven Klasse
    y_pred = y_prob[:,1]
    
    # Diagramm erzeugen
    dst_roc_predictions(y_test, y_pred)
        
    
# Ausgabe der Receiver Operation Characteristic (ROC)
# mit Hilfe der Lösung (y_true) und der Prognosen der Modells (y_pred)
# in Phase 4 (Modeling) bzw. 5 (Evaluation) der Datenanalyse nach CRISP-DM
def dst_roc_predictions(y_true, y_pred):
    
    # Grafik initialisieren
    fig, ax = plt.subplots()
    
    # False Positive Rate (FPR) und True Positive Rate (TPR) berechnen
    fpr, tpr, threshold = roc_curve(y_true, y_pred)
    
    # AURC berechnen
    aurc = 100 * roc_auc_score(y_true, y_pred)
    
    # ROC zeichnen
    ax.plot(fpr, tpr, color='blue', alpha=0.8, lw=2, label='ROC [AURC %0.2f]' % (aurc))
    
    # Zufall (Diagonale)
    ax.plot([0, 1], [0, 1], linestyle='--', color='red', alpha=0.8, lw=2, label='Zufall')

    # Beschriftungen
    ax.set(xlim=[-0.05, 1.05], ylim=[-0.05, 1.05], title='Receiver Operating Characteristic (ROC)', \
           xlabel='False Positive Rate (FPR)', ylabel='True Positive Rate (TPR)')
    ax.legend(loc='lower right')
    
    # Grafik ausgeben
    plt.show()  