#
# Korrelationsfunktionen
#
# Autoren: Anna, Max (i3-Versicherung)
# Webseite: Data Science Training https://data-science.training
# Datum: 23.03.2023
#
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import chi2_contingency

# Korrelationsmatrix mit den linearen Korrelationskoeffizienten nach Pearson
#  (KNIME: "Linear Correlation")
def dst_correlation_matrix(df):
    # Nur numerische Attribute auswählen
    df1 = df.select_dtypes(include=[np.number])
    # Korrelationsmatrix berechnen
    corr_matrix = df1.corr(method='pearson')
    # Rückgabe
    return corr_matrix

# Heatmap mit Korrelationskoeffizienten anzeigen
#  (KNIME: "Linear Correlation")
def dst_correlation_heatmap(corr_matrix):
    # Labels
    labels = corr_matrix.keys()
    # Größe der Grafik festlegen
    plt.figure(figsize=(8,6))
    # Korrelationsmaxtrix mit Farbpalette "Blues" anzeigen
    plt.matshow(corr_matrix, cmap='Blues', fignum=1, aspect='auto')
    # Ticks mit Beschriftungen, X-Labels um 45 Grad gedreht 
    plt.xticks(range(len(labels)), labels, rotation=45)
    plt.yticks(range(len(labels)), labels)
    # Farbskala anzeigen
    plt.colorbar()
    # Grafik ausgeben
    plt.show()
    
# (Starke) Korrelationskoeffizienten berechnen
#  (KNIME: "Linear Correlation", "Correlation Filter")
def dst_correlation_measures_filtered(corr_matrix, treshold=0.75):
    # Absolute Werte
    corr_matrix = corr_matrix.abs()
    # Diagonale löschen (Nullwerte)
    #corr_matrix[corr_matrix == 1] = 0
    np.fill_diagonal(corr_matrix.values, 0)
    # Dreispaltige Tabelle erstellen: Attribut 1, Attribut 2, Koeffizient
    corr_measures = corr_matrix.unstack()
    # Duplikate entfernen
    corr_measures = corr_measures.drop_duplicates()
    # Filer anwenden
    corr_measures = corr_measures[corr_measures > treshold]
    # Absteigende Sortierung
    corr_measures = corr_measures.sort_values(ascending=False)
    # Kleine Matrix erstellen
    #corr_measures = corr_measures.unstack()
    # Rückgabe
    return corr_measures

# Cramer's V
#  für kategorische Variablen
#
# Quelle: https://towardsdatascience.com/the-search-for-categorical-correlation-a1cf7f1888c9
#
#
def dst_cramers_v(x, y):
    confusion_matrix = pd.crosstab(x, y)
    chi2 = chi2_contingency(confusion_matrix)[0]
    n = confusion_matrix.sum().sum()
    phi2 = chi2/n
    r,k = confusion_matrix.shape
    phi2corr = max(0, phi2-((k-1)*(r-1))/(n-1))
    rcorr = r-((r-1)**2)/(n-1)
    kcorr = k-((k-1)**2)/(n-1)
    return np.sqrt(phi2corr/min((kcorr-1),(rcorr-1)))

# Korrelationsmatrix mit Cramer's V
#  (KNIME: "Linear Correlation")
def dst_categorical_correlation_matrix(df):
    # Nur kategorische Attribute auswählen
    df1 = df.select_dtypes(include=['string', 'category'])
    # Fehlende Werte behandeln
    df1 = df1.astype('string').fillna('<XXX>')
    # https://stackoverflow.com/questions/52741236/how-to-calculate-p-values-for-pairwise-correlation-of-columns-in-pandas
    df_corr = pd.DataFrame() # Correlation matrix
    for x in df1.columns:
        for y in df1.columns:
            corr = dst_cramers_v(df1[x], df1[y])
            df_corr.loc[x,y] = corr
    # Rückgabe
    return df_corr