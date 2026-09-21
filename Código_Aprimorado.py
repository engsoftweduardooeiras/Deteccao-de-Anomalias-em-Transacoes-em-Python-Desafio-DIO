# 1. IMPORTAÇÃO DAS BIBLIOTECAS NECESSÁRIAS
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.datasets import make_classification
from sklearn.ensemble import RandomForestClassifier
# Adicionado roc_auc_score para cálculo da métrica de área sob a curva
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score

# Certifique-se de instalar: pip install imbalanced-learn
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler

# 2. CRIAÇÃO DE UM CONJUNTO DE DADOS DESBALANCEADO (SIMULAÇÃO)
# Criando um dataset onde 95% pertence à classe 0 e apenas 5% à classe 1
X_raw, y_raw = make_classification(
    n_samples=10000, n_features=10, n_clusters_per_class=1,
    weights=[0.95, 0.05], flip_y=0, random_state=42
)

# Divisão fundamental: SEMPRE separe os dados de teste antes de balancear
X_train, X_test, y_train, y_test = train_test_split(
    X_raw, y_raw, test_size=0.3, random_state=42, stratify=y_raw
)

print(f"Distribuição original no treino: {np.bincount(y_train)}")

# ESTRATÉGIA A: OVERSAMPLING COM SMOTE (Gera dados sintéticos da classe rara)
smote = SMOTE(random_state=42)
X_train_smote, y_train_smote = smote.fit_resample(X_train, y_train)

model_smote = RandomForestClassifier(random_state=42)
model_smote.fit(X_train_smote, y_train_smote)
y_pred_smote = model_smote.predict(X_test)
# Captura as probabilidades para o cálculo do ROC AUC
y_proba_smote = model_smote.predict_proba(X_test)[:, 1]

# ESTRATÉGIA B: UNDERSAMPLING ALEATÓRIO (Reduz a classe majoritária)
rus = RandomUnderSampler(random_state=42)
X_train_rus, y_train_rus = rus.fit_resample(X_train, y_train)

model_rus = RandomForestClassifier(random_state=42)
model_rus.fit(X_train_rus, y_train_rus)
y_pred_rus = model_rus.predict(X_test)
# Captura as probabilidades para o cálculo do ROC AUC
y_proba_rus = model_rus.predict_proba(X_test)[:, 1]

# ESTRATÉGIA C: AJUSTE DE PESOS INTERNOS (Class Weight - Sem alterar os dados)
model_weighted = RandomForestClassifier(class_weight='balanced', random_state=42)
model_weighted.fit(X_train, y_train)
y_pred_weighted = model_weighted.predict(X_test)
# Captura as probabilidades para o cálculo do ROC AUC
y_proba_weighted = model_weighted.predict_proba(X_test)[:, 1]

# 3. EXIBIÇÃO DOS RESULTADOS (FOCO NAS MÉTRICAS CORRETAS)
def imprimir_metricas_completas(titulo, y_verdadeiro, y_predito, y_probabilidade):
    """Função auxiliar para organizar e exibir as novas métricas de forma limpa."""
    print(f"\n" + "="*50)
    print(f" {titulo} ")
    print("="*50)
    
    # 1. Relatório de Classificação Tradicional (Precision, Recall, F1-Score)
    print("\n[Relatório de Classificação]")
    print(classification_report(y_verdadeiro, y_predito))
    
    # 2. Matriz de Confusão detalhada
    print("[Matriz de Confusão]")
    cm = confusion_matrix(y_verdadeiro, y_predito)
    print(f"Verdadeiros Negativos (Classe 0): {cm[0][0]} | Falsos Positivos (Erros classe 0): {cm[0][1]}")
    print(f"Falsos Negativos (Erros classe 1): {cm[1][0]} | Verdadeiros Positivos (Classe 1): {cm[1][1]}")
    
    # 3. Métrica ROC AUC (Excelente para avaliar separação de classes)
    roc_auc = roc_auc_score(y_verdadeiro, y_probabilidade)
    print(f"\nROC AUC Score: {roc_auc:.4f}")

# Chamada das funções com os novos elementos incluídos
imprimir_metricas_completas("RESULTADO COM SMOTE", y_test, y_pred_smote, y_proba_smote)
imprimir_metricas_completas("RESULTADO COM UNDERSAMPLING", y_test, y_pred_rus, y_proba_rus)
imprimir_metricas_completas("RESULTADO COM CLASS WEIGHT BALANCED", y_test, y_pred_weighted, y_proba_weighted)
