# Importar Bibliotecas 
import numpy as np
import pandas as pd
import yfinance as yf
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, GRU, Dropout
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt

# Baixar dados da ação AAPL
DIAS = 1
END_DATE = '2025-07-30'
START_DATE = '2020-01-01'
TICKER = 'AAPL'
SEQ_LEN = 60

dados_aapl = yf.download(TICKER, start=START_DATE, end=END_DATE)

# Verificar valores ausentes
# valores faltantes nas colunas são substituídos pela última observação válida anterior.
dados_aapl.ffill(inplace=True)

# Escalar os dados
# Normalização: Transforma os valores de fechamento para um intervalo de 0 a 1
escalador = MinMaxScaler(feature_range=(0, 1))
dados_escalados = escalador.fit_transform(dados_aapl['Close'].values.reshape(-1, 1))

# Criar sequências
# Previsões baseadas em contexto histórico fixo.
# Forma clássica de transformar dados sequenciais em problemas de aprendizado supervisionado: 
# Usar os últimos N valores como entrada para prever o (N+1)‑ésimo valor.
def criar_sequencias(dados, comprimento_seq):
    X, y = [], []
    for i in range(len(dados) - comprimento_seq):
        X.append(dados[i:i+comprimento_seq])
        y.append(dados[i+comprimento_seq])
    return np.array(X), np.array(y)

comprimento_seq = SEQ_LEN
X, y = criar_sequencias(dados_escalados, comprimento_seq)


# Dividir em conjuntos de treinamento e teste
divisao = int(0.8 * len(X))  # Primeiros 80% dos dados para treinamento.
X_treino, X_teste = X[:divisao], X[divisao:]
y_treino, y_teste = y[:divisao], y[divisao:]

# Redimensionar para entrada do modelo
X_treino = X_treino.reshape((X_treino.shape[0], X_treino.shape[1], 1))
X_teste = X_teste.reshape((X_teste.shape[0], X_teste.shape[1], 1))

# Construir modelo GRU com Keras.
modelo_gru = Sequential([
    GRU(50, return_sequences=True, input_shape=(comprimento_seq, 1)),
    Dropout(0.2),
    GRU(50, return_sequences=False),
    Dropout(0.2),
    Dense(25),
    Dense(1)
])
modelo_gru.compile(optimizer='adam', loss='mse')
modelo_gru.fit(X_treino, y_treino, epochs=10, batch_size=16, verbose=1)

# Fazer previsões
previsoes_gru = modelo_gru.predict(X_teste)
# Converter previsões para valores reais.
previsoes_gru = escalador.inverse_transform(previsoes_gru)
# Trazer os labels de teste para valores reais
y_teste_real = escalador.inverse_transform(y_teste.reshape(-1, 1))

# Prever preços futuros
def prever_futuro(modelo, X_teste, dias):
    entrada_previsao = X_teste[-1].reshape(1, comprimento_seq, 1)
    previsao = []
    for _ in range(dias):
        pred = modelo.predict(entrada_previsao)[0][0]
        previsao.append(pred)
        entrada_previsao = np.roll(entrada_previsao, -1, axis=1)
        entrada_previsao[0, -1, 0] = pred
    return escalador.inverse_transform(np.array(previsao).reshape(-1, 1))

previsao_gru = prever_futuro(modelo_gru, X_teste, dias = DIAS)

# Calcular métricas de avaliação
mse = mean_squared_error(y_teste_real, previsoes_gru)
r2 = r2_score(y_teste_real, previsoes_gru)
# Métricas de avaliação:
# Erro Quadrático Médio (MSE): Mede a diferença média entre os valores previstos e reais. Quanto menor o MSE, melhor o desempenho do modelo.
# Coeficiente de Determinação (R²): Mede a proporção da variação total dos valores reais que é explicada pelas previsões do modelo. Ideal se proximo de 1.

# Criar tabela de métricas de avaliação
tabela_avaliacao = pd.DataFrame({
    'Métrica': ['Erro Quadrático Médio (MSE)', 'R-quadrado (R²)'],
    'Valor': [f'{mse:.4f}', f'{r2:.4f}']
})

# Exibir a tabela
print("--------------------------------")
print("\nMétricas de Avaliação:")
print(tabela_avaliacao.to_string(index=False))

# Preparar dados para visualização e exibição
indice_data = dados_aapl.index[-len(y_teste):]

# Exibir a previsão
print("--------------------------------")
print(f"{indice_data[-1].strftime('%d/%m/%Y')} - Valor Real: ${float(dados_aapl['Close'].iloc[-1].iloc[0]):.2f}")
print("\nPrevisões:")
datas_previsao = pd.date_range(start=indice_data[-1], periods=11)[1:]
for i, (data, valor) in enumerate(zip(datas_previsao, previsao_gru), 1):
    print(f"{data.strftime('%d/%m/%Y')} - Previsão: ${valor[0]:.2f}")

# Visualização
plt.figure(figsize=(10, 6))
plt.plot(indice_data, y_teste_real, label='Preço de Fechamento Real', color='blue', linewidth=1.5)
plt.plot(indice_data, previsoes_gru, label='Previsões GRU', color='green', linestyle='-', linewidth=1)
datas_previsao = pd.date_range(start=indice_data[-1], periods=DIAS + 1)[1:]
plt.plot(datas_previsao, previsao_gru, label='Previsão GRU', color='red', linestyle='--', linewidth=1.5, alpha=0.9)
plt.axvline(x=indice_data[-1], color='gray', linestyle=':', linewidth=1.5, label='Início da Previsão')
plt.axvspan(indice_data[-1], datas_previsao[-1], color='skyblue', alpha=0.5, label='Período de Previsão')
plt.title("Previsões e Previsões de Unidades Recorrentes com Portas (GRU) vs Preço de Fechamento Real (Previsão de 10 Dias)", fontsize=12, pad=20)
plt.xlabel("Data", fontsize=12)
plt.ylabel("Preço de Fechamento", fontsize=12)
plt.grid(True, linestyle='--', alpha=0.6)
plt.legend(fontsize=10, loc='upper left')
plt.tight_layout()
plt.show()
