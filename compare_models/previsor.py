import numpy as np
import pandas as pd
import yfinance as yf
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, GRU, LSTM, SimpleRNN 
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt


class PrevisorBase:
    def __init__(self, modelo_nome, dias=10, seq_len=60, ticker='AAPL', start='2020-01-01', end='2025-04-10'):
        self.modelo_nome = modelo_nome
        self.dias = dias
        self.seq_len = seq_len
        self.ticker = ticker
        self.start = start
        self.end = end
        self.modelo = None

    def preparar_dados(self):
        print(f"Baixando dados de {self.ticker}...")
        self.dados = yf.download(self.ticker, start=self.start, end=self.end)
        self.dados.ffill(inplace=True)

        self.indice_data = self.dados.index
        self.escalador = MinMaxScaler(feature_range=(0, 1))
        dados_escalados = self.escalador.fit_transform(self.dados['Close'].values.reshape(-1, 1))

        X, y = self.criar_sequencias(dados_escalados, self.seq_len)
        divisao = int(0.8 * len(X))
        self.X_treino, self.X_teste = X[:divisao], X[divisao:]
        self.y_treino, self.y_teste = y[:divisao], y[divisao:]

        self.X_treino = self.X_treino.reshape((self.X_treino.shape[0], self.seq_len, 1))
        self.X_teste = self.X_teste.reshape((self.X_teste.shape[0], self.seq_len, 1))
        self.y_teste_real = self.escalador.inverse_transform(self.y_teste.reshape(-1, 1))
        self.indice_data = self.dados.index[-len(self.y_teste):]

    def criar_sequencias(self, dados, comprimento_seq):
        X, y = [], []
        for i in range(len(dados) - comprimento_seq):
            X.append(dados[i:i + comprimento_seq])
            y.append(dados[i + comprimento_seq])
        return np.array(X), np.array(y)

    def construir_modelo(self):
        model_dict = {
            'GRU': GRU,
            'LSTM': LSTM,
            'RNN': SimpleRNN
        }
        model = model_dict[self.modelo_nome]

        self.modelo = Sequential([
            model(50, return_sequences=True, input_shape=(self.seq_len, 1)),
            Dropout(0.2),
            model(50, return_sequences=False),
            Dropout(0.2),
            Dense(25),
            Dense(1)
        ])
        self.modelo.compile(optimizer='adam', loss='mse')

    def treinar(self):
        self.modelo.fit(self.X_treino, self.y_treino, epochs=10, batch_size=16, verbose=1)

    def prever(self):
        previsoes = self.modelo.predict(self.X_teste)
        self.previsoes = self.escalador.inverse_transform(previsoes)

    def prever_futuro(self):
        forecast_input = self.X_teste[-1].reshape(1, self.seq_len, 1)
        forecast = []
        for _ in range(self.dias):
            pred = self.modelo.predict(forecast_input)[0][0]
            forecast.append(pred)
            forecast_input = np.roll(forecast_input, -1, axis=1)
            forecast_input[0, -1, 0] = pred
        self.previsao_futura = self.escalador.inverse_transform(np.array(forecast).reshape(-1, 1))

    def avaliar(self):
        mse = mean_squared_error(self.y_teste_real, self.previsoes)
        r2 = r2_score(self.y_teste_real, self.previsoes)
        print("\n--------------------------------")
        print(f"Métricas de Avaliação - {self.modelo_nome}:")
        print(f"Erro Quadrático Médio (MSE): {mse:.4f}")
        print(f"R-quadrado (R²): {r2:.4f}")
        return mse, r2

    def mostrar_previsoes(self):
        print("--------------------------------")
        print(f"{self.indice_data[-1].strftime('%d/%m/%Y')} - Valor Real: ${float(self.dados['Close'].iloc[-1]):.2f}")
        datas_prev = pd.date_range(start=self.indice_data[-1], periods=self.dias + 1)[1:]
        print("\nPrevisões Futuras:")
        for data, valor in zip(datas_prev, self.previsao_futura):
            print(f"{data.strftime('%d/%m/%Y')} - Previsão: ${valor[0]:.2f}")

    def visualizar(self):
        datas_prev = pd.date_range(start=self.indice_data[-1] + pd.Timedelta(days=1), periods=self.dias)
        plt.figure(figsize=(10, 6))
        plt.plot(self.indice_data, self.y_teste_real, label='Real', color='blue', linewidth=1.5)
        plt.plot(self.indice_data, self.previsoes, label=f'Previsões {self.modelo_nome}', color='green')
        plt.plot(datas_prev, self.previsao_futura, label='Previsão Futura', color='red', linestyle='--')
        plt.axvline(x=self.indice_data[-1], color='gray', linestyle=':', linewidth=1.5)
        plt.axvspan(self.indice_data[-1], datas_prev[-1], color='skyblue', alpha=0.5)
        plt.title(f"{self.modelo_nome} - Previsão de {self.dias} Dias", fontsize=12)
        plt.xlabel("Data")
        plt.ylabel("Preço de Fechamento")
        plt.grid(True, linestyle='--', alpha=0.6)
        plt.legend()
        plt.tight_layout()
        plt.show()

    def run(self):
        self.preparar_dados()
        self.construir_modelo()
        self.treinar()
        self.prever()
        self.prever_futuro()
        self.avaliar()
        self.mostrar_previsoes()
        self.visualizar()


# Classes específicas para cada tipo de modelo
class PrevisorGRU(PrevisorBase):
    def __init__(self, **kwargs):
        super().__init__('GRU', **kwargs)


class PrevisorLSTM(PrevisorBase):
    def __init__(self, **kwargs):
        super().__init__('LSTM', **kwargs)


class PrevisorRNN(PrevisorBase):
    def __init__(self, **kwargs):
        super().__init__('RNN', **kwargs)


