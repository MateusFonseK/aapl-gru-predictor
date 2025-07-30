from previsor import PrevisorGRU, PrevisorLSTM, PrevisorRNN

DIAS = 10
SEQ_LEN = 60
TICKER = 'AAPL'
START = '2020-01-01'
END = '2025-04-10'


def menu():
    print("\n=== Previsor de Preço de Ações ===")
    print(f"Ticker: {TICKER}")
    print(f"Data final: {END}")
    print(f"Dias previstos: {DIAS}")
    print("--------------------------------")
    print("1 - Executar modelo GRU")
    print("2 - Executar modelo LSTM")
    print("3 - Executar modelo RNN")
    print("4 - Alterar dados")
    print("0 - Sair")
    return input("Escolha o modelo: ")

def alterar_dados():
    global DIAS, TICKER, END
    novo_ticker = input(f"Novo ticker (Enter para manter '{TICKER}'): ").strip()
    if novo_ticker:
        TICKER = novo_ticker.upper()
    
    nova_data = input(f"Nova data fim (Enter para manter '{END}'): ").strip()
    if nova_data:
        END = nova_data
    
    novo_dias = input(f"Novos dias previstos (Enter para manter {DIAS}): ").strip()
    if novo_dias and novo_dias.isdigit():
        DIAS = int(novo_dias)
    
    print("Dados atualizados!")


if __name__ == "__main__":
    
    while True:
        escolha = menu()

        if escolha == '1':
            modelo = PrevisorGRU(
                dias=DIAS,
                seq_len=SEQ_LEN,
                ticker=TICKER,
                start=START,
                end=END
            )
            modelo.run()

        elif escolha == '2':
            modelo = PrevisorLSTM(
                dias=DIAS,
                seq_len=SEQ_LEN,
                ticker=TICKER,
                start=START,
                end=END
            )
            modelo.run()

        elif escolha == '3':
            modelo = PrevisorRNN(
                dias=DIAS,
                seq_len=SEQ_LEN,
                ticker=TICKER,
                start=START,
                end=END
            )
            modelo.run()

        elif escolha == '4':
            alterar_dados()

        elif escolha == '0':
            break
            break