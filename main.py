import random
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay

plt.close("all")

# 0) Reprodutibilidade
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

# 1) Pré-processamento dos dados
def preprocessar_dados(caminho_csv: str):
    dados = pd.read_csv(caminho_csv)
    print("Dimensão inicial:", dados.shape)

    # variável-alvo binária: aprovado (média >= 70)
    dados["média"] = dados[["math score", "reading score", "writing score"]].mean(axis=1)
    dados["aprovado"] = (dados["média"] >= 70).astype(int)

    # separação das variáveis independentes e alvo
    X = dados.drop(columns=["média", "aprovado"])
    y = dados["aprovado"]

    # one-hot nas variáveis categóricas
    X = pd.get_dummies(X)

    # normalização
    normalizador = StandardScaler()
    X[X.columns] = normalizador.fit_transform(X)

    # divisão treino/teste
    X_treino, X_teste, y_treino, y_teste = train_test_split(
        X, y, test_size=0.30, random_state=SEED, stratify=y
    )

    print(f"Tamanho do treino: {X_treino.shape}, Tamanho do teste: {X_teste.shape}")
    return X_treino, X_teste, y_treino, y_teste

# 2) Modelo de Rede Neural MLP
class MLP(nn.Module):
    def __init__(self, dimensao_entrada: int, funcao_ativacao):
        super().__init__()
        self.rede = nn.Sequential(
            nn.Linear(dimensao_entrada, 64),
            funcao_ativacao(),
            nn.Linear(64, 32),
            funcao_ativacao(),
            nn.Linear(32, 16),
            funcao_ativacao(),
            nn.Linear(16, 1),
            nn.Sigmoid(),  # saída binária
        )

    def forward(self, x):
        return self.rede(x)

# 3) Treinamento do modelo (sem plots)
def treinar_modelo(X_treino, y_treino, X_teste, y_teste, funcao_ativacao, epocas=50, batch_size=32, lr=1e-3):
    print(f"\nTreinando com ativação: {funcao_ativacao.__name__}")

    # tensores
    X_treino_t = torch.tensor(X_treino.values, dtype=torch.float32)
    y_treino_t = torch.tensor(y_treino.values, dtype=torch.float32).view(-1, 1)
    X_teste_t  = torch.tensor(X_teste.values,  dtype=torch.float32)
    y_teste_t  = torch.tensor(y_teste.values,  dtype=torch.float32).view(-1, 1)

    # dataloader
    dados_treino = TensorDataset(X_treino_t, y_treino_t)
    carregador = DataLoader(dados_treino, batch_size=batch_size, shuffle=True)

    # modelo, perda e otimizador
    modelo = MLP(X_treino.shape[1], funcao_ativacao)
    criterio = nn.BCELoss()
    otimizador = torch.optim.Adam(modelo.parameters(), lr=lr)

    perdas, acuracias = [], []

    for epoca in range(epocas):
        modelo.train()
        perda_acum = 0.0

        for xb, yb in carregador:
            otimizador.zero_grad()
            saida = modelo(xb)
            perda = criterio(saida, yb)
            perda.backward()
            otimizador.step()
            perda_acum += perda.item()

        # avaliação por época
        modelo.eval()
        with torch.no_grad():
            probas = modelo(X_teste_t)
            preds  = (probas >= 0.5).float()
            acc    = accuracy_score(y_teste_t.numpy(), preds.numpy())

        perdas.append(perda_acum / len(carregador))
        acuracias.append(acc)

        if epoca % 10 == 0:
            print(f"Época {epoca:03d} | Perda: {perdas[-1]:.4f} | Acurácia: {acc:.4f}")

    # resultados
    resultados = pd.DataFrame({
        "Aprovado_Real": y_teste_t.flatten().numpy().astype(int),
        "Aprovado_Previsto": preds.flatten().numpy().astype(int)
    })
    print("\nAmostra de previsões (1=aprovado, 0=reprovado):")
    print(resultados.head(10))
    print("\nTotal de acertos:", (resultados["Aprovado_Real"] == resultados["Aprovado_Previsto"]).sum())
    print("Total de erros:",   (resultados["Aprovado_Real"] != resultados["Aprovado_Previsto"]).sum())

    y_true = resultados["Aprovado_Real"].to_numpy()
    y_pred = resultados["Aprovado_Previsto"].to_numpy()
    return perdas, acuracias, y_true, y_pred

# 4) Funções auxiliares de plotagem
def plot_curvas(hist_dict, titulo, ylabel):
    fig, ax = plt.subplots(figsize=(12, 5))
    for nome, valores in hist_dict.items():
        ax.plot(valores, label=nome)
    ax.set_title(titulo)
    ax.set_xlabel("Épocas")
    ax.set_ylabel(ylabel)
    ax.legend()
    plt.tight_layout()
    plt.show()
    plt.close(fig)


def plot_matriz_confusao(y_true, y_pred, titulo):
    fig, ax = plt.subplots(figsize=(6, 5))
    mat = confusion_matrix(y_true, y_pred)
    disp = ConfusionMatrixDisplay(mat, display_labels=["Reprovado", "Aprovado"])
    disp.plot(cmap="Blues", ax=ax, colorbar=True)
    ax.set_title(titulo)
    plt.tight_layout()
    plt.show()
    plt.close(fig)

# 5) Comparação automática de funções de ativação
def comparar_ativacoes(X_treino, y_treino, X_teste, y_teste, epocas=50):
    ativacoes = {
        "ReLU": nn.ReLU,
        "Tanh": nn.Tanh,
        "Sigmoid": nn.Sigmoid,
        "LeakyReLU": nn.LeakyReLU,
    }

    plt.close("all")

    perdas_hist = {}
    accs_hist   = {}
    acuracia_media = {}
    melhor = {"nome": None, "acc": -1, "y_true": None, "y_pred": None}

    for nome, ativ in ativacoes.items():
        perdas, accs, y_true, y_pred = treinar_modelo(
            X_treino, y_treino, X_teste, y_teste, ativ, epocas=epocas
        )
        perdas_hist[nome] = perdas
        accs_hist[nome]   = accs
        media = float(np.mean(accs))
        acuracia_media[nome] = media

        if media > melhor["acc"]:
            melhor = {"nome": nome, "acc": media, "y_true": y_true, "y_pred": y_pred}

    plot_curvas(perdas_hist, "Comparação de Funções de Ativação - Erro (Loss)", "Perda Média")
    plot_curvas(accs_hist,   "Comparação de Funções de Ativação - Acurácia",   "Acurácia no Teste")

    print("\nAcurácias médias por ativação:")
    for nome, val in acuracia_media.items():
        print(f"{nome}: {val:.4f}")

    print(f"\n🔹 Melhor função identificada automaticamente: {melhor['nome']} (acc média = {melhor['acc']:.4f})")
    return melhor

# 6) Execução principal
if __name__ == "__main__":
    plt.close("all")

    X_treino, X_teste, y_treino, y_teste = preprocessar_dados("src/databases/StudentsPerformance.csv")

    melhor = comparar_ativacoes(X_treino, y_treino, X_teste, y_teste, epocas=50)

    # matriz de confusão final
    plot_matriz_confusao(
        melhor["y_true"],
        melhor["y_pred"],
        f"Matriz de Confusão Final - {melhor['nome']}"
    )

    plt.close("all")
    print("✅ Execução completa!")