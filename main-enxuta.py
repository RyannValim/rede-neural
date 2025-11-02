import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay
import numpy as np

# --- Pré-processamento ---
def preprocessar_dados(caminho_csv):
    dados = pd.read_csv(caminho_csv)
    dados["média"] = dados[["math score", "reading score", "writing score"]].mean(axis=1)
    dados["aprovado"] = (dados["média"] >= 70).astype(int)
    X = pd.get_dummies(dados.drop(columns=["média", "aprovado"]))
    X[X.columns] = StandardScaler().fit_transform(X)
    return train_test_split(X, dados["aprovado"], test_size=0.3, random_state=42, stratify=dados["aprovado"])

# --- Modelo ---
class MLP(nn.Module):
    def __init__(self, entrada, ativacao):
        super().__init__()
        self.rede = nn.Sequential(
            nn.Linear(entrada, 64), ativacao(),
            nn.Linear(64, 32), ativacao(),
            nn.Linear(32, 16), ativacao(),
            nn.Linear(16, 1), nn.Sigmoid()
        )
    def forward(self, x): return self.rede(x)

# --- Treinamento ---
def treinar(Xt, yt, Xv, yv, ativacao, epocas=50):
    Xt, yt = torch.tensor(Xt.values, dtype=torch.float32), torch.tensor(yt.values, dtype=torch.float32).view(-1,1)
    Xv, yv = torch.tensor(Xv.values, dtype=torch.float32), torch.tensor(yv.values, dtype=torch.float32).view(-1,1)
    modelo, opt = MLP(Xt.shape[1], ativacao), torch.optim.Adam(params=MLP(Xt.shape[1], ativacao).parameters(), lr=1e-3)
    criterio, loader = nn.BCELoss(), DataLoader(TensorDataset(Xt, yt), batch_size=32, shuffle=True)
    perdas, accs = [], []
    for ep in range(epocas):
        modelo.train(); total = 0
        for xb, yb in loader:
            opt.zero_grad(); out = modelo(xb)
            loss = criterio(out, yb); loss.backward(); opt.step()
            total += loss.item()
        modelo.eval()
        with torch.no_grad():
            pred = (modelo(Xv) >= 0.5).float()
            acc = accuracy_score(yv.numpy(), pred.numpy())
        perdas.append(total/len(loader)); accs.append(acc)
        if ep % 10 == 0: print(f"Época {ep:03d} | Perda {perdas[-1]:.4f} | Acurácia {acc:.4f}")
    return perdas, accs, yv.numpy().astype(int), pred.numpy().astype(int)

# --- Comparação de ativações ---
def comparar(Xt, yt, Xv, yv):
    ativs = {"ReLU": nn.ReLU, "Tanh": nn.Tanh, "Sigmoid": nn.Sigmoid, "LeakyReLU": nn.LeakyReLU}
    hist_loss, hist_acc, medias, melhor = {}, {}, {}, {"nome": "", "acc": 0}
    for nome, f in ativs.items():
        print(f"\nAtivação: {nome}")
        l, a, y_true, y_pred = treinar(Xt, yt, Xv, yv, f)
        hist_loss[nome], hist_acc[nome], medias[nome] = l, a, np.mean(a)
        if medias[nome] > melhor["acc"]:
            melhor = {"nome": nome, "acc": medias[nome], "y_true": y_true, "y_pred": y_pred}
    for hist, t, y in [(hist_loss, "Comparação de Funções de Ativação - Erro (Loss)", "Perda Média"),
                       (hist_acc, "Comparação de Funções de Ativação - Acurácia", "Acurácia no Teste")]:
        plt.figure(figsize=(10,5))
        for n,v in hist.items(): plt.plot(v, label=n)
        plt.title(t); plt.xlabel("Épocas"); plt.ylabel(y); plt.legend(); plt.show()
    print("\nAcurácias médias:")
    for n,v in medias.items(): print(f"{n}: {v:.4f}")
    print(f"\nMelhor função: {melhor['nome']} ({melhor['acc']:.4f})")
    return melhor

# --- Execução ---
if __name__ == "__main__":
    Xt, Xv, yt, yv = preprocessar_dados("src/databases/StudentsPerformance.csv")
    melhor = comparar(Xt, yt, Xv, yv)
    plt.figure(figsize=(6,5))
    ConfusionMatrixDisplay(confusion_matrix(melhor["y_true"], melhor["y_pred"]),
                           display_labels=["Reprovado","Aprovado"]).plot(cmap="Blues")
    plt.title(f"Matriz de Confusão Final - {melhor['nome']}"); plt.show()
    print("Execução concluída com sucesso!")
