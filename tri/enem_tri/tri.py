import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim


def load_data(filepath):
    """
    Carrega a CSV de respostas booleanas (1=True, 0=False) com alunos nas linhas e itens nas colunas.
    Converte para DataFrame de booleanos e retorna como 0/1 em float32.
    """
    # Lê o CSV e força interpretação como inteiros
    df = pd.read_csv(filepath, dtype=np.int8)
    # Converte valores para booleano e depois para float32 (0.0 ou 1.0)
    bool_df = df.astype(bool)
    float_df = bool_df.astype(np.float32)
    return float_df


class ThreePLIrtModel(nn.Module):
    """
    Implementa o modelo 3PL (três parâmetros logístico) de IRT.
    Parâmetros:
      a (discriminação), b (dificuldade), c (chance), θ (habilidade).
    """
    def __init__(self, num_items, num_students, device='cpu'):
        super().__init__()
        # Inicialização dos parâmetros do item
        self.a = nn.Parameter(torch.ones(num_items, device=device))       # discriminação > 0
        self.b = nn.Parameter(torch.zeros(num_items, device=device))      # dificuldade (real)
        self.c = nn.Parameter(torch.full((num_items,), 0.2, device=device))  # palpite em [0,1]
        # Habilidades dos alunos
        self.theta = nn.Parameter(torch.zeros(num_students, device=device))

    def forward(self):
        # Organiza tensores para cálculo vetorizado
        theta = self.theta.unsqueeze(1)        # [N_alunos, 1]
        a     = self.a.unsqueeze(0)            # [1, N_itens]
        b     = self.b.unsqueeze(0)            # [1, N_itens]
        c     = torch.clamp(self.c.unsqueeze(0), 0.0, 1.0)  # garante [0,1]

        # Curva logística: σ(a (θ - b))
        logistic = torch.sigmoid(a * (theta - b))
        # Modelo 3PL: P = c + (1-c) * logistic
        P = c + (1.0 - c) * logistic
        return P


def fit_3pl(response_df, lr=0.01, epochs=100, device='cpu'):
    """
    Ajusta o modelo 3PL pelo método de máxima verossimilhança via gradiente.

    Args:
      response_df: DataFrame com 0.0/1.0 (float32) indicando erros/acertos.
      lr: taxa de aprendizado.
      epochs: número de iterações de treino.
      device: 'cpu' ou 'cuda'.

    Returns:
      dict com arrays numpy: a, b, c, theta.
    """
    # Converte DataFrame para numpy e tensor
    data = response_df.values  # já float32
    num_students, num_items = data.shape

    device = torch.device(device)
    data_tensor = torch.from_numpy(data).to(device)

    # Instancia modelo e otimizador
    model = ThreePLIrtModel(num_items, num_students, device=device).to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)

    # Treinamento
    for epoch in range(1, epochs + 1):
        model.train()
        optimizer.zero_grad()

        P = model()  # previsão [N, M]
        # Log-verossimilhança (com epsilon para estabilidade)
        eps = 1e-9
        ll = data_tensor * torch.log(P + eps) + (1 - data_tensor) * torch.log(1 - P + eps)
        loss = -ll.mean()

        loss.backward()
        optimizer.step()

        if epoch == 1 or epoch % max(1, epochs // 10) == 0:
            print(f"Epoch {epoch}/{epochs} - Loss: {loss.item():.6f}")

    # Extrai estimativas finais
    a_est     = model.a.detach().cpu().numpy()
    b_est     = model.b.detach().cpu().numpy()
    c_est     = model.c.detach().cpu().numpy()
    theta_est = model.theta.detach().cpu().numpy()

    return {
        'a': a_est,
        'b': b_est,
        'c': c_est,
        'theta': theta_est
    }


def main():
    import argparse
    parser = argparse.ArgumentParser(description='Estima habilidades usando o modelo 3PL de IRT')
    parser.add_argument('--data', type=str, required=True,
                        help='Caminho para o CSV de respostas (1=True, 0=False)')
    parser.add_argument('--lr', type=float, default=0.01, help='Taxa de aprendizado')
    parser.add_argument('--epochs', type=int, default=100, help='Número de épocas')
    parser.add_argument('--device', type=str, default='cpu', help="'cpu' ou 'cuda'")
    parser.add_argument('--output', type=str, default='estimates.npz',
                        help='Arquivo de saída .npz com parâmetros')
    args = parser.parse_args()

    # Carrega e converte a base booleana
    df = load_data(args.data)
    # Treina o modelo
    results = fit_3pl(df, lr=args.lr, epochs=args.epochs, device=args.device)
    # Salva em NPZ
    np.savez(args.output,
             a=results['a'],
             b=results['b'],
             c=results['c'],
             theta=results['theta'])
    print(f"Estimativas salvas em {args.output}")


if __name__ == '__main__':
    main()