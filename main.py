import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
import time


def generate_data(n_samples=2000, noise_std=0.05):
    """Create random points in [-1,1]^2 with target function sin(pi*x1)*cos(pi*x2)."""
    torch.manual_seed(42)
    X = torch.rand(n_samples, 2) * 2 - 1  # uniform in [-1, 1]
    y = torch.sin(np.pi * X[:, 0]) * torch.cos(np.pi * X[:, 1])
    if noise_std > 0:
        y += noise_std * torch.randn_like(y)
    return X, y.unsqueeze(1)  # shape (n_samples, 1)

X, y = generate_data(2000)
X_train, X_test = X[:1500], X[1500:]
y_train, y_test = y[:1500], y[1500:]

class BaseKAN(nn.Module):
    """
    Base class: single layer with basis expansion per input feature.
    For each input-output pair we have `num_basis` trainable coefficients.
    Output: y_j = sum_i sum_k coeff[i,j,k] * basis_k(x_i)
    """
    def __init__(self, input_dim, output_dim, num_basis):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.num_basis = num_basis
        # Coefficients: (input_dim, output_dim, num_basis)
        self.coeff = nn.Parameter(torch.randn(input_dim, output_dim, num_basis) * 0.1)

    def basis_functions(self, x):
        """
        x: (batch_size, input_dim)
        returns: (batch_size, input_dim, num_basis) with basis values
        """
        raise NotImplementedError

    def forward(self, x):
        # x: (batch_size, input_dim)
        basis_vals = self.basis_functions(x)          # (B, in, num_basis)
        coeff_exp = self.coeff.unsqueeze(0)           # (1, in, out, num_basis)
        basis_exp = basis_vals.unsqueeze(2)           # (B, in, 1, num_basis)
        # Sum over input features and basis functions
        out = (coeff_exp * basis_exp).sum(dim=(1, 3)) # (B, out)
        return out

class ChebyKAN(BaseKAN):
    """Chebyshev polynomials of the first kind as basis functions."""
    def basis_functions(self, x):
        # x in [-1, 1]
        B, in_dim = x.shape
        # Recursively compute T_k(x) for k = 0 .. num_basis-1
        T = [torch.ones_like(x)]          # T0
        if self.num_basis > 1:
            T.append(x)                    # T1
        for k in range(2, self.num_basis):
            T_k = 2 * x * T[k-1] - T[k-2]
            T.append(T_k)
        # Stack: (num_basis, B, in) -> (B, in, num_basis)
        basis = torch.stack(T, dim=0).permute(1, 2, 0)
        return basis

class LinearSplineKAN(BaseKAN):
    """Linear splines (hat functions) on a uniform grid in [-1, 1]."""
    def __init__(self, input_dim, output_dim, num_basis):
        super().__init__(input_dim, output_dim, num_basis)
        self.register_buffer('grid', torch.linspace(-1, 1, num_basis))
        self.spacing = 2.0 / (num_basis - 1)

    def basis_functions(self, x):
        # x: (B, in)
        B, in_dim = x.shape
        # Distance to each grid point: (B, in, num_basis)
        dist = torch.abs(x.unsqueeze(-1) - self.grid)
        # Hat function: max(0, 1 - dist / spacing)
        basis = torch.clamp(1 - dist / self.spacing, min=0.0)
        return basis

class PolyKAN(BaseKAN):
    """Plain monomials x^k (ill-conditioned for large k)."""
    def basis_functions(self, x):
        B, in_dim = x.shape
        powers = torch.arange(self.num_basis, device=x.device).float()
        # Expand x to powers: (B, in, 1) ^ (1, 1, num_basis)
        x_exp = x.unsqueeze(-1).expand(B, in_dim, self.num_basis)
        basis = x_exp ** powers
        return basis

def train_model(model, X_train, y_train, X_test, y_test, epochs=500, lr=1e-2):
    optimizer = optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()
    train_losses = []
    test_losses = []
    start_time = time.time()
    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        y_pred = model(X_train)
        loss = loss_fn(y_pred, y_train)
        loss.backward()
        optimizer.step()
        train_losses.append(loss.item())

        model.eval()
        with torch.no_grad():
            y_test_pred = model(X_test)
            test_loss = loss_fn(y_test_pred, y_test)
            test_losses.append(test_loss.item())

        if (epoch+1) % 100 == 0:
            print(f'Epoch {epoch+1}/{epochs} - Train Loss: {loss.item():.4e} - Test Loss: {test_loss.item():.4e}')

    elapsed = time.time() - start_time
    return train_losses, test_losses, elapsed

num_basis = 10
models = {
    'ChebyKAN': ChebyKAN(input_dim=2, output_dim=1, num_basis=num_basis),
    'LinearSplineKAN': LinearSplineKAN(2, 1, num_basis),
    'PolyKAN': PolyKAN(2, 1, num_basis)
}

results = {}
for name, model in models.items():
    print(f"\n--- Training {name} ---")
    losses_train, losses_test, train_time = train_model(model, X_train, y_train, X_test, y_test)
    results[name] = {
        'train_losses': losses_train,
        'test_losses': losses_test,
        'time': train_time,
        'final_test_loss': losses_test[-1]
    }


plt.figure(figsize=(10,6))
for name, res in results.items():
    plt.plot(res['train_losses'], label=f'{name} (train)')
plt.yscale('log')
plt.xlabel('Epoch')
plt.ylabel('MSE (train)')
plt.title('Efficiency Comparison: ChebyKAN vs Traditional KANs')
plt.legend()
plt.grid(True)
plt.show()

print("\n=== Summary ===")
for name, res in results.items():
    print(f"{name:15s} | Total time: {res['time']:.2f}s | Final test MSE: {res['final_test_loss']:.3e}")
