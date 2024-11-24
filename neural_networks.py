import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from matplotlib.animation import FuncAnimation
import os
from functools import partial
from matplotlib.patches import Circle

result_dir = "results"
os.makedirs(result_dir, exist_ok=True)

# Define a simple MLP class
class MLP:
    def __init__(self, input_dim, hidden_dim, output_dim, lr, activation='tanh'):
        np.random.seed(0)
        self.lr = lr  # learning rate
        self.activation_fn = activation  # activation function
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.W1 = np.random.randn(input_dim, hidden_dim)
        self.b1 = np.zeros(hidden_dim)
        self.W2 = np.random.randn(hidden_dim, output_dim)
        self.b2 = np.zeros(output_dim)
        self.hidden_activations = []
        self.gradients = {'W1': [], 'b1': [], 'W2': [], 'b2': []}

    def _activate(self, x):
        """Apply the selected activation function."""
        if self.activation_fn == 'tanh':
            return np.tanh(x)
        elif self.activation_fn == 'relu':
            return np.maximum(0, x)
        elif self.activation_fn == 'sigmoid':
            return 1 / (1 + np.exp(-x))
        else:
            raise ValueError(f"Unsupported activation function: {self.activation_fn}")

    def _activate_derivative(self, x):
        """Compute the derivative of the activation function."""
        if self.activation_fn == 'tanh':
            return 1 - np.tanh(x) ** 2
        elif self.activation_fn == 'relu':
            return (x > 0).astype(float)
        elif self.activation_fn == 'sigmoid':
            sigmoid_x = 1 / (1 + np.exp(-x))
            return sigmoid_x * (1 - sigmoid_x)
        else:
            raise ValueError(f"Unsupported activation function: {self.activation_fn}")

    def forward(self, X):
        self.hidden = np.dot(X, self.W1) + self.b1
        self.hidden_activations = self._activate(self.hidden)
        out = np.dot(self.hidden_activations, self.W2) + self.b2
        return out

    def binary_cross_entropy(self, y_pred, y_true):
        epsilon = 1e-15 
        y_pred = np.clip(y_pred, epsilon, 1 - epsilon)  
        return -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))

    def backward(self, X, y):
        m = X.shape[0]
        output_error = (self.forward(X) - y) / m
        hidden_error = np.dot(output_error, self.W2.T) * self._activate_derivative(self.hidden)

        grad_W2 = np.dot(self.hidden_activations.T, output_error)
        grad_b2 = np.sum(output_error, axis=0)
        grad_W1 = np.dot(X.T, hidden_error)
        grad_b1 = np.sum(hidden_error, axis=0)

        self.W1 -= self.lr * grad_W1
        self.b1 -= self.lr * grad_b1
        self.W2 -= self.lr * grad_W2
        self.b2 -= self.lr * grad_b2

        self.gradients['W1'] = grad_W1
        self.gradients['b1'] = grad_b1
        self.gradients['W2'] = grad_W2
        self.gradients['b2'] = grad_b2

def generate_data(n_samples=100):
    np.random.seed(0)
    X = np.random.randn(n_samples, 2)
    y = (X[:, 0] ** 2 + X[:, 1] ** 2 > 1).astype(int) * 2 - 1  
    y = y.reshape(-1, 1)
    return X, y

# Visualization update function
def update(frame, mlp, ax_input, ax_hidden, ax_gradient, X, y):
    ax_hidden.clear()
    ax_input.clear()
    ax_gradient.clear()

    for _ in range(10):
        mlp.forward(X)
        mlp.backward(X, y)

    ax_hidden.scatter(
        mlp.hidden_activations[:, 0], mlp.hidden_activations[:, 1], mlp.hidden_activations[:, 2],
        c=y.ravel(), cmap='bwr', alpha=0.7
    )
    ax_hidden.set_title('Hidden Space at Step {}'.format(frame * 10))

    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 100), np.linspace(y_min, y_max, 100))
    grid_points = np.c_[xx.ravel(), yy.ravel()]
    Z = np.sign(mlp.forward(grid_points))
    Z = Z.reshape(xx.shape)

    ax_input.contourf(xx, yy, Z, cmap='bwr', alpha=0.5)
    ax_input.scatter(X[:, 0], X[:, 1], c=y.ravel(), cmap='bwr', edgecolors='k', alpha=0.7)
    ax_input.set_title('Input Space at Step {}'.format(frame * 10))

    ax_gradient.set_title('Gradients at Step {}'.format(frame * 10))
    input_positions = np.linspace(-1, 1, mlp.input_dim)
    hidden_positions = np.linspace(-1, 1, mlp.hidden_dim)
    output_positions = [0]

    for i, input_pos in enumerate(input_positions):
        for j, hidden_pos in enumerate(hidden_positions):
            gradient_magnitude = abs(mlp.gradients['W1'][i, j])
            ax_gradient.plot([0, 1], [input_pos, hidden_pos], color='purple',
                             linewidth=gradient_magnitude * 12, alpha=0.9)

    for j, hidden_pos in enumerate(hidden_positions):
        for k, output_pos in enumerate(output_positions):
            gradient_magnitude = abs(mlp.gradients['W2'][j, k])
            ax_gradient.plot([1, 2], [hidden_pos, output_pos], color='purple',
                             linewidth=gradient_magnitude * 13, alpha=0.9)

    for i, input_pos in enumerate(input_positions):
        ax_gradient.add_patch(Circle((0, input_pos), 0.1, color='blue', ec='purple', lw=2))
    for j, hidden_pos in enumerate(hidden_positions):
        ax_gradient.add_patch(Circle((1, hidden_pos), 0.1, color='green', ec='purple', lw=2))
    for k, output_pos in enumerate(output_positions):
        ax_gradient.add_patch(Circle((2, output_pos), 0.1, color='red', ec='purple', lw=2))

    ax_gradient.set_xlim(-0.5, 2.5)
    ax_gradient.set_ylim(-1.5, 1.5)
    ax_gradient.axis('off')

def visualize(activation, lr, step_num):
    X, y = generate_data()
    mlp = MLP(input_dim=2, hidden_dim=3, output_dim=1, lr=lr, activation=activation)

    matplotlib.use('agg')
    fig = plt.figure(figsize=(21, 7))
    ax_hidden = fig.add_subplot(131, projection='3d')
    ax_input = fig.add_subplot(132)
    ax_gradient = fig.add_subplot(133)

    ani = FuncAnimation(fig, partial(update, mlp=mlp, ax_input=ax_input, ax_hidden=ax_hidden,
                                      ax_gradient=ax_gradient, X=X, y=y),
                        frames=step_num//10, repeat=False)

    ani.save(os.path.join(result_dir, "visualize.gif"), writer='pillow', fps=10)
    plt.close()

if __name__ == "__main__":
    activation = "tanh"  
    lr = 0.1
    step_num = 1000
    visualize(activation, lr, step_num)
