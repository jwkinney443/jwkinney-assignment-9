import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from matplotlib.animation import FuncAnimation
import os
from functools import partial
from matplotlib.patches import Circle
from mpl_toolkits.mplot3d import Axes3D

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
        # TODO: define layers and initialize weights
        self.W1 = np.random.randn(input_dim, hidden_dim)
        self.b1 = np.zeros(hidden_dim)
        self.W2 = np.random.randn(hidden_dim, output_dim)
        self.b2 = np.zeros(output_dim)
        self.hidden_activations = []
        self.gradients = {'W1': [], 'b1': [], 'W2': [], 'b2': []}

    def forward(self, X):
        # TODO: forward pass, apply layers to input X
        self.hidden = np.dot(X, self.W1) + self.b1
        self.hidden_activations = self._activate(self.hidden)  # store activations for visualization
        out = np.dot(self.hidden_activations, self.W2) + self.b2
        return out

    def backward(self, X, y):
        # TODO: compute gradients using chain rule
        m = X.shape[0]
        output_error = (self.forward(X) - y) / m
        hidden_error = np.dot(output_error, self.W2.T) * self._activate_derivative(self.hidden)

        grad_W2 = np.dot(self.hidden_activations.T, output_error)
        grad_b2 = np.sum(output_error, axis=0)
        grad_W1 = np.dot(X.T, hidden_error)
        grad_b1 = np.sum(hidden_error, axis=0)

        # TODO: update weights with gradient descent
        self.W1 -= self.lr * grad_W1
        self.b1 -= self.lr * grad_b1
        self.W2 -= self.lr * grad_W2
        self.b2 -= self.lr * grad_b2

        # TODO: store gradients for visualization
        self.gradients['W1'] = grad_W1
        self.gradients['b1'] = grad_b1
        self.gradients['W2'] = grad_W2
        self.gradients['b2'] = grad_b2

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

def generate_data(n_samples=100):
    np.random.seed(0)
    # Generate input
    X = np.random.randn(n_samples, 2)
    y = (X[:, 0] ** 2 + X[:, 1] ** 2 > 1).astype(int) * 2 - 1  # Circular boundary
    y = y.reshape(-1, 1)
    return X, y

def update(frame, mlp, ax_input, ax_hidden, ax_gradient, X, y):
    ax_input.cla()
    ax_hidden.cla()
    ax_gradient.cla()

    # Perform training steps
    for _ in range(10):
        mlp.forward(X)
        mlp.backward(X, y)

    # Plot hidden features (3D scatter plot)
    hidden_features = mlp.hidden_activations
    ax_hidden.scatter(hidden_features[:, 0], hidden_features[:, 1], hidden_features[:, 2], 
                      c=y.ravel(), cmap='bwr', alpha=0.7)
    ax_hidden.set_title(f'Hidden Space at Step {frame * 10}')

    # Fix the hidden space axes
    ax_hidden.set_xlim3d(-1.5, 1.5)
    ax_hidden.set_ylim3d(-1.5, 1.5)
    ax_hidden.set_zlim3d(-1.5, 1.5)

    #TODO ADD a plane showing where the input points on the hidden plot are
    plane_x = np.linspace(X[:, 0].min(), X[:, 0].max(), 30)
    plane_y = np.linspace(X[:, 1].min(), X[:, 1].max(), 30)
    plane_xx, plane_yy = np.meshgrid(plane_x, plane_y)
    plane_grid = np.c_[plane_xx.ravel(), plane_yy.ravel()]

    # Transform the grid points using the hidden layer weights and activation
    plane_hidden = np.dot(plane_grid, mlp.W1) + mlp.b1
    plane_hidden_activations = mlp._activate(plane_hidden)  # Apply activation function

    # Reshape transformed grid back to plane dimensions
    plane_hidden_x = plane_hidden_activations[:, 0].reshape(plane_xx.shape)
    plane_hidden_y = plane_hidden_activations[:, 1].reshape(plane_yy.shape)
    plane_hidden_z = plane_hidden_activations[:, 2].reshape(plane_xx.shape)

    # Plot the distorted plane in the hidden space
    ax_hidden.plot_surface(plane_hidden_x, plane_hidden_y, plane_hidden_z, color='cyan', alpha=0.2)

    # Plot decision hyperplane in the hidden space
    # The decision boundary corresponds to where the output is 0
    hidden_features = mlp.hidden_activations
    w = mlp.W2[:, 0]  # We assume 1D output so we take the weights from hidden to output for classifying
    b = mlp.b2[0]  # Bias term for the output neuron

    # Create a grid in the hidden space
    x_hidden = np.linspace(-1.5, 1.5, 50)
    y_hidden = np.linspace(-1.5, 1.5, 50)
    xx, yy = np.meshgrid(x_hidden, y_hidden)
    
    # Find corresponding zz values using the decision plane equation: w1*x + w2*y + b = 0
    zz = -(w[0] * xx + w[1] * yy + b) / w[2]
    
    ax_hidden.plot_surface(xx, yy, zz, color='gray', alpha=0.5, rstride=100, cstride=100)

    # Distorted input space transformed by the hidden layer
    ax_input.scatter(X[:, 0], X[:, 1], c=y.ravel(), cmap='bwr', edgecolors='k', alpha=0.7)

    # Plot input layer decision boundary
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 100), np.linspace(y_min, y_max, 100))
    grid_points = np.c_[xx.ravel(), yy.ravel()]
    Z = np.sign(mlp.forward(grid_points))
    Z = Z.reshape(xx.shape)

    ax_input.contourf(xx, yy, Z, cmap='bwr', alpha=0.3)

    # Fix the input space axes
    ax_input.set_xlim(x_min, x_max)
    ax_input.set_ylim(y_min, y_max)
    ax_input.set_title(f'Input Space at Step {frame * 10}')

    # Visualize features and gradients as circles and edges
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

    # Add circles and labels
    for i, input_pos in enumerate(input_positions):
        ax_gradient.add_patch(Circle((0, input_pos), 0.1, color='blue', ec='purple', lw=2))
        ax_gradient.text(0.15, input_pos, f'Input {i+1}', color='blue', fontsize=10)

    for j, hidden_pos in enumerate(hidden_positions):
        ax_gradient.add_patch(Circle((1, hidden_pos), 0.1, color='green', ec='purple', lw=2))
        ax_gradient.text(1.15, hidden_pos, f'Hidden {j+1}', color='green', fontsize=10)

    for k, output_pos in enumerate(output_positions):
        ax_gradient.add_patch(Circle((2, output_pos), 0.1, color='red', ec='purple', lw=2))
        ax_gradient.text(2.15, output_pos, f'Output {k+1}', color='red', fontsize=10)

    ax_gradient.set_xlim(-0.5, 2.5)
    ax_gradient.set_ylim(-1.5, 1.5)
    ax_gradient.axis('off')


def visualize(activation, lr, step_num):
    X, y = generate_data()
    mlp = MLP(input_dim=2, hidden_dim=3, output_dim=1, lr=lr, activation=activation)

    # Set up visualization
    matplotlib.use('agg')
    fig = plt.figure(figsize=(21, 7))
    ax_hidden = fig.add_subplot(131, projection='3d')
    ax_input = fig.add_subplot(132)
    ax_gradient = fig.add_subplot(133)

    # Fix axes limits for all frames
    ax_hidden.set_xlim3d(-1.5, 1.5)
    ax_hidden.set_ylim3d(-1.5, 1.5)
    ax_hidden.set_zlim3d(-1.5, 1.5)

    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    ax_input.set_xlim(x_min, x_max)
    ax_input.set_ylim(y_min, y_max)

    # Create animation
    ani = FuncAnimation(fig, partial(update, mlp=mlp, ax_input=ax_input, ax_hidden=ax_hidden, ax_gradient=ax_gradient, X=X, y=y),
                        frames=step_num // 10, repeat=False)

    # Save the animation as a GIF
    ani.save(os.path.join(result_dir, "visualize.gif"), writer='pillow', fps=10)
    plt.close()

if __name__ == "__main__":
    activation = "tanh"
    lr = 0.1
    step_num = 1000
    visualize(activation, lr, step_num)
