import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


class MLP:
    def __init__(self, learning_rate=0.01, n_iters=1000):
        self.learning_rate = learning_rate
        self.n_iters = n_iters
        self.w1 = None
        self.b1 = None
        self.w2 = None
        self.b2 = None

    def _sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def _sigmoid_derivative(self, x):
        return x * (1 - x)

    def fit(self, X, y):
        n_samples, n_features = X.shape
        n_outputs = y.shape[1]

        # Initialize weights and biases
        self.w1 = np.random.randn(n_features, 64) * 0.01
        self.b1 = np.zeros((1, 64))
        self.w2 = np.random.randn(64, n_outputs) * 0.01
        self.b2 = np.zeros((1, n_outputs))

        # Gradient Descent
        for _ in range(self.n_iters):
            # Forward propagation
            hidden_layer_input = np.dot(X, self.w1) + self.b1
            hidden_layer_output = self._sigmoid(hidden_layer_input)
            output_layer_input = np.dot(hidden_layer_output, self.w2) + self.b2
            output_layer_output = self._sigmoid(output_layer_input)

            # Backpropagation
            error = y - output_layer_output
            d_output = error * self._sigmoid_derivative(output_layer_output)
            error_hidden_layer = d_output.dot(self.w2.T)
            d_hidden_layer = error_hidden_layer * self._sigmoid_derivative(hidden_layer_output)

            # Update weights and biases
            self.w2 += hidden_layer_output.T.dot(d_output) * self.learning_rate
            self.b2 += np.sum(d_output, axis=0, keepdims=True) * self.learning_rate
            self.w1 += X.T.dot(d_hidden_layer) * self.learning_rate
            self.b1 += np.sum(d_hidden_layer, axis=0, keepdims=True) * self.learning_rate

    def predict(self, X):
        hidden_layer_input = np.dot(X, self.w1) + self.b1
        hidden_layer_output = self._sigmoid(hidden_layer_input)
        output_layer_input = np.dot(hidden_layer_output, self.w2) + self.b2
        output_layer_output = self._sigmoid(output_layer_input)
        return output_layer_output

# Define functions to create datasets D1 and D2
def create_D1():
    np.random.seed(0)
    class_1_D1 = np.random.randn(100, 2) + np.array([2, 2])
    class_2_D1 = np.random.randn(100, 2) + np.array([-2, -2])
    D1 = np.vstack((class_1_D1, class_2_D1))
    labels_D1 = np.hstack((np.ones(100), np.zeros(100)))
    return D1, labels_D1

def create_D2():
    class_1_D2 = np.random.randn(100, 2)
    class_2_D2 = np.random.randn(100, 2) + np.array([1.5, 1.5])
    D2 = np.vstack((class_1_D2, class_2_D2))
    labels_D2 = np.hstack((np.ones(100), np.zeros(100)))
    return D2, labels_D2


D1, labels_D1 = create_D1()
D2, labels_D2 = create_D2()


D1_train, D1_test, labels_D1_train, labels_D1_test = train_test_split(D1, labels_D1, test_size=0.2, random_state=42)
D2_train, D2_test, labels_D2_train, labels_D2_test = train_test_split(D2, labels_D2, test_size=0.2, random_state=42)

mlp_D1 = MLP()
mlp_D1.fit(D1_train, labels_D1_train.reshape(-1, 1))
mlp_D1_predictions = (mlp_D1.predict(D1_test) > 0.5).astype(int).flatten()
mlp_D1_accuracy = accuracy_score(labels_D1_test, mlp_D1_predictions)

mlp_D2 = MLP()
mlp_D2.fit(D2_train, labels_D2_train.reshape(-1, 1))
mlp_D2_predictions = (mlp_D2.predict(D2_test) > 0.5).astype(int).flatten()
mlp_D2_accuracy = accuracy_score(labels_D2_test, mlp_D2_predictions)

print("MLP Accuracy (D1):", mlp_D1_accuracy)
print("MLP Accuracy (D2):", mlp_D2_accuracy)



def plot_decision_boundary(model, X, y, title):
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.01),
                         np.arange(y_min, y_max, 0.01))
    Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = (Z > 0.5).astype(int)
    Z = Z.reshape(xx.shape)
    plt.contourf(xx, yy, Z, alpha=0.8)
    plt.scatter(X[:, 0], X[:, 1], c=y, edgecolors='k', marker='o')
    plt.title(title)
    plt.show()


plot_decision_boundary(mlp_D1, D1_test, labels_D1_test, "MLP Decision Boundary for D1")
plot_decision_boundary(mlp_D2, D2_test, labels_D2_test, "MLP Decision Boundary for D2")



