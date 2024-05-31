import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler


class SVM:
    def __init__(self, learning_rate=0.0001, lambda_param=0.01, n_iters=10000, soft_margin=False, C=1.0):
        self.learning_rate = learning_rate
        self.lambda_param = lambda_param
        self.n_iters = n_iters
        self.soft_margin = soft_margin
        self.C = C
        self.w = None
        self.b = None

    def fit(self, X, y):
        n_samples, n_features = X.shape
        y_ = np.where(y <= 0, -1, 1)

        # Initialize weights
        self.w = np.zeros(n_features)
        self.b = 0

        # Gradient Descent
        for _ in range(self.n_iters):
            for idx, x_i in enumerate(X):
                condition = y_[idx] * (np.dot(x_i, self.w) - self.b) >= 1
                if condition:
                    self.w -= self.learning_rate * (2 * self.lambda_param * self.w)
                else:
                    self.w -= self.learning_rate * (2 * self.lambda_param * self.w - np.dot(x_i, y_[idx]))
                    self.b -= self.learning_rate * y_[idx]

                if self.soft_margin:
                    slack = max(0, 1 - y_[idx] * (np.dot(x_i, self.w) - self.b))
                    if slack > 0:
                        self.w -= self.learning_rate * (2 * self.lambda_param * self.w - self.C * slack * np.dot(x_i, y_[idx]))
                        self.b -= self.learning_rate * self.C * slack * y_[idx]

    def predict(self, X):
        approx = np.dot(X, self.w) - self.b
        return np.sign(approx)

def calculate_accuracy(y_true, y_pred):
    return np.mean(y_true == y_pred)


def visualize_svm(clf, X, y, title):
    def get_hyperplane_value(x, w, b, offset):
        return (-w[0] * x + b + offset) / w[1]

    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    plt.scatter(X[:, 0], X[:, 1], marker="o", c=y)

    x0_1 = np.amin(X[:, 0])
    x0_2 = np.amax(X[:, 0])

    x1_1 = get_hyperplane_value(x0_1, clf.w, clf.b, 0)
    x1_2 = get_hyperplane_value(x0_2, clf.w, clf.b, 0)

    x1_1_m = get_hyperplane_value(x0_1, clf.w, clf.b, -1)
    x1_2_m = get_hyperplane_value(x0_2, clf.w, clf.b, -1)

    x1_1_p = get_hyperplane_value(x0_1, clf.w, clf.b, 1)
    x1_2_p = get_hyperplane_value(x0_2, clf.w, clf.b, 1)

    ax.plot([x0_1, x0_2], [x1_1, x1_2], "y--")
    ax.plot([x0_1, x0_2], [x1_1_m, x1_2_m], "k")
    ax.plot([x0_1, x0_2], [x1_1_p, x1_2_p], "k")

    x1_min = np.amin(X[:, 1])
    x1_max = np.amax(X[:, 1])
    ax.set_ylim([x1_min - 3, x1_max + 3])

    plt.title(title)
    plt.show()


def create_D1():
    np.random.seed(0)
    class_1_D1 = np.random.randn(100, 2) + np.array([2, 2])
    class_2_D1 = np.random.randn(100, 2) + np.array([-2, -2])
    D1 = np.vstack((class_1_D1, class_2_D1))
    labels_D1 = np.hstack((np.ones(100), -1 * np.ones(100)))
    return D1, labels_D1

def create_D2():
    np.random.seed(0)
    class_1_D2 = np.random.randn(100, 2)
    class_2_D2 = np.random.randn(100, 2) + np.array([1.5, 1.5])
    D2 = np.vstack((class_1_D2, class_2_D2))
    labels_D2 = np.hstack((np.ones(100), -1 * np.ones(100)))
    return D2, labels_D2


if __name__ == '__main__':
    # Load datasets D1 and D2
    D1, labels_D1 = create_D1()
    D2, labels_D2 = create_D2()
    scaler = StandardScaler()
    D1 = scaler.fit_transform(D1)
    D2 = scaler.fit_transform(D2)


    hard_margin_svm_D1 = SVM(learning_rate=0.0001, lambda_param=0.01, n_iters=1000, soft_margin=False)
    hard_margin_svm_D1.fit(D1, labels_D1)

    soft_margin_svm_D2 = SVM(learning_rate=0.0001, lambda_param=0.01, n_iters=1000, soft_margin=True, C=1.0)
    soft_margin_svm_D2.fit(D2, labels_D2)


    hard_margin_svm_D1_predictions = hard_margin_svm_D1.predict(D1)
    hard_margin_svm_D1_accuracy = calculate_accuracy(labels_D1, hard_margin_svm_D1_predictions)


    soft_margin_svm_D2_predictions = soft_margin_svm_D2.predict(D2)
    soft_margin_svm_D2_accuracy = calculate_accuracy(labels_D2, soft_margin_svm_D2_predictions)

    print("Hard Margin SVM Accuracy (D1):", hard_margin_svm_D1_accuracy)
    print("Soft Margin SVM Accuracy (D2):", soft_margin_svm_D2_accuracy)


    visualize_svm(hard_margin_svm_D1, D1, labels_D1, "Hard Margin SVM (D1)")
    visualize_svm(soft_margin_svm_D2, D2, labels_D2, "Soft Margin SVM (D2)")
