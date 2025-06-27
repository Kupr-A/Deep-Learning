import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

df = pd.read_csv("student-mat.csv")

mapping = {
    "school": {"GP": 0, "MS": 1},
    "sex": {"F": 0, "M": 1},
    "address": {"U": 0, "R": 1},
    "famsize": {"LE3": 0, "GT3": 1},
    "Pstatus": {"A": 0, "T": 1},
    "Mjob": {"teacher": 0, "health": 1, "services": 2, "at_home": 3, "other": 4},
    "Fjob": {"teacher": 0, "health": 1, "services": 2, "at_home": 3, "other": 4},
    "reason": {"home": 0, "reputation": 1, "course": 2, "other": 3},
    "guardian": {"mother": 0, "father": 1, "other": 2},
    "schoolsup": {"yes": 1, "no": 0},
    "famsup": {"yes": 1, "no": 0},
    "paid": {"yes": 1, "no": 0},
    "activities": {"yes": 1, "no": 0},
    "nursery": {"yes": 1, "no": 0},
    "higher": {"yes": 1, "no": 0},
    "internet": {"yes": 1, "no": 0},
    "romantic": {"yes": 1, "no": 0},
}

for col, mapping_dict in mapping.items():
    df[col] = df[col].map(mapping_dict)


scaler = StandardScaler()
df[df.columns] = scaler.fit_transform(df)


df["G3"] = pd.cut(df["G3"], bins=[-1, 1, 3,5, 7, 8, 10, 12, 14, 16, 18, 20 ], labels=[0, 1, 2, 3,4,5,6,7,8, 9, 10])
X = df.drop(columns=["G3"]).values
y = pd.get_dummies(df["G3"]).values


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


class TrainableTransformation:
    def __init__(self, input_dim, output_dim, activation="identity", learning_rate=0.01):
        self.W = np.random.randn(output_dim, input_dim) * 0.01
        self.b = np.zeros(output_dim)
        self.activation = activation
        self.G_W = np.zeros_like(self.W)
        self.G_b = np.zeros_like(self.b)
        self.learning_rate = learning_rate

    def forward(self, x):
        self.x = x
        self.z = np.dot(x, self.W.T) + self.b
        self.a = self.apply_activation(self.z)
        return self.a

    def apply_activation(self, z):
        if self.activation == "identity":
            return z
        elif self.activation == "relu":
            return np.maximum(0, z)
        elif self.activation == "tanh":
            return np.tanh(z)
        else:
            raise ValueError(f"Unsupported activation: {self.activation}")

    def backward(self, grad_output):
        grad_z = self.activation_derivative(self.z) * grad_output
        grad_W = np.dot(grad_z.T, self.x)
        grad_b = np.sum(grad_z, axis=0)
        grad_x = np.dot(grad_z, self.W)

        self.G_W += grad_W ** 2
        self.G_b += grad_b ** 2

        adjusted_lr_W = self.learning_rate / (np.sqrt(self.G_W) + 1e-8)
        adjusted_lr_b = self.learning_rate / (np.sqrt(self.G_b) + 1e-8)

        self.W -= adjusted_lr_W * grad_W
        self.b -= adjusted_lr_b * grad_b
        return grad_x

    def activation_derivative(self, z):
        if self.activation == "identity":
            return np.ones_like(z)
        elif self.activation == "relu":
            return (z > 0).astype(float)
        elif self.activation == "tanh":
            return 1 - np.tanh(z) ** 2
        else:
            raise ValueError(f"Unsupported activation: {self.activation}")


class RBFTrainableTransformation:
    def __init__(self, input_dim, output_dim, gamma=0.1, learning_rate=0.01):
        self.centers = np.random.randn(output_dim, input_dim) * 0.01
        self.b = np.zeros(output_dim)
        self.gamma = gamma
        self.learning_rate = learning_rate
        self.G_centers = np.zeros_like(self.centers)
        self.G_b = np.zeros_like(self.b)

    def forward(self, x):
        self.X = x
        distances = np.linalg.norm(x[:, np.newaxis] - self.centers, axis=2) ** 2
        return np.exp(-self.gamma * distances) + self.b

    def backward(self, grad_output):
        expanded_X = np.expand_dims(self.X, 1)
        expanded_centers = np.expand_dims(self.centers, 0)
        grad_centers = np.sum((expanded_X - expanded_centers) * np.expand_dims(grad_output, 2) * (-2 * self.gamma),
                              axis=0)
        grad_b = np.sum(grad_output, axis=0)

        self.G_centers += grad_centers ** 2
        self.G_b += grad_b ** 2


        adjusted_lr_centers = self.learning_rate / (np.sqrt(self.G_centers) + 1e-8)
        adjusted_lr_b = self.learning_rate / (np.sqrt(self.G_b) + 1e-8)

        self.centers -= adjusted_lr_centers * grad_centers
        self.b -= adjusted_lr_b * grad_b


def softargmax(L):
    exp_L = np.exp(L - np.max(L, axis=1, keepdims=True))
    return exp_L / np.sum(exp_L, axis=1, keepdims=True)


def cross_entropy_loss(y_pred, y_true):
    y_pred = np.clip(y_pred, 1e-9, 1 - 1e-9)
    return -np.mean(np.sum(y_true * np.log(y_pred), axis=1))


def train(model, X_train, y_train, batch_size=32, epochs=10, learning_rate=0.01):
    num_samples = X_train.shape[0]
    for epoch in range(epochs):
        indices = np.random.permutation(num_samples)
        X_train = X_train[indices]
        y_train = y_train[indices]
        total_loss = 0
        for i in range(0, num_samples, batch_size):
            X_batch = X_train[i: i + batch_size]
            y_batch = y_train[i: i + batch_size]
            output = model.forward(X_batch)
            probabilities = softargmax(output)
            loss = cross_entropy_loss(probabilities, y_batch)
            total_loss += loss
            grad_output = probabilities - y_batch # MSE
            model.backward(grad_output)
        print(f"Epoch {epoch + 1}/{epochs}, Loss: {total_loss / (num_samples // batch_size):.4f}")



class MultiLayerTrainableTransformation:
    def __init__(self, input_dim, output_dim, num_layers=3, hidden_dim=64):
        self.layers = []
        for _ in range(num_layers):
            self.layers.append(TrainableTransformation(input_dim, hidden_dim, activation="relu"))
            input_dim = hidden_dim
        self.output_layer = TrainableTransformation(hidden_dim, output_dim, activation="identity")

    def forward(self, x):
        for layer in self.layers:
            x = layer.forward(x)
        return self.output_layer.forward(x)

    def backward(self, grad_output):
        grad_combined = self.output_layer.backward(grad_output)
        for layer in reversed(self.layers):
            grad_combined = layer.backward(grad_combined)
        return grad_combined


class MultiLayerModifiedTrainableTransformation:
    def __init__(self, input_dim, output_dim, num_layers=3, hidden_dim=64):
        self.layers = []
        for _ in range(num_layers):
            self.layers.append(TrainableTransformation(input_dim, hidden_dim, activation="tanh"))
            input_dim = hidden_dim
        self.output_layer = TrainableTransformation(hidden_dim, output_dim, activation="identity")

    def forward(self, x):
        for layer in self.layers:
            x = layer.forward(x)
        return self.output_layer.forward(x)

    def backward(self, grad_output):
        grad_combined = self.output_layer.backward(grad_output)
        for layer in reversed(self.layers):
            grad_combined = layer.backward(grad_combined)
        return grad_combined



def evaluate_model(model, X_train, y_train, X_test, y_test, batch_size=32, epochs=10, learning_rate=0.01):
    print("Training Model")
    train(model, X_train, y_train, batch_size, epochs, learning_rate)

    predictions = model.forward(X_test)
    probabilities = softargmax(predictions)
    predicted_labels = np.argmax(probabilities, axis=1)
    true_labels = np.argmax(y_test, axis=1)
    accuracy = accuracy_score(true_labels, predicted_labels)

    print(f"Model Accuracy: {accuracy:.4f}\n")


print("Training Multi-Layer Trainable Transformation Model")
multi_layer_model = MultiLayerTrainableTransformation(X_train.shape[1], y_train.shape[1], num_layers=3)
evaluate_model(multi_layer_model, X_train, y_train, X_test, y_test)

print("Training Multi-Layer Modified Trainable Transformation Model")
multi_layer_modified_model = MultiLayerModifiedTrainableTransformation(X_train.shape[1], y_train.shape[1], num_layers=3)
evaluate_model(multi_layer_modified_model, X_train, y_train, X_test, y_test)

import matplotlib.pyplot as plt
from sklearn.metrics import f1_score


def evaluate_model(model, X_train, y_train, X_test, y_test, batch_size=32, epochs=10, learning_rate=0.01):
    print("Training Model")
    train(model, X_train, y_train, batch_size, epochs, learning_rate)

    predictions = model.forward(X_test)
    probabilities = softargmax(predictions)
    predicted_labels = np.argmax(probabilities, axis=1)
    true_labels = np.argmax(y_test, axis=1)

    f1 = f1_score(true_labels, predicted_labels, average='weighted')
    print(f"Model F1-Score: {f1:.4f}\n")
    return f1


def experiment_with_transformations(X_train, y_train, X_test, y_test):

    num_layers_range = range(1, 11, 2)

    f1_scores_standard = []
    f1_scores_modified = []



    for num_layers in num_layers_range:
        print(f"Training model with {num_layers} layers.")


        model_standard = MultiLayerTrainableTransformation(X_train.shape[1], y_train.shape[1], num_layers=num_layers)
        f1_standard = evaluate_model(model_standard, X_train, y_train, X_test, y_test, epochs=5)
        f1_scores_standard.append(f1_standard)


        model_modified = MultiLayerModifiedTrainableTransformation(X_train.shape[1], y_train.shape[1],
                                                                   num_layers=num_layers)
        f1_modified = evaluate_model(model_modified, X_train, y_train, X_test, y_test, epochs=5)
        f1_scores_modified.append(f1_modified)


    plt.figure(figsize=(10, 6))
    plt.plot(num_layers_range, f1_scores_standard, marker='o', color='b', label='Standard Model F1-Score')
    plt.plot(num_layers_range, f1_scores_modified, marker='x', color='r', label='Modified Model F1-Score')
    plt.xlabel('Number of Transformations (Layers)')
    plt.ylabel('F1-Score')
    plt.title('F1-Score vs Number of Transformations (Layers)')
    plt.grid(True)
    plt.legend()
    plt.show()

experiment_with_transformations(X_train, y_train, X_test, y_test)
# кривая обучения