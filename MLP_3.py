import numpy as np
import torchvision
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
import torchvision.transforms as transforms


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def softmax(x):
    exponent = np.exp(x)
    return exponent / exponent.sum(axis=1, keepdims=True)


def dataloader(train_dataset, test_dataset, batch_size=128):
    train_loader = DataLoader(
        dataset=train_dataset, batch_size=batch_size, shuffle=True
    )
    test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)
    return train_loader, test_loader


def load_data():
    transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))]
    )
    train_dataset = torchvision.datasets.MNIST(
        root="./data/mnist", train=True, download=True, transform=transform
    )
    test_dataset = torchvision.datasets.MNIST(
        root="./data/mnist", train=False, download=True, transform=transform
    )
    print("The number of training data:", len(train_dataset))
    print("The number of testing data:", len(test_dataset))
    return dataloader(train_dataset, test_dataset)


class MLP:
    def __init__(self, input_size, hidden_size, output_size, lr, params_path=None):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.lr = lr
        if params_path is not None:
            self.load_model_params(params_path)
        else:
            self.weights1 = np.random.randn(input_size, hidden_size) * 0.01
            self.bias1 = np.zeros(hidden_size)
            self.weights2 = np.random.randn(hidden_size, output_size) * 0.01
            self.bias2 = np.zeros(output_size)

    def forward(self, x):
        self.hidden_layer_input = np.dot(x, self.weights1) + self.bias1
        self.hidden_layer_output = sigmoid(self.hidden_layer_input)
        self.output_layer_input = (
            np.dot(self.hidden_layer_output, self.weights2) + self.bias2
        )
        outputs = softmax(self.output_layer_input)
        return outputs

    def backward(self, x, y, pred):

        one_hot_y = np.zeros((len(y), self.output_size))
        one_hot_y[np.arange(len(y)), y] = 1

        d_output = pred - one_hot_y
        d_weights2 = np.dot(self.hidden_layer_output.T, d_output)
        d_bias2 = np.sum(d_output, axis=0)
        d_hidden = np.dot(d_output, self.weights2.T) * (
            self.hidden_layer_output * (1 - self.hidden_layer_output)
        )
        d_weights1 = np.dot(x.T, d_hidden)
        d_bias1 = np.sum(d_hidden, axis=0)

        self.weights2 -= self.lr * d_weights2
        self.bias2 -= self.lr * d_bias2
        self.weights1 -= self.lr * d_weights1
        self.bias1 -= self.lr * d_bias1

    def save_model_params(self):
        np.savez(
            f"mlp_params_{self.hidden_size}.npz",
            weights1=self.weights1,
            bias1=self.bias1,
            weights2=self.weights2,
            bias2=self.bias2,
        )

    def load_model_params(self, params_path):
        params = np.load(params_path)
        self.weights1 = params["weights1"]
        self.bias1 = params["bias1"]
        self.weights2 = params["weights2"]
        self.bias2 = params["bias2"]

    def train(self, x, y):

        pred = self.forward(x)

        one_hot_y = np.zeros((len(y), self.output_size))
        one_hot_y[np.arange(len(y)), y] = 1
        loss = -np.mean(np.sum(one_hot_y * np.log(pred), axis=1))

        self.backward(x, y, pred)

        return loss, pred

    def save_losses_and_accuracies(self, losses, accuracies):
        np.save(f"mlp_losses_{self.hidden_size}.npy", losses)
        np.save(f"mlp_accuracies_{self.hidden_size}.npy", accuracies)


def main():
    train_loader, test_loader = load_data()

    input_size = 28 * 28
    hidden_size = 256
    output_size = 10
    lr = 0.01
    num_epochs = 100

    train_loss = []
    train_accuracy = []

    model = MLP(input_size, hidden_size, output_size, lr)
    for epoch in range(num_epochs):
        total_loss = 0
        correct_pred = 0
        total_pred = 0

        for inputs, labels in train_loader:
            x = inputs.view(-1, input_size).numpy()
            y = labels.numpy()
            loss, pred = model.train(x, y)
            total_loss += loss
            predicted_labels = np.argmax(pred, 1)
            correct_pred += np.sum(predicted_labels == y)
            total_pred += len(labels)

        train_loss.append(total_loss / len(train_loader))
        train_accuracy.append(correct_pred / total_pred)

        print(
            f"Epoch {epoch+1}/{num_epochs}, Train Loss: {train_loss[-1]:.4f} Train Accuracy: {train_accuracy[-1]:.4f}"
        )

    correct_pred = 0
    total_pred = 0
    for inputs, labels in test_loader:
        x = inputs.view(-1, input_size).numpy()
        y = labels.numpy()
        pred = model.forward(x)
        predicted_labels = np.argmax(pred, 1)
        correct_pred += np.sum(predicted_labels == y)
        total_pred += len(labels)
    print(f"Test Accuracy: {correct_pred/total_pred}")

    model.save_model_params()

    model.save_losses_and_accuracies(train_loss, train_accuracy)


if __name__ == "__main__":
    main()
