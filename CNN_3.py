import numpy as np
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

# ===================== Utility Functions ===================== #


def relu(x):
    return np.maximum(0, x)


def softmax(x):
    e_x = np.exp(x - np.max(x, axis=1, keepdims=True))
    return e_x / e_x.sum(axis=1, keepdims=True)


# ===================== Data Loading ===================== #
def dataloader(train_dataset, test_dataset, batch_size=64):
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

    # train_dataset = Subset(train_dataset, range(10))
    # test_dataset = Subset(test_dataset, range(10))
    print("Training samples:", len(train_dataset))
    print("Testing samples:", len(test_dataset))
    return dataloader(train_dataset, test_dataset)


# ===================== CNN Structure ===================== #
class CNN:
    def __init__(
        self, input_size, num_filters, kernel_size, fc_output_size, lr, params_path=None
    ):
        self.lr = lr
        self.num_filters = num_filters
        self.kernel_size = kernel_size
        self.fc_output_size = fc_output_size

        # Initialize filters and FC layer
        if params_path is not None:
            self.load_model_params(params_path)
        else:
            self.filters = np.random.randn(
                num_filters, 1, kernel_size, kernel_size
            ) * np.sqrt(2.0 / (kernel_size * kernel_size))
            conv_output_dim = input_size - kernel_size + 1  # no padding
            flatten_size = num_filters * conv_output_dim * conv_output_dim

            self.fc_weights = np.random.randn(flatten_size, fc_output_size) * np.sqrt(
                2.0 / flatten_size
            )
            self.fc_bias = np.zeros((1, fc_output_size))

    def convolve(self, x, filters):
        B, _, H, W = x.shape
        F, _, KH, KW = filters.shape
        OH, OW = H - KH + 1, W - KW + 1

        # Extract patches (B, OH, OW, KH, KW)
        patches = np.lib.stride_tricks.as_strided(
            x,
            shape=(B, OH, OW, KH, KW),
            strides=(
                x.strides[0],
                x.strides[2],
                x.strides[3],
                x.strides[2],
                x.strides[3],
            ),
            writeable=False,
        )

        # Reshape: (B, OH*OW, KH*KW)
        patches = patches.reshape(B, OH * OW, KH * KW)

        # Reshape filters: (F, KH*KW)
        filters_flat = filters.reshape(F, KH * KW)

        # Matrix multiply: (B, OH*OW, F)
        out = np.matmul(patches, filters_flat.T)

        # Reshape back: (B, F, OH, OW)
        out = out.transpose(0, 2, 1).reshape(B, F, OH, OW)
        return out

    def forward(self, x):
        """Forward propagation"""
        self.x = x
        self.conv = self.convolve(x, self.filters)  # (B, F, OH, OW)
        self.relu = relu(self.conv)
        self.flat = self.relu.reshape(x.shape[0], -1)
        self.logits = np.dot(self.flat, self.fc_weights) + self.fc_bias
        self.probs = softmax(self.logits)
        return self.probs

    def backward(self, x, y, pred):
        """Backward propagation"""
        B = y.shape[0]
        _, _, OH, OW = self.conv.shape
        KH, KW = self.kernel_size, self.kernel_size

        # 1. one-hot encode the labels
        y_onehot = np.eye(self.fc_output_size)[y]

        # 2. Calculate softmax cross-entropy loss gradient
        d_logits = (pred - y_onehot) / B

        # 3. Fully connected layer gradients
        d_fc_weights = np.dot(self.flat.T, d_logits)
        d_fc_bias = np.sum(d_logits, axis=0, keepdims=True)
        d_flat = np.dot(d_logits, self.fc_weights.T)
        d_relu = d_flat.reshape(self.relu.shape)

        # 4. Backpropagate through ReLU
        d_conv = d_relu * (self.conv > 0)

        # 5. Compute convolution gradients
        patches = np.lib.stride_tricks.as_strided(
            self.x,
            shape=(B, OH, OW, KH, KW),
            strides=(
                self.x.strides[0],
                self.x.strides[2],
                self.x.strides[3],
                self.x.strides[2],
                self.x.strides[3],
            ),
            writeable=False,
        ).reshape(
            B, OH * OW, KH * KW
        )  # (B, OH*OW, KH*KW)

        d_out_flat = d_conv.reshape(B, self.num_filters, OH * OW)  # (B, F, OH*OW)

        d_filters = np.zeros_like(self.filters)
        for b in range(B):
            d_filters += np.matmul(d_out_flat[b], patches[b]).reshape(
                self.num_filters, 1, KH, KW
            )

        # 6. Update parameters
        self.fc_weights -= self.lr * d_fc_weights
        self.fc_bias -= self.lr * d_fc_bias
        self.filters -= self.lr * d_filters

    def forward_old(self, x):
        """Forward propagation"""
        self.x = x
        self.conv = self.convolve(x, self.filters)
        self.relu = relu(self.conv)
        self.flat = self.relu.reshape(x.shape[0], -1)
        self.logits = np.dot(self.flat, self.fc_weights) + self.fc_bias
        self.probs = softmax(self.logits)
        return self.probs

    def convolve_old(self, x, filters):
        B, _, H, W = x.shape
        F, _, KH, KW = filters.shape
        OH, OW = H - KH + 1, W - KW + 1
        output = np.zeros((B, F, OH, OW))
        for b in range(B):
            for f in range(F):
                for i in range(OH):
                    for j in range(OW):
                        output[b, f, i, j] = np.sum(
                            x[b, 0, i : i + KH, j : j + KW] * filters[f, 0]
                        )
        return output

    def backward_old(self, x, y, pred):
        """Backward propagation"""
        B = y.shape[0]

        # 1. one-hot encode the labels
        y_onehot = np.eye(self.fc_output_size)[y]

        # 2. Calculate softmax cross-entropy loss gradient
        d_logits = (pred - y_onehot) / B

        # 3. Calculate fully connected layer gradient
        d_fc_weights = np.dot(self.flat.T, d_logits)
        d_fc_bias = np.sum(d_logits, axis=0, keepdims=True)
        d_flat = np.dot(d_logits, self.fc_weights.T)
        d_relu = d_flat.reshape(self.relu.shape)

        # 4. Backpropagate through ReLU
        d_conv = d_relu * (self.conv > 0)

        # 5. Calculate convolution kernel gradient
        d_filters = np.zeros_like(self.filters)
        for b in range(B):
            for f in range(self.num_filters):
                for i in range(d_conv.shape[2]):
                    for j in range(d_conv.shape[3]):
                        d_filters[f, 0] += (
                            d_conv[b, f, i, j]
                            * self.x[
                                b, 0, i : i + self.kernel_size, j : j + self.kernel_size
                            ]
                        )

        # 6. Update parameters
        self.fc_weights -= self.lr * d_fc_weights
        self.fc_bias -= self.lr * d_fc_bias
        self.filters -= self.lr * d_filters

    def train(self, x, y):
        pred = self.forward(x)
        loss = -np.mean(np.log(pred[np.arange(len(y)), y] + 1e-9))
        self.backward(x, y, pred)
        return loss, pred

    def save_losses_and_accuracies(self, losses, accuracies):
        np.save(f"cnn_losses_{self.kernel_size}.npy", losses)
        np.save(f"cnn_accuracies_{self.kernel_size}.npy", accuracies)

    def save_model_params(self):
        np.savez(
            f"cnn_params_{self.kernel_size}.npz",
            filters=self.filters,
            fc_weights=self.fc_weights,
            fc_bias=self.fc_bias,
        )

    def load_model_params(self, params_path):
        params = np.load(params_path)
        self.filters = params["filters"]
        self.fc_weights = params["fc_weights"]
        self.fc_bias = params["fc_bias"]


# ===================== Training Process ===================== #
def main():
    # First, load data
    train_loader, test_loader = load_data()

    # Second, define hyperparameters
    input_size = 28
    num_epochs = 5
    num_filters = 1
    kernel_size = 7
    fc_output_size = 10
    lr = 0.01

    model = CNN(input_size, num_filters, kernel_size, fc_output_size, lr)

    train_loss = []
    train_accuracy = []
    # Then, train the model
    for epoch in range(num_epochs):
        total_loss = 0
        correct_pred = 0
        total_pred = 0
        for inputs, labels in train_loader:  # define training phase
            x = inputs.numpy()
            y = labels.numpy()
            loss, pred = model.train(x, y)
            predicted_labels = np.argmax(pred, 1)
            correct_pred += np.sum(predicted_labels == y)
            total_pred += len(labels)
            total_loss += loss

        train_loss.append(total_loss / len(train_loader))
        train_accuracy.append(correct_pred / total_pred)

        print(
            f"Epoch {epoch+1}/{num_epochs}, Train Loss: {train_loss[-1]:.4f} Train Accuracy: {train_accuracy[-1]:.4f}"
        )  # print the loss for each epoch

    # Finally, evaluate the model
    correct_pred = 0
    total_pred = 0
    for inputs, labels in test_loader:
        x = inputs.numpy()
        x = x.reshape(-1, 1, 28, 28)
        y = labels.numpy()
        pred = model.forward(x)  # trained model
        predicted_labels = np.argmax(pred, axis=1)
        correct_pred += np.sum(predicted_labels == y)
        total_pred += len(labels)
    print(f"Test Accuracy: {correct_pred/total_pred:.4f}")

    model.save_model_params()

    model.save_losses_and_accuracies(train_loss, train_accuracy)


if __name__ == "__main__":
    main()
