import marimo

__generated_with = "0.19.8"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    import numpy as np
    import matplotlib.pyplot as plt

    return np, plt


@app.cell
def _():
    # https://copilot.microsoft.com/chats/uWtreD2NSRKW6NaHr76rv
    # https://realpython.com/python-ai-neural-network/
    return


@app.cell
def _(np, plt):

    # -----------------------------
    # 1. Training data (OR function)
    # -----------------------------
    X = np.array([
        [0, 0],
        [0, 1],
        [1, 0],
        [1, 1]
    ])

    y = np.array([[0], [1], [1], [1]])  # column vector

    # -----------------------------
    # 2. Hyperparameters
    # -----------------------------
    np.random.seed(42)
    learning_rate = 0.9
    epochs = 1000

    # -----------------------------
    # 3. Network architecture
    #    2 inputs → 2 hidden → 1 output
    # -----------------------------
    W1 = np.random.randn(2, 2)  # weights input → hidden
    b1 = np.random.randn(1, 2)  # bias for hidden layer

    W2 = np.random.randn(2, 1)  # weights hidden → output
    b2 = np.random.randn(1, 1)  # bias for output layer

    # -----------------------------
    # 4. Activation functions
    # -----------------------------
    def sigmoid(x):
        return 1 / (1 + np.exp(-x))

    def sigmoid_derivative(x):
        return x * (1 - x)  # derivative using output of sigmoid

    # -----------------------------
    # 5. Training loop
    # -----------------------------
    loss_history = []

    for epoch in range(epochs):

        # ---- Forward pass ----
        z1 = np.dot(X, W1) + b1
        a1 = sigmoid(z1)

        z2 = np.dot(a1, W2) + b2
        a2 = sigmoid(z2)  # final output

        # ---- Loss (MSE) ----
        loss = np.mean((y - a2) ** 2)
        loss_history.append(loss)

        # ---- Backpropagation ----
        # Output layer error
        error_output = a2 - y
        delta_output = error_output * sigmoid_derivative(a2)

        # Hidden layer error
        error_hidden = delta_output.dot(W2.T)
        delta_hidden = error_hidden * sigmoid_derivative(a1)

        # ---- Gradient descent update ----
        W2 -= learning_rate * a1.T.dot(delta_output)
        b2 -= learning_rate * np.sum(delta_output, axis=0, keepdims=True)

        W1 -= learning_rate * X.T.dot(delta_hidden)
        b1 -= learning_rate * np.sum(delta_hidden, axis=0, keepdims=True)

    # -----------------------------
    # 6. Results
    # -----------------------------
    print("Final predictions after training:")
    print(a2)

    # -----------------------------
    # 7. Plot loss curve
    # -----------------------------
    plt.plot(loss_history)
    plt.title("Training Loss (MSE)")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.grid(True)
    plt.show()

    return


if __name__ == "__main__":
    app.run()
