import marimo

__generated_with = "0.19.8"
app = marimo.App()


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Build Model
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Imports
    """)
    return


@app.cell
def _():
    import numpy as np
    import marimo as mo
    import polars as pl
    import matplotlib.pyplot as plt

    return mo, np, pl, plt


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Neuron definition
    """)
    return


@app.cell
def _(np):
    class Neuron:
        def __init__(self, num_inputs):
            self.weights = np.random.uniform(size=num_inputs)
            self.bias = np.random.uniform()

        def activate(self, inputs):
            self.inputs = inputs
            self.output = self.sigmoid(np.dot(inputs, self.weights) + self.bias)
            return self.output

        def sigmoid(self, x):
            return 1 / (1 + np.exp(-x))

        def sigmoid_derivative(self):
            return self.output * (1 - self.output)

        def update_weights(self, delta, learning_rate):
            self.weights += learning_rate * delta * self.inputs
            self.bias += learning_rate * delta

    return (Neuron,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Layer definition
    """)
    return


@app.cell
def _(Neuron, np):
    class Layer:
        def __init__(self, num_neurons, num_inputs_per_neuron):
            self.neurons = [Neuron(num_inputs_per_neuron) for _ in range(num_neurons)]

        def forward(self, inputs):
            return np.array([neuron.activate(inputs) for neuron in self.neurons])

        def backward(self, errors, learning_rate):
            deltas = []
            for i, neuron in enumerate(self.neurons):
                delta = errors[i] * neuron.sigmoid_derivative()
                neuron.update_weights(delta, learning_rate)
                deltas.append(delta)
            return np.dot(np.array([neuron.weights for neuron in self.neurons]).T, deltas)

    return (Layer,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Neural Network definition
    """)
    return


@app.cell
def _(Layer, np):
    class NeuralNetwork:
        def __init__(self, layers, learning_rate=0.1, epochs=10000):
            self.learning_rate = learning_rate
            self.epochs = epochs
            self.layers = []

            # Initialize layers
            for i in range(len(layers) - 1):
                self.layers.append(Layer(layers[i+1], layers[i]))

        def train(self, inputs, outputs):
            for epoch in range(self.epochs):
                total_error = 0
                for x, y in zip(inputs, outputs):
                    # Forward pass
                    activations = [x]
                    for layer in self.layers:
                        activations.append(layer.forward(activations[-1]))

                    # Calculate error (MSE)
                    output_errors = y - activations[-1]
                    total_error += np.sum(output_errors ** 2)

                    # Backward pass
                    errors = output_errors
                    for i in reversed(range(len(self.layers))):
                        errors = self.layers[i].backward(errors, self.learning_rate)

                # Print MSE every 1000 epochs
                if epoch % 1000 == 0:
                    mse = total_error / len(inputs)
                    print(f'Epoch {epoch}, MSE: {mse}')

        def predict(self, inputs):
            activations = inputs
            for layer in self.layers:
                activations = layer.forward(activations)
            return activations

    return (NeuralNetwork,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Initialization and Training
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Training Data preperation
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    ui_dropdown_operation = mo.ui.dropdown(
        options=["AND", "OR", "XOR"], value="XOR", label="Choose operation:"
    )
    ui_dropdown_operation
    return (ui_dropdown_operation,)


@app.cell
def _(np, ui_dropdown_operation):
    target_operation = ui_dropdown_operation.value
    if target_operation == "XOR":
        inputs = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
        outputs = np.array([[0], [1], [1], [0]])
    elif target_operation == "AND":
        inputs = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
        outputs = np.array([[0], [0], [0], [1]])
    elif target_operation == "OR":
        inputs = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
        outputs = np.array([[0], [1], [1], [1]])
    return inputs, outputs


@app.cell(hide_code=True)
def _(inputs, outputs, pl):
    # Visualize inputs and output using a dataframe
    df_input = pl.DataFrame({
        "Input A": inputs[:, 0],
        "Input B": inputs[:, 1],
        "Output": outputs.ravel()
    })
    df_input
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Train model
    """)
    return


@app.cell
def _(NeuralNetwork, inputs, outputs):
    # Initialize the neural network with an input layer, one hidden layer, and an output layer
    layers = [2,2,1]  # 2 input neurons, 2 neurons in hidden layer, 1 output neuron
    nn = NeuralNetwork(layers, 0.1, 10000)

    # Train the neural network
    nn.train(inputs, outputs)
    return (nn,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Evaluation
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Simple test
    """)
    return


@app.cell
def _(inputs, nn, np):
    # Test the neural network
    predicted_output = np.array([nn.predict(x) for x in inputs])
    print("Predicted Output:\n", predicted_output)
    return (predicted_output,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Visualization
    """)
    return


@app.cell
def _(inputs, nn, np, outputs, plt, predicted_output):
    # Round the predicted output to get binary predictions
    predicted_output_binary = np.round(predicted_output)

    # Plot the decision boundary
    x_min, x_max = inputs[:, 0].min() - 0.5, inputs[:, 0].max() + 0.5
    y_min, y_max = inputs[:, 1].min() - 0.5, inputs[:, 1].max() + 0.5
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.1),
                         np.arange(y_min, y_max, 0.1))
    Z = np.array([nn.predict(np.array([x, y])) for x, y in zip(xx.ravel(), yy.ravel())])
    Z = np.round(Z.reshape(xx.shape))

    plt.contourf(xx, yy, Z, cmap=plt.cm.Spectral)
    plt.scatter(inputs[:, 0], inputs[:, 1], c=outputs.ravel(), cmap=plt.cm.Spectral)
    plt.title("XOR Neural Network Decision Boundary")
    plt.xlabel("Input 1")
    plt.ylabel("Input 2")
    plt.show()

    # Print the accuracy
    accuracy = np.mean(predicted_output_binary.ravel() == outputs.ravel())
    print("Accuracy:", accuracy)
    return


if __name__ == "__main__":
    app.run()
