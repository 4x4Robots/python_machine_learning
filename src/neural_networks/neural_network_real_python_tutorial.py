import marimo

__generated_with = "0.19.8"
app = marimo.App(width="medium")


@app.cell
def _():
    import numpy as np
    import marimo as mo
    import matplotlib as plt

    return mo, np


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Neural Networks in Python

    Reference: https://realpython.com/python-ai-neural-network/
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Linear regression

    Linear regression uses the **weighted sum** of independent variables and a bias, which sets the result when all other independent variables are zero.
    Real world example to predict house price:

    $$
    \text{price} = (w_1 \cdot \text{area}) + (w_2 \cdot \text{age}) + b
    $$
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Understanding the dot product

    Given two vectors $\mathbf{a} = [a_1, a_2, a_3]$ and $\mathbf{b} = [b_1, b_2, b_3]$, the dot product (in other contexts inner product) is calculated as:

    $$
    \mathbf{a} \cdot \mathbf{b} = (a_1 \cdot b_1) + (a_2 \cdot b_2) + (a_3 \cdot b_3) = \sum_{i=1}^{n} a_i b_i
    $$
    """)
    return


@app.cell
def _(np):
    def calculate_2n_dot_product(a, b):
        first_indices_multiplied = a[0] * b[0]
        second_indices_multiplied = a[1] * b[1]
        dot_product = first_indices_multiplied + second_indices_multiplied
        return dot_product
        #return np.dot(a, b)

    def test_numpy_dot_product():
        input_vector = [1.72, 1.23]
        weigths_1 = [1.26, 0]
        weigths_2 = [2.17, 0.32]

        result_manual = calculate_2n_dot_product(input_vector, weigths_1)
        print(f"Manual dot product (weights 1): {result_manual}")

        result_1 = np.dot(input_vector, weigths_1)
        print(f"Numpy dot product  (weights 1): {result_1}")

        assert np.isclose(result_manual, result_1), "The manual dot product does not match the numpy dot product for weights 1."

        result_2 = np.dot(input_vector, weigths_2)
        print(f"Numpy dot product  (weights 2): {result_2}")

    test_numpy_dot_product()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Making our first prediction

    > Since this is your very first neural network, you’ll keep things straightforward and build a network with only two layers. So far, you’ve seen that the only two operations used inside the neural network were the dot product and a sum. Both are **linear operations**.
    >
    > If you add more layers but keep using only linear operations, then adding more layers would have no effect because each layer will always have some correlation with the input of the previous layer. This implies that, for a network with multiple layers, there would always be a network with fewer layers that predicts the same results.
    >
    > What you want is to find an operation that makes the middle layers sometimes correlate with an input and sometimes not correlate.
    >
    > You can achieve this behavior by using nonlinear functions. These nonlinear functions are called activation functions. There are many types of **activation functions**. The [ReLU (rectified linear unit)](https://en.wikipedia.org/wiki/Rectifier_(neural_networks)), for example, is a function that converts all negative numbers to zero. This means that the network can “turn off” a weight if it’s negative, adding nonlinearity.
    >
    > The network you’re building will use the [sigmoid activation function](https://en.wikipedia.org/wiki/Sigmoid_function). You’ll use it in the last layer, layer_2. The only two possible outputs in the dataset are 0 and 1, and the sigmoid function limits the output to a range between 0 and 1. This is the formula to express the sigmoid function:
    >
    >$$
    S(x) = \frac{1}{1 + e^{-x}}
    $$

    excerpt from: https://realpython.com/python-ai-neural-network/ (06.02.2026)
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Classification problem

    A subset of supervised learning problems in which you have a dataset with the inputs and the known targets.
    Here are the inputs and the outputs of the dataset:

    | input vector | target |
    | --- | --- |
    | [1.66, 1.56] | 1 |
    | [2, 1.5] | 0 |
    """)
    return


@app.cell
def _(np):
    def sigmoid(x):
        return 1 / (1 + np.exp(-x))
    
    def make_prediction(input_vector, weights, bias):
        layer_1 = np.dot(input_vector, weights) + bias
        #print(f"Layer 1 output: {layer_1}")
        layer_2 = sigmoid(layer_1)
        #print(f"Layer 2 output (after sigmoid): {layer_2}")
        return layer_2

    def first_prediction(input_vector):
        print(f"Performing first prediction for input vector: {input_vector}") 
        # Wrapping the vectors in NumPy arrays
        #input_vector = np.array([1.66, 1.56])
        weights_1 = np.array([1.45, -0.66])
        bias = np.array([0.0])

        prediction = make_prediction(input_vector, weights_1, bias)
        print(f"The prediction result is: {prediction}")
        return prediction

    if first_prediction(np.array([1.66, 1.56])) > 0.5:
        print("The predicted class is: 1 which is correct")

    print("-------------")
    if first_prediction(np.array([2, 1.5])) < 0.5:
        print("The predicted class is: 0 which is correct")
    else:
        print("The predicted class is: 1 which is **incorrect**")
    return first_prediction, make_prediction, sigmoid


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Computing the prediction error

    Use the [mean squared error (MSE)](https://en.wikipedia.org/wiki/Mean_squared_error) to compute the error of the prediction. Use this as the **cost function** or **loss function**. The MSE is calculated as:

    $$
    \text{MSE} = \frac{1}{n} \sum_{i=1}^{n} (y_i - \hat{y}_i)^2
    $$
    """)
    return


@app.cell
def _(first_prediction, np):
    def calculate_mse():
        prediction = first_prediction(np.array([2, 1.5]))
        target = 0
        mse = np.square(prediction - target)

        print(f"Prediction: {prediction}, Target: {target}, MSE: {mse}")

    calculate_mse()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Reducing the error with gradient descent

    Use the derivative of the cost function to find the direction in which to adjust the weights and bias to reduce the error. This process is called **gradient descent**.
    """)
    return


@app.cell
def _(make_prediction, np, sigmoid):
    def sigmoid_derivative(x):
        return sigmoid(x) * (1 - sigmoid(x))

    def calculate_gradient_descent():
        input_vector = np.array([2, 1.5])

        weights_1 = np.array([1.45, -0.66])
        bias = np.array([0.0])
        print(f"Initial weights: {weights_1}, bias: {bias}")
        prediction = make_prediction(input_vector, weights_1, bias)
        target = 0
        error = np.square(prediction - target)
        print(f"Prediction: {prediction} {'==' if (prediction > 0.5) == target else '!='} Target: {target} with an Error: {error}")

        # d/dx mse = d/dx (prediction - target)^2
        derivative = 2 * (prediction - target)

        print(f"Derivative of MSE is: {derivative}")

        # Updating the weights
        weights_1 = weights_1 - derivative
        print(f"Updated weights: {weights_1}")

        prediction = make_prediction(input_vector, weights_1, bias)
        error = np.square(prediction - target)
        print(f"Prediction: {prediction} {'==' if (prediction > 0.5) == target else '!='} Target: {target} with an Error: {error}")

        derror_dprediction = 2 * (prediction - target)
        layer_1 = np.dot(input_vector, weights_1) + bias
        dprediction_dlayer1 = sigmoid_derivative(layer_1)
        dlayer1_dbias = 1

        derivative_bias = derror_dprediction * dprediction_dlayer1 * dlayer1_dbias
        print(f"Derivative with respect to bias: {derivative_bias}")
    

    calculate_gradient_descent()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Neural Network Class

    Combining everything into one class.
    """)
    return


@app.cell
def _(np):
    class NeuralNetwork:
        def __init__(self, learning_rate):
            self.weights = np.array([np.random.randn(), np.random.randn()])
            self.bias = np.random.randn()
            self.learning_rate = learning_rate

        def _sigmoid(self, x):
            return 1 / (1 + np.exp(-x))

        def _sigmoid_deriv(self, x):
            return self._sigmoid(x) * (1 - self._sigmoid(x))

        def predict(self, input_vector):
            layer_1 = np.dot(input_vector, self.weights) + self.bias
            layer_2 = self._sigmoid(layer_1)
            prediction = layer_2
            return prediction

        def _compute_gradients(self, input_vector, target):
            layer_1 = np.dot(input_vector, self.weights) + self.bias
            layer_2 = self._sigmoid(layer_1)
            prediction = layer_2

            derror_dprediction = 2 * (prediction - target)
            dprediction_dlayer1 = self._sigmoid_deriv(layer_1)
            dlayer1_dbias = 1
            dlayer1_dweights = (0 * self.weights) + (1 * input_vector)

            derror_dbias = derror_dprediction * dprediction_dlayer1 * dlayer1_dbias
            derror_dweights = derror_dprediction * dprediction_dlayer1 * dlayer1_dweights

            return derror_dbias, derror_dweights

        def _update_parameters(self, derror_dbias, derror_dweights):
            self.bias = self.bias - (derror_dbias * self.learning_rate)
            self.weights = self.weights - (derror_dweights * self.learning_rate)

    return (NeuralNetwork,)


@app.cell
def _(mo):
    ui_learning_rate = mo.ui.number(label="Learning Rate", value=0.1, step=0.01)
    ui_learning_rate
    return (ui_learning_rate,)


@app.cell
def _(NeuralNetwork, mo, np, ui_learning_rate):
    def neural_network_class_first_prediction():
        neural_network = NeuralNetwork(learning_rate=ui_learning_rate.value)

        return neural_network.predict(np.array([2, 1.5]))

    mo.vstack([
        mo.md("### First Prediction using Neural Network class\nDue to the random initialisation the prediction always changes."),
        neural_network_class_first_prediction()
    ])
    return


if __name__ == "__main__":
    app.run()
