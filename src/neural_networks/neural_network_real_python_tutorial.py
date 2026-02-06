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
        print(f"Manual dot product: {result_manual}")

        result_1 = np.dot(input_vector, weigths_1)
        print(f"Numpy dot product (weights 1): {result_1}")

        assert np.isclose(result_manual, result_1), "The manual dot product does not match the numpy dot product for weights 1."

        result_2 = np.dot(input_vector, weigths_2)
        print(f"Numpy dot product (weights 2): {result_2}")

    test_numpy_dot_product()
    return


if __name__ == "__main__":
    app.run()
