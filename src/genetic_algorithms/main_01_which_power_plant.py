import marimo

__generated_with = "0.19.6"
app = marimo.App(width="medium")


@app.cell
def _():
    import pygad
    import scipy
    import numpy as np
    import polars as pl
    import marimo as mo
    import altair as alt
    import seaborn as sns
    import matplotlib.pyplot as plt
    return alt, mo, np, pl, plt, pygad, scipy, sns


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Genetic algorithms in Python with PyGAD

    This notebook tries to simulate the learnings of the lecture "Expertensysteme in der elektrischen Energieversorgung" for genetic algorithms
    and takes some inspiration from
    [this blog post](https://anderfernandez.com/en/blog/genetic-algorithm-in-python/).
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Preliminary testing with simple functions

    Find the maximum of a simple quadratic function using a genetic algorithm.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    l_number_dimensions = ["one", "two", "three"]
    ui_number_dimensions = mo.ui.dropdown(
        label="Number of dimensions",
        options=l_number_dimensions,
        value=l_number_dimensions[1],
    )

    ui_min_x = mo.ui.number(
        label="Minimum x",
        value=40,
    )

    ui_max_x = mo.ui.number(
        label="Maximum x",
        value=240,
    )

    ui_min_y = mo.ui.number(
        label="Minimum y",
        value=1000,
    )

    ui_max_y = mo.ui.number(
        label="Maximum y",
        value=10000,
    )

    mo.vstack([
        mo.md("### Settings"),
        mo.hstack([
            ui_number_dimensions,
        ], justify="start"),
        mo.hstack([
            ui_min_x,
            ui_max_x,
        ], justify="start"),
        mo.hstack([
            ui_min_y,
            ui_max_y,
        ], justify="start"),
    ])
    return (
        l_number_dimensions,
        ui_max_x,
        ui_max_y,
        ui_min_x,
        ui_min_y,
        ui_number_dimensions,
    )


@app.cell
def _(l_number_dimensions, ui_max_x, ui_max_y, ui_min_x, ui_min_y):
    def analytic_function(x, dimension):
        """Return the analytic function value for the given dimension at value x."""
        if dimension == l_number_dimensions[0]:
            # parabola with maximum 8/10 (max - min) at x = 3/5 * (max - min)
            return -(x - 4/5 * (ui_max_x.value - ui_min_x.value)) ** 2 + 2 * 8/10 * (ui_max_y.value - ui_min_y.value)
        else:
            raise ValueError(f"Invalid dimension: '{dimension}'")
    return (analytic_function,)


@app.cell
def _(analytic_function, np):
    def calculate_analytic_values(x_min, x_max, steps, dimension):
        """Calculate the analytic function values for the given dimension."""
        x_values = np.linspace(x_min, x_max, steps)
        y_values = analytic_function(x_values, dimension)
        return x_values, y_values
    return (calculate_analytic_values,)


@app.cell
def _(
    calculate_analytic_values,
    l_number_dimensions,
    plt,
    ui_max_x,
    ui_min_x,
    ui_number_dimensions,
):
    def plot_analytic_function(dimension):
        """Plot the analytic function for the given dimension."""
        if dimension == l_number_dimensions[0]:
            x, y = calculate_analytic_values(ui_min_x.value, ui_max_x.value, 100, dimension)
            plt.plot(x, y)
            plt.title("Analytic Function (1D)")
            plt.xlabel("x")
            plt.ylabel("f(x)")
            plt.grid()
            plt.show()
        else:
            raise ValueError(f"Invalid dimension: '{dimension}'")

    plot_analytic_function(ui_number_dimensions.value)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Cost functions of power plants

    I don't have the exact cost functions used in the lecture, but here is an image non-linear cost functions
    """)
    return


@app.cell
def _(mo):
    mo.image(src="data/20260126_KostenKraftwerk.png")
    return


@app.cell
def _():
    categories = ["BlockA", "BlockB", "BlockC"]
    return (categories,)


@app.cell
def _(categories, pl):
    # read the data in cost_a.csv cost_b.csv cost_c.csv files then add for each thier category "BlockA", "BlockB", "BlockC" and combine the data into a single polars dataframe
    _cost_a = pl.read_csv("data/cost_a.csv").with_columns(pl.lit(categories[0]).alias("category"))
    _cost_b = pl.read_csv("data/cost_b.csv").with_columns(pl.lit(categories[1]).alias("category"))
    _cost_c = pl.read_csv("data/cost_c.csv").with_columns(pl.lit(categories[2]).alias("category"))
    costs = pl.concat([_cost_a, _cost_b, _cost_c])
    costs
    return (costs,)


@app.cell
def _(costs, plt, sns):
    def plot_costs(df):
        fig, ax = plt.subplots()
        sns.lineplot(
            df,
            x="power",
            y="cost",
            hue="category",
            ax=ax,
        )
        ax.set_title("Cost functions of power plants")
        ax.set_xlabel("Power [MW]")
        ax.set_ylabel("Cost [€/h]")

        return fig

    plot_costs(costs)
    return


@app.cell(disabled=True)
def _(alt, costs, mo):
    chart = mo.ui.altair_chart(alt.Chart(costs).mark_point().encode(
        x='power',
        y='cost',
        color='category'
    ))
    chart
    return


@app.cell
def _(costs, np, pl):
    def get_interpolated_cost(power: float, category: str):
        """Return the power plant cost for a given power and category interpolated from the costs dataframe."""
        df = costs.filter(pl.col("category") == category).to_pandas()
        return np.interp(power, df["power"], df["cost"])
    return (get_interpolated_cost,)


@app.cell
def _():
    demand = 470  # Example demand in MW
    return (demand,)


@app.cell
def _(categories, demand, get_interpolated_cost, np):
    def calculate_total_power(solution):
        return np.sum(solution)# * 3 * 240 # Assuming each plant can generate up to 240 MW)

    def calculate_total_cost(solution):
        total_cost = 0
        # Calculate total cost based on the power generated by each plant
        for i, sol in enumerate(solution):
            category = categories[i]
            cost = get_interpolated_cost(sol, category)
            total_cost += cost
        return total_cost

    def fitness_func(ga, solution, solution_idx):
        """Fitness function to minimize the cost of power generation while meeting demand."""
        total_power = calculate_total_power(solution)
        total_cost = calculate_total_cost(solution)

        # Penalize solutions that do not meet the demand
        if total_power < demand:
            total_cost += (demand - total_power) * 10000  # High penalty for not meeting demand

        # Penalize solutions with negative power
        if np.any(solution < 0):
            total_cost += 1e8  # High penalty for negative power

        # Since PyGAD maximizes fitness, we return the negative cost
        return -total_cost
    return calculate_total_cost, calculate_total_power, fitness_func


@app.cell
def _(calculate_total_cost, calculate_total_power, fitness_func, pygad):
    def run_genetic_algorithm():  # takes 9 seconds with 500 generations and 10 solutions per population

        # Define the parameters for the genetic algorithm
        num_generations = 500
        sol_per_pop = 10
        num_parents_mating = 2
        num_genes = 3  # Number of power plants
        #gene_space = range(0, 101)  # Genes can take values from 0 to 100

        # Create an instance of the GA class
        ga_instance = pygad.GA(
            num_generations=num_generations,
            sol_per_pop=sol_per_pop,
            num_genes=num_genes,
            num_parents_mating=num_parents_mating,
            fitness_func=fitness_func,
            mutation_type="random",
            mutation_probability=0.6,
            #init_range_low=-1,
            #init_range_high=200,
            #gene_space=gene_space,
        )

        # Run the genetic algorithm
        ga_instance.run()

        # Fetch the best solution
        solution, solution_fitness, _ = ga_instance.best_solution()
        output = calculate_total_power(solution)
        final_cost = calculate_total_cost(solution)
        print(f"Best solution: {solution}, Fitness: {solution_fitness}, Output: {output} Cost: {final_cost}")

        fig = ga_instance.plot_fitness()

    run_genetic_algorithm()
    return


@app.cell
def _(fitness_func, scipy):
    def find_optimum_brute():
        def objective(x):
            return -fitness_func(None, x, None)

        # Define the ranges for each power plant
        ranges = (slice(0, 240, 8), slice(0, 240, 8), slice(0, 240, 8))
        # 24 ** 3 = 13824 combinations to evaluate takes 21.5 seconds

        # Perform brute-force optimization
        resbrute = scipy.optimize.brute(objective, ranges, finish=None, full_output=True)
        #total_power = calculate_total_power(resbrute[0])
        #total_cost = calculate_total_cost(resbrute[0])
        #assert total_cost == resbrute[1]

        #print(f"Brute-force result: {resbrute}, Total Power: {total_power}, Total Cost: {total_cost}")
        print(resbrute)
        return resbrute

    resbrute = find_optimum_brute()
    resbrute
    return


@app.cell
def _(fitness_func, np):
    -fitness_func(None, np.array([144.16177874, 163.73550704, 162.36926992]), None)
    return


if __name__ == "__main__":
    app.run()
