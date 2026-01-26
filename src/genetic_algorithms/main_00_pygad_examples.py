import marimo

__generated_with = "0.19.6"
app = marimo.App(width="medium")


@app.cell
def _():
    import pygad
    import numpy
    import marimo as mo
    return mo, numpy, pygad


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Genetic algorithms in Python with PyGAD

    This notebook is heavily inspired by
    [this blog post](https://anderfernandez.com/en/blog/genetic-algorithm-in-python/)
    and tries to simulate the learning of the lecture "Expertensysteme in der elektrischen Energieversorgung".
    """)
    return


@app.cell
def _(numpy, pygad):
    def first_test():
        inputs = [0.4, 1, 0, 7, 8]
        desired_output = 32
    
        # Define the fitness function
        def fitness_func(ga, solution, solution_idx):
            output = numpy.sum(solution*inputs)
            fitness = 1.0 / (numpy.abs(output - desired_output) + 1e-6)  # Avoid division by zero
            return fitness

        # Define the parameters for the genetic algorithm
        num_generations = 100
        sol_per_pop = 10
        num_parents_mating = 2
        num_genes = len(inputs)
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
            #gene_space=gene_space,
        )

        # Run the genetic algorithm
        ga_instance.run()

        # Fetch the best solution
        solution, solution_fitness, _ = ga_instance.best_solution()
        output = numpy.sum(solution*inputs)
        print(f"Best solution: {solution}, Fitness: {solution_fitness}, Output: {output}")

        fig = ga_instance.plot_fitness()
    

    first_test()
    return


if __name__ == "__main__":
    app.run()
