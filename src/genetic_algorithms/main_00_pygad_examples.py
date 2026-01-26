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
    and tries to simulate the learnings of the lecture "Expertensysteme in der elektrischen Energieversorgung".
    """)
    return


@app.cell
def _(mo, numpy, pygad):
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

    mo.md("## First Fitness Taste")
    return


@app.cell
def _(mo, numpy, pygad):
    np = numpy
    def optimize_production():
        margin = np.array([2, 2.5])
        material_consumption = np.array([2,3]) 
        material_max = 500
    
        def fitness_func(ga, solution, solution_idx):
          solution_added = solution + 50
          calculated_margin = np.sum(solution_added*margin)
          material_consumed = np.sum(solution_added*material_consumption)
      
          if material_consumed> material_max:
            return 0
          else:
            return calculated_margin
    
        # We check a possible solution
        print(f"Fitness for correct scenario: {fitness_func(None, np.array([0, 0]), '')}") 
        print(f"Fitness for incorrect scenario: {fitness_func(None, np.array([0, 1]), '')}") 

        ga_instance = pygad.GA(num_generations=1000,
                           sol_per_pop=10,
                           num_genes=2,
                           num_parents_mating=2,
                           fitness_func=fitness_func,
                           mutation_type="random",
                           mutation_probability=0.6
                           )

        ga_instance.run()

        print("Best solution:", ga_instance.best_solution()[0] + 50)
    
        ga_instance.plot_fitness()

    optimize_production()

    mo.md("## Optimize Production with Material Constraints\nThe generated plot is always changing (stochastic process), but it should show a steady increase in fitness over generations.")
    return


@app.cell
def _(numpy, pygad):
    def optimize_production2():  # Optimize production costs while meeting demands (CoPilot suggested)
        # Example data: production costs and demands
        production_costs = [20, 15, 30, 25]  # Cost per unit for each plant
        demands = [50, 60, 70, 80]  # Demand for each plant

        # Define the fitness function
        def fitness_func(ga, solution, solution_idx):
            total_cost = numpy.sum(solution * production_costs)
            total_demand = numpy.sum(solution)
            # Fitness is higher for lower costs and meeting demand
            fitness = 1.0 / (total_cost + abs(total_demand - sum(demands)) + 1e-6)
            return fitness

        # Define the parameters for the genetic algorithm
        num_generations = 200
        sol_per_pop = 20
        num_parents_mating = 5
        num_genes = len(production_costs)

        # Create an instance of the GA class
        ga_instance = pygad.GA(
            num_generations=num_generations,
            sol_per_pop=sol_per_pop,
            num_genes=num_genes,
            num_parents_mating=num_parents_mating,
            fitness_func=fitness_func,
            mutation_type="random",
            mutation_probability=0.3,
        )

        # Run the genetic algorithm
        ga_instance.run()

        # Fetch the best solution
        solution, solution_fitness, _ = ga_instance.best_solution()
        total_cost = numpy.sum(solution * production_costs)
        total_demand = numpy.sum(solution)
        print(f"Best solution: {solution}, Fitness: {solution_fitness}, Total Cost: {total_cost}, Total Demand: {total_demand}")

        fig = ga_instance.plot_fitness()

    optimize_production2()
    return


if __name__ == "__main__":
    app.run()
