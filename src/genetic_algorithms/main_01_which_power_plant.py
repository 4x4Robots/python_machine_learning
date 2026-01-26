import marimo

__generated_with = "0.19.6"
app = marimo.App(width="medium")


@app.cell
def _():
    import pygad
    import numpy as np
    import polars as pl
    import marimo as mo
    import altair as alt
    import seaborn as sns
    import matplotlib.pyplot as plt
    return alt, mo, np, pl, plt, sns


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Genetic algorithms in Python with PyGAD

    This notebook is heavily inspired by
    [this blog post](https://anderfernandez.com/en/blog/genetic-algorithm-in-python/)
    and tries to simulate the learnings of the lecture "Expertensysteme in der elektrischen Energieversorgung".
    """)
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
def _(pl):
    # read the data in cost_a.csv cost_b.csv cost_c.csv files then add for each thier category "BlockA", "BlockB", "BlockC" and combine the data into a single polars dataframe
    _cost_a = pl.read_csv("data/cost_a.csv").with_columns(pl.lit("BlockA").alias("category"))
    _cost_b = pl.read_csv("data/cost_b.csv").with_columns(pl.lit("BlockB").alias("category"))
    _cost_c = pl.read_csv("data/cost_c.csv").with_columns(pl.lit("BlockC").alias("category"))
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


@app.cell
def _(costs, np):
    # interpolate the cost functions with numpy polyfit of degree 3 for each category
    def interpolate_costs():
        cost_functions = {}
        # use polars group_by to group by category
        for category, group in costs.group_by("category"):
            coeffs = np.polyfit(group["power"], group["cost"], deg=3)
            cost_functions[category] = np.poly1d(coeffs)
        return cost_functions

    interpolate_costs()
    return


@app.cell
def _(costs, mo, plt):
    # plot the chart with matplotlib
    fig, ax = plt.subplots()
    for category, group in costs.groupby("category"):
        ax.plot(group["power"], group["cost"], label=category)
    ax.set_xlabel("P [MW]")
    ax.set_ylabel("Kosten [€/h]")
    ax.legend()
    plt.show()
    mo.image(fig=fig)
    return


@app.cell
def _(alt, costs):
    _chart = (
        alt.Chart(costs)
        .mark_line()
        .encode(
            x=alt.X(field='power', type='quantitative', title='P [MW]'),
            y=alt.Y(field='cost', type='quantitative', title='Kosten [€/h]', aggregate='mean'),
            color=alt.Color(field='category', type='nominal'),
            tooltip=[
                alt.Tooltip(field='power', format=',.0f', title='P [MW]'),
                alt.Tooltip(field='cost', aggregate='mean', format=',.2f', title='Kosten [€/h]'),
                alt.Tooltip(field='category')
            ]
        )
        .properties(
            height=290,
            width='container',
            config={
                'axis': {
                    'grid': False
                }
            }
        )
    )
    _chart
    return


@app.cell
def _(alt, costs, mo):
    chart = mo.ui.altair_chart(alt.Chart(costs).mark_point().encode(
        x='power',
        y='cost',
        color='category'
    ))
    chart
    return


if __name__ == "__main__":
    app.run()
