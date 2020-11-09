# aa-228-projects: Project 1 and 2 for Decision-Making Under Uncertainty

[AA228/CS238: Decision Making under Uncertainty](https://aa228.stanford.edu), Autumn 2020, Stanford University.

This repository provides my implemented code and data for Projects 1 and 2.

## Project 1: Bayesian Structure Learning

[![Project 1 Details](https://img.shields.io/badge/project1-details-blue)](https://web.stanford.edu/class/aa228/cgi-bin/wp/project-1/) [![Project 1 Template](https://img.shields.io/badge/project1-LaTeX%20template-white)](https://www.overleaf.com/read/hxwgtnksxtts)

    project1/
    ├── data                    # CSV data files to apply structured learning
    │   ├── small.csv               # Titanic dataset¹
    │   ├── medium.csv              # Wine dataset²
    │   └── large.csv               # Secret dataset
    ├── example                 # Helpful examples
    │   ├── example.gph             # Example graph (3 parents, 1 child each)
    │   ├── example.csv             # Example data generated from "example.gph"
    │   ├── example.score           # Bayesian score of the "examples.gph" given the data "examples.csv"
    │   ├── examples.pdf            # Visualized "examples.gph" as a TikZ graph
    │   └── titanicexample.pdf      # Simple example network using "small.csv"
    ├── project1.jl             # Starter code in Julia (optional, meant to help)
    └── project1.py             # Starter code in Python (optional, meant to help)

<sup>1</sup>https://cran.r-project.org/web/packages/titanic/titanic.pdf
<br>
<sup>2</sup>https://archive.ics.uci.edu/ml/datasets/Wine+Quality

#### Graph Plotting
Here are some resources for plotting graphs in Julia, Python, and MATLAB.
- Julia:
    - `TikzGraphs.jl` https://nbviewer.jupyter.org/github/JuliaTeX/TikzGraphs.jl/blob/master/doc/TikzGraphs.ipynb
    - `GraphPlot.jl` https://github.com/JuliaGraphs/GraphPlot.jl
- Python:
    - `NetworkX` https://networkx.github.io/documentation/stable/tutorial.html
- MATLAB:
    - `GraphPlot` https://www.mathworks.com/help/matlab/ref/matlab.graphics.chart.primitive.graphplot.html

## Project 2: Reinforcement Learning

[![Project 2 Details](https://img.shields.io/badge/project2-details-blue)](https://web.stanford.edu/class/aa228/cgi-bin/wp/project-2/) [![Project 2 Template](https://img.shields.io/badge/project2-LaTeX%20template-white)](https://www.overleaf.com/read/gsptsmcrzpdv)

    project2/
    └── data                      # CSV data files of (s,a,r,sp)
        ├── small.csv                 # 10x10 grid world
        ├── medium.csv                # MountainCarContinuous-v0
        └── large.csv                 # Secret RL problem

*Note: no starter code provided for Project 2.*
