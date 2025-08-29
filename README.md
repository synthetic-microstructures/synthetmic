# SynthetMic
A Python package for generating synthetic polycrystalline microstructures using Laguerre diagrams, powered by [pysdot](https://github.com/sd-ot/pysdot).

## Installation
To install the latest version of the package via `pip`, run
```
pip install synthetmic
```
> If you are using `uv` to manage your project, run the following command instead:
>
> uv add synthetmic

## Usage
To use this package to generate synthetic microstructures, you need to import the generator class as follows:
```python
from synthetmic import LaguerreDiagramGenerator
```

Create an instance of the class with the default arguments:
```python
generator = LaguerreDiagramGenerator()
```
or with custom parameters:
```python
generator = LaguerreDiagramGenerator(
                tol=0.1,
                n_iter=5,
                damp_param=1.0,
                verbose=True,
)
```

We can fit this class to some data by calling the `fit` method. For example, we can create a Laguerre tessellation of the unit cube [0, 1] x [0, 1] x [0, 1] with 1000 cells of equal volume as follows:
```python
import numpy as np
    
domain = np.array([[0, 1],[0, 1],[0, 1]])
domain_vol = np.prod(domain[:, 1] - domain[:, 0])
    
n_grains = 1000
    
seeds = np.column_stack(
        [np.random.uniform(low=d[0], high=d[1], size=n_grains) for d in domain]
)
volumes = (np.ones(n_grains) / n_grains) * domain_vol
    
# call the fit method on data
generator.fit(
        seeds=seeds,
        volumes=volumes,
        domain=domain,
)
```

After calling the fit method, you can use the instance to get various properties of the diagram, e.g., get the centroids and vertices of the cells:
```python
centroids = generator.get_centroids()
vertices = generator.get_vertices()
    
print("diagram centroids:\n", centroids)
print("diagram vertices:\n", vertices)
```

You can plot the diagram in static or interactive mode by using the fitted instance:
```python
from synthetmic.plot import plot_cells_as_pyvista_fig
plot_cells_as_pyvista_fig(
        generator=generator,
        save_path="./example_diagram.html",
)
```

The generated HTML file can be viewed via any browser of your choice.

If you prefer a static figure, you can save it with any of the file formats or extensions namely pdf, eps, ps, tex, and svg. Saving the figure as pdf looks like:
```python
plot_cells_as_pyvista_fig(
        generator=generator,
        save_path="./example_diagram.pdf",
    )
```

To see more usage examples, see the `examples` folder or check below on how to run them via `cli.py`.

The example above uses a custom data. If you would like to use one of the data provided by this package, they can be loaded from the `synthetmic.data.paper` and `synthetmic.data.toy` modules. The former gives access to the data for generating some figures from this [paper](https://www.tandfonline.com/doi/full/10.1080/14786435.2020.1790053) and the latter provides access to some useful toy data. All data creators or loaders from these modules return a `synthetmic.data.utils.SynthetMicData` data object which contains the following fields: `seeds`, `volumes`, `domain`, `periodic`, and `init_weights`.

Each of the fields of the data object can be passed to the `LeguerreDiagramGenerator().fit` method either as keyword/positional arguments or as dictionary. For instance, let's load some data from the `synthetmic.data.paper` and pass the fields as keyword arguments:
```python
from synthetmic.data.paper import create_example5p5_data

data = create_example5p5_data(is_periodic=False)
generator.fit(
        seeds=data.seeds,
        volumes=data.volumes,
        domain=data.domain,
)
```
or pass the fields as dictionary:
```python
from dataclasses import asdict
generator.fit(**asdict(data))
```

## Working with source codes
### Build from source
If you would like to build this project from source either for development purposes or for any other reason, it is recommended to install [uv](https://docs.astral.sh/uv/). This is what is adopted in this project. To install uv, follow the instructions in this [link](https://docs.astral.sh/uv/getting-started/installation/).

If you don't want to use uv, you can use other alternatives like [pip](https://pip.pypa.io/en/stable/).

The following instructions use uv for building synthetmic from source.

1. Clone the repository by running

    ```
    git clone https://github.com/synthetic-microstructures/synthetmic
    ```

1. Create a python virtual environment by running

    ```
     uv venv .venv --python PYTHON_VERSION
    ```
    > Here, PYTHON_VERSION is the supported Python version. Note that this project requires version >=3.12.3

1. Activate the virtual environment by running

    ```
    source .venv/bin/activate
    ```

1. Prepare all modules and dependencies by running the following:

    ```
    uv sync --all-extras
    ```

### Running examples
We created a command line interface (cli) for recreating some of the examples provided in the this [paper](https://www.tandfonline.com/doi/full/10.1080/14786435.2020.1790053) (and lots more!).

To check the available commands in the cli, run

```
python cli.py --help
```

There are currently two commands available in the cli: `recreate` and `analyse`.

You can check information about each of these commands by running

```
python cli.py COMMAND --help
```
where `COMMAND` is any of the commands.

Running a command with its appropriate args is simple. For instance, if you would like to recreate some of the two-dimensional examples in the above-mentioned paper, and save the generated plots in the ./plots dir, run

```
python cli.py recreate --example 2d --save-dir ./plots
```
You can do the same for three-dimension examples. You can pass the flag `--interactive` or `-i` to save the generated plots as a `.html` file, which can then be opened in a browser to interact with them:

```
python cli.py recreate --example 2d --save-dir ./plots --interactive
```

> Note: by default, the generated plots will be saved as `.pdf`. Passing `--interactive` flag to 2d case will be skipped since this is not that interesting for interactivity.

### Running tests
To run all tests, run

```
pytest -v tests
```

## Authors and maintainers
- [R. O. Ibraheem](https://github.com/Rasheed19)
- [D. P. Bourne](https://github.com/DPBourne)
- [S. M. Roper](https://github.com/smr29git)

## References
If you use this package in your research, please refer to the link to this project. Additionally, please consider citing the following paper:
```bibtex
@article{Bourne01112020,
author = {D. P. Bourne and P. J. J. Kok and S. M. Roper and W. D. T. Spanjer},
title = {Laguerre tessellations and polycrystalline microstructures: a fast algorithm for generating grains of given volumes},
journal = {Philosophical Magazine},
volume = {100},
number = {21},
pages = {2677--2707},
year = {2020},
publisher = {Taylor \& Francis},
doi = {10.1080/14786435.2020.1790053},
URL = {https://doi.org/10.1080/14786435.2020.1790053},
eprint = {https://doi.org/10.1080/14786435.2020.1790053}
}
```
You may also be interested in some of our other libraries:
* [LPM](https://github.com/DPBourne/Laguerre-Polycrystalline-Microstructures) - MATLAB code for generating synthetic polycrystalline microstructures using Laguerre diagrams
* [pyAPD](https://github.com/mbuze/PyAPD) - a Python library for computing *anisotropic* Laguerre diagrams
* [SynthetMic-GUI](https://github.com/synthetic-microstructures/synthetmic-gui) - a web app for generating 2D and 3D synthetic polycrystalline microstructures using Laguerre tessellations
