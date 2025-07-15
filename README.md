# SynthetMic
A Python package for generating synthetic Laguerre polycrystalline microstructures.

## Installation
Coming soon stay stuned!

## Usage
coming soon stay stuned!

## Working with source codes
### Build from source
If you would like to build this project from source either for development purposes or for any other reasons, it is recommended to install [uv](https://docs.astral.sh/uv/). This is what is adopted in this project. To install uv, follow the instructions in this [link](https://docs.astral.sh/uv/getting-started/installation/).

If you don't want to use uv, you can use other alternatives like [pip](https://pip.pypa.io/en/stable/).

The following instructions use uv for building systhetmic from source.

1. Clone the repository by running

    ```
    git clone https://github.com/synthetic-microstructures/synthetmic
    ```

1. Navigate to ./synthetmic, create a python virtual environment by running

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
We create command line interface (cli) for recreating some of the examples provided in the this [paper](https://www.tandfonline.com/doi/full/10.1080/14786435.2020.1790053) (and lots more!).

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

Running a command with its appropriate args is simple.

For instance, if you would like to recreate some of the two-dimensional examples in the above-mentioned paper, and save the generated plots in the ./plots dir, run

```
python cli.py recreate --example 2d --save-dir ./plots
```
You can do the same for three-dimension examples, in faact, you can pass the flat `--interactive` or `-i` to save the generated plots as `.html`, which can then be open in a browser and interact with them:

```
python cli.py recreate --example 2d --save-dir ./plots --interactive
```

> Note: by default, the generated plots will be saved as `.pdf`. Passing `--interactive` flag to 2d case will be skipped since this is not that "interesting" for interactivity.

### Running tests
To run all tests, run

```
pytest -v tests
```

## TODO:
    - add information about the package to readme
    - update information in the pyproject.toml file
    - add more methods to the base generator: get_vertices(), etc.
    - add workflow for publishing package to pypi

## References
Coming soon stay stuned!

## Contributing
Coming soon stay stuned!
