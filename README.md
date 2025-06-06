# EPTHenOpt: Heat Exchanger Network Synthesis and Optimization

![License](https://img.shields.io/badge/license-MIT-blue.svg)
![Python Version](https://img.shields.io/badge/python-3.8+-brightgreen.svg)
![Build Status](https://img.shields.io/badge/build-passing-brightgreen.svg)
![Version](httpshttps://img.shields.io/badge/version-0.3.0-informational)

---

## Introduction

**EPTHenOpt** is a powerful Python package designed for the synthesis and optimization of Heat Exchanger Networks (HEN). The goal of HEN synthesis is to design a network of heat exchangers that minimizes the total annualized cost (TAC) by maximizing heat recovery between hot and cold process streams, thereby reducing the need for expensive external utilities.

This package provides robust, easy-to-use implementations of two powerful metaheuristic algorithms—**Genetic Algorithm (GA)** and **Teaching-Learning-Based Optimization (TLBO)**—specifically adapted for the complexities of HEN problems. It is built to serve academic, research, and industrial users who are interested in process integration, energy efficiency, and systems optimization.

## Theoretical Foundation

The optimization engine in this package is based on the **Stage-Wise Superstructure (SWS)** model, a foundational concept in HEN synthesis developed by Yee and Grossmann (1990). This approach allows for the simultaneous optimization of energy and capital costs without relying on pinch-based heuristics.

For a detailed explanation of the underlying methodology and its academic context, please see our **[literature review and background](./LITERATURE.md)**.

## Features

-   **Two Metaheuristic Solvers**: Choose between a Genetic Algorithm (GA) or Teaching-Learning-Based Optimization (TLBO) to find optimal HEN configurations.
-   **Detailed Cost Model**: The fitness function accurately calculates the Total Annualized Cost (TAC), including capital costs (CapEx) for exchangers, heaters, and coolers, and operating costs (OpEx) for utilities.
-   **Advanced Constraint Handling**: Utilizes an adaptive penalty system to intelligently handle critical process constraints, such as Minimum Approach Temperature (EMAT), pinch violations, and unmet temperature targets.
-   **Flexible Problem Definition**: Easily define complex HEN problems through simple CSV files, with support for:
    -   Specific U-values and cost laws for certain matches.
    -   Forbidden and required matches between streams.
-   **Parallel Processing**: Accelerate the optimization process using multiprocessing with an island-model migration strategy, allowing multiple optimization runs to share their best solutions.
-   **Comprehensive and Clear Results**: The tool provides detailed summaries of the problem definition and a full breakdown of the best solution found, including its structure, costs, and performance.
-   **Dual Usage Modes**: Can be run as a command-line tool or imported as a Python library for programmatic use in other scripts.

## Installation

1.  **Clone the repository:**

    ```bash
    git clone [https://github.com/may3rd/EPTHenOpt.git](https://github.com/may3rd/EPTHenOpt.git)
    cd EPTHenOpt
    ```

2.  **Install the package:**
    You can install the package and its dependencies (`numpy`, `pytest`) using `pip`. For development, it's recommended to install in "editable" mode.

    ```bash
    # For development (recommended)
    pip install -e .

    # For a standard installation
    pip install .
    ```

## Usage

EPTHenOpt can be used in two ways: as a command-line tool or as a Python library.

### 1. Running via Command Line (CLI)

After installation, the package provides a command-line script `run_hen_problem`.

**Basic Example (using Genetic Algorithm):**
This command runs the optimization using the default GA model and looks for `streams.csv` and `utilities.csv` in the current directory.

```bash
run_hen_problem
```

**Advanced Example (using TLBO with Parallel Workers):**
This command runs a TLBO optimization using 8 parallel workers on your Mac, with custom epoch and population settings.

```bash
run_hen_problem --model TLBO --number_of_workers 8 --epochs 20 --population_size 250
```

**Displaying Help:**
To see a full list of all available command-line arguments, use the `--help` or `-h` flag.

```bash
run_hen_problem --help
```

### 2. Programmatic Usage (API)

You can import and use `EPTHenOpt` directly in your Python scripts. This is ideal for integration with other tools or for running automated analyses.

**Example Script (`my_analysis.py`):**

```python
import EPTHenOpt

# The if __name__ == '__main__': block is essential for
# multiprocessing to work correctly on macOS and Windows.
if __name__ == '__main__':
    print("--- Starting a GA Optimization ---")
    EPTHenOpt.run(
        model='GA',
        epochs=10,
        population_size=100,
        number_of_workers=8,
        noverbose=True # Suppress generation-by-generation output
    )

    print("\n--- Starting a new TLBO run with different files ---")
    EPTHenOpt.run(
        model='TLBO',
        epochs=5,
        streams_file='path/to/my_streams.csv',
        utilities_file='path/to/my_utilities.csv'
    )
```

### Input Data Format

All input data is provided through CSV files. The default filenames are listed below, but you can specify different paths via CLI or API arguments.

-   **`streams.csv`**: Defines the process streams.

    -   `Name`: Unique identifier for the stream (e.g., H1, C1).
    -   `Type`: `hot` or `cold`.
    -   `TIN_spec`: Inlet temperature (K).
    -   `TOUT_spec`: Target outlet temperature (K).
    -   `Fcp`: Heat capacity flowrate (kW/K).

-   **`utilities.csv`**: Defines the available hot and cold utilities.

    -   `Name`: Unique identifier (e.g., S1, W1).
    -   `Type`: `hot_utility` or `cold_utility`.
    -   `TIN_utility`, `TOUT_utility`: Inlet and outlet temperatures (K).
    -   `Unit_Cost_Energy`: Cost per unit of energy ($/kW).
    -   ...and other cost parameters.

-   **`matches_U_cost.csv`** (Optional): Specify custom U-values and costs for specific stream matches.
-   **`forbidden_matches.csv`** (Optional): List pairs of streams that are not allowed to be matched.
-   **`required_matches.csv`** (Optional): List pairs of streams that must be matched in the final solution.

## File Descriptions

| File/Folder                   | Description                                                                        |
| ----------------------------- | ---------------------------------------------------------------------------------- |
| `setup.py`                    | The main setup script for package installation and CLI entry point definition.     |
| `LITERATURE.md`               | An explanation of the underlying Stage-Wise Superstructure (SWS) model.            |
| `run_problem.py`              | Contains the logic for the `run_hen_problem` command-line tool.                    |
| `EPTHenOpt/`                  | The core Python package source code.                                               |
| `EPTHenOpt/__init__.py`       | Defines the package's public API, including the `EPTHenOpt.run()` function.        |
| `EPTHenOpt/hen_models.py`     | Contains the data models: `Stream`, `Utility`, `CostParameters`, and `HENProblem`. |
| `EPTHenOpt/base_optimizer.py` | Base class for all optimizers, containing the shared fitness calculation logic.    |
| `EPTHenOpt/ga_helpers.py`     | Implementation of the `GeneticAlgorithmHEN` optimizer.                             |
| `EPTHenOpt/tlbo_helpers.py`   | Implementation of the `TeachingLearningBasedOptimizationHEN` optimizer.            |
| `EPTHenOpt/cores.py`          | Manages parallel processing and inter-worker communication (migration).            |
| `EPTHenOpt/utils.py`          | Utility functions for loading data, calculating LMTD, and displaying results.      |
| `tests/`                      | Contains unit tests for the package components.                                    |
| `*.csv`                       | Default input data files for streams, utilities, and match constraints.            |
| `requirements.txt`            | A list of Python package dependencies for `pip`.                                   |
| `LICENSE.md`                  | The MIT License file for the project.                                              |

## Citation

If you use EPTHenOpt in your research or work, please cite it as follows:

```bibtex
@software{Lorprajuksiri_EPTHenOpt_2025,
  author = {Lorprajuksiri, Maetee},
  title = {{EPTHenOpt: A Python Package for Heat Exchanger Network Synthesis and Optimization}},
  month = {6},
  year = {2025},
  url = {[https://github.com/may3rd/EPTHenOpt](https://github.com/may3rd/EPTHenOpt)}
}
```

## Contributing

Contributions are welcome! If you would like to contribute to the project, please follow these steps:

1.  Fork the repository.
2.  Create a new branch for your feature or bug fix (`git checkout -b feature/your-feature-name`).
3.  Make your changes and commit them (`git commit -m 'Add some feature'`).
4.  Push to the branch (`git push origin feature/your-feature-name`).
5.  Open a pull request.

Please make sure to add or update tests as appropriate.

## License

This project is licensed under the MIT License - see the [LICENSE.md](LICENSE.md) file for details.

## Contact

This package was developed by:
**Maetee Lorprajuksiri**

-   **Email**: 26008353@pttgcgroup.com
-   **Affiliation**: E-PT-PX Department, GC Maintenance and Engineering Co. Ltd.
