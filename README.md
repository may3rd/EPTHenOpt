# EPTHenOpt: Heat Exchanger Network Synthesis and Optimization

![License](https://img.shields.io/badge/license-MIT-blue.svg)
![Python Version](https://img.shields.io/badge/python-3.8+-brightgreen.svg)
![Build Status](https://img.shields.io/badge/build-passing-brightgreen.svg)
![Version](https://img.shields.io/badge/version-0.7.0-informational)

---

## Introduction

**EPTHenOpt** is a powerful Python package designed for the synthesis and optimization of Heat Exchanger Networks (HEN). The goal of HEN synthesis is to design a network of heat exchangers that minimizes the total annualized cost (TAC) by maximizing heat recovery between hot and cold process streams, thereby reducing the need for expensive external utilities.

This package provides robust, easy-to-use implementations of several powerful metaheuristic algorithms and supports both single-objective and multi-objective optimization. It is built to serve academic, research, and industrial users who are interested in process integration, energy efficiency, and systems optimization.

## Theoretical Foundation

The optimization engine in this package is based on the **Stage-Wise Superstructure (SWS)** model, a foundational concept in HEN synthesis developed by Yee and Grossmann (1990). This approach allows for the simultaneous optimization of energy and capital costs without relying on pinch-based heuristics.

For a detailed explanation of the underlying methodology and its academic context, please see our **[literature review and background](./LITERATURE.md)**.

## Features

-   **Multiple Metaheuristic Solvers**: Choose between Genetic Algorithm (GA), Teaching-Learning-Based Optimization (TLBO), Particle Swarm Optimization (PSO), Simulated Annealing (SA), or Ant Colony Optimization (ACO) for single-objective optimization.
-   **Multi-Objective Optimization**: Employs the industry-standard NSGA-II algorithm to find the optimal trade-off front (Pareto Front) between economic cost (TAC) and environmental impact (CO₂ emissions).
-   **Detailed and Flexible Cost Model**: The fitness function accurately calculates the Total Annualized Cost (TAC). Default cost parameters can be set independently for process exchangers, heaters, and coolers.
-   **Advanced Constraint Handling**: Utilizes an adaptive penalty system to intelligently handle critical process constraints, such as Minimum Approach Temperature (EMAT), pinch violations, and unmet temperature targets.
-   **Flexible Problem Definition**: Easily define complex HEN problems through simple CSV files or a JSON configuration file.
-   **Parallel Processing**: Accelerate the optimization process using multiprocessing with an island-model migration strategy for single-objective solvers.
-   **Dual Usage Modes**: Can be run as a command-line tool or imported as a Python library for programmatic use in other scripts.
-   **Structured Results Export**: Save optimization results, including network structure and the full Pareto front, to CSV and JSON files for easy analysis in Excel or other tools.

## Installation

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/may3rd/EPTHenOpt.git
    cd EPTHenOpt
    ```

2.  **Install the package:**
    You can install the package and its dependencies (`numpy`, `pytest`) using `pip`. For development, it's recommended to install in "editable" mode with development tools like Sphinx.
    ```bash
    # For development (recommended)
    pip install -e .[dev]

    # For a standard installation
    pip install .
    ```

## Usage

EPTHenOpt can be configured and run in three flexible ways: using a JSON config file, command-line arguments, or as a Python library. Command-line arguments will always override settings from a config file.

### 1. Configuration via `config.json`

The easiest way to manage settings is through the `config.json` file. It now supports sections for all optimizers and environmental parameters.

**Example `config.json`:**
```json
{
  "file_paths": {
    "streams_file": "streams.csv",
    "utilities_file": "utilities.csv"
  },
  "core_optimization": {
    "model": "GA",
    "population_size": 200,
    "epochs": 10,
    "number_of_workers": 8,
    "output_dir": "./latest_run_results"
  },
  "environmental_parameters": {
    "objective": "single",
    "default_co2_hot_utility": 0.2,
    "default_co2_cold_utility": 0.05
  },
}
```

### 2. Running via Command Line (CLI)

The `run_hen_problem` script can be used to execute optimizations from anywhere in your terminal.

**Single-Objective Example (PSO):**
```bash
run_hen_problem --model PSO --pso_inertia_weight 0.6 --number_of_workers 8
```

**Multi-Objective Example (NSGA-II):**
This command runs a multi-objective optimization to find the trade-off between cost and CO₂ emissions. The results (Pareto front) will be saved to the specified output directory.
```bash
run_hen_problem --objective multi --output_dir ./results_tac_vs_co2
```

### 3. Programmatic Usage (API)

Import and use `EPTHenOpt` directly in your Python scripts for maximum flexibility.

**Example Script (`my_analysis.py`):**
```python
import EPTHenOpt

# The if __name__ == '__main__': block is essential for
# multiprocessing to work correctly on macOS and Windows.
if __name__ == '__main__':
    # Run a single-objective optimization
    print("--- Starting a Single-Objective SA Run ---")
    EPTHenOpt.run(
        model='SA',
        epochs=50,
        population_size=100,
        output_dir='./sa_results'
    )

    # Run a multi-objective optimization
    print("\n--- Starting a Multi-Objective NSGA-II Run ---")
    EPTHenOpt.run(
        objective='multi',
        epochs=100,
        population_size=200,
        output_dir='./pareto_results'
    )
```

## File Descriptions

| File/Folder                   | Description                                                                        |
| ----------------------------- | ---------------------------------------------------------------------------------- |
| `setup.py`                    | The main setup script for package installation and CLI entry point definition.     |
| `config.json`                 | A JSON file for easily configuring optimization runs.                              |
| `LITERATURE.md`               | An explanation of the underlying Stage-Wise Superstructure (SWS) model.            |
| `run_problem.py`              | Contains the logic for the `run_hen_problem` command-line tool.                    |
| `EPTHenOpt/`                  | The core Python package source code.                                               |
| `EPTHenOpt/__init__.py`       | Defines the package's public API, including the `EPTHenOpt.run()` function.          |
| `EPTHenOpt/hen_models.py`     | Contains the data models: `Stream`, `Utility`, `CostParameters`, and `HENProblem`. |
| `EPTHenOpt/base_optimizer.py` | Base class for all optimizers, containing the shared fitness calculation logic.    |
| `EPTHenOpt/ga_helpers.py`     | Implementation of the `GeneticAlgorithmHEN` optimizer.                             |
| `EPTHenOpt/tlbo_helpers.py`   | Implementation of the `TeachingLearningBasedOptimizationHEN` optimizer.            |
| `EPTHenOpt/pso_helpers.py`    | Implementation of the `ParticleSwarmOptimizationHEN` optimizer.                    |
| `EPTHenOpt/sa_helpers.py`     | Implementation of the `SimulatedAnnealingHEN` optimizer.                           |
| `EPTHenOpt/aco_helpers.py`    | Implementation of the `AntColonyOptimizationHEN` optimizer.                        |
| `EPTHenOpt/nsga2_helpers.py`  | Implementation of the multi-objective `NSGAIIHEN` optimizer.                       |
| `EPTHenOpt/cores.py`          | Manages parallel processing and inter-worker communication (migration).            |
| `EPTHenOpt/utils.py`          | Utility functions for loading data, displaying results, and exporting files.       |
| `tests/`                      | Contains unit tests for the package components.                                    |
| `*.csv`                       | Default input data files for streams, utilities, and match constraints.            |
| `requirements.txt`            | A list of Python package dependencies for `pip`.                                   |
| `LICENSE.md`                  | The MIT License file for the project.                                              |
| `CONTRIBUTING.md`             | Guidelines for developers who want to contribute to the project.                   |

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

Contributions are welcome! If you would like to contribute to the project, please follow the guidelines in our [CONTRIBUTING.md](CONTRIBUTING.md) file.

## License

This project is licensed under the MIT License - see the [LICENSE.md](LICENSE.md) file for details.

## Contact

This package was developed by:
**Maetee Lorprajuksiri**

-   **Email**: 26008353@pttgcgroup.com
-   **Affiliation**: E-PT-PX Department, GC Maintenance and Engineering Co. Ltd.
