# EPTHenOpt: Heat Exchanger Network Synthesis and Optimization

![License](https://img.shields.io/badge/license-MIT-blue.svg)
![Python Version](https://img.shields.io/badge/python-3.9+-brightgreen.svg)
![Build Status](https://img.shields.io/badge/build-passing-brightgreen.svg)
![Code Coverage](https://img.shields.io/badge/coverage-85%25-yellow.svg)
![Version](https://img.shields.io/badge/version-0.2.2-informational)

---

## Introduction

**EPTHenOpt** is a powerful Python package designed for the synthesis and optimization of Heat Exchanger Networks (HEN). The goal of HEN synthesis is to design a network of heat exchangers that minimizes the total annualized cost (TAC) by maximizing heat recovery between hot and cold process streams, thereby reducing the need for expensive external utilities.

This package provides robust, easy-to-use implementations of two powerful metaheuristic algorithms—**Genetic Algorithm (GA)** and **Teaching-Learning-Based Optimization (TLBO)**—specifically adapted for the complexities of HEN problems. It is built to serve academic, research, and industrial users who are interested in process integration, energy efficiency, and systems optimization.

## Features

-   **Two Metaheuristic Solvers**: Choose between a Genetic Algorithm (GA) or Teaching-Learning-Based Optimization (TLBO) to find optimal HEN configurations.
-   **Detailed Cost Model**: The fitness function accurately calculates the Total Annualized Cost (TAC), including capital costs (CapEx) for exchangers, heaters, and coolers, and operating costs (OpEx) for utilities.
-   **Advanced Constraint Handling**: Utilizes an adaptive penalty system to intelligently handle critical process constraints, such as Minimum Approach Temperature (EMAT), pinch violations, and unmet temperature targets.
-   **Flexible Problem Definition**: Easily define complex HEN problems through simple CSV files, with support for:
    -   Specific U-values and cost laws for certain matches.
    -   Forbidden and required matches between streams.
-   **Parallel Processing**: Accelerate the optimization process using multiprocessing with an island-model migration strategy, allowing multiple optimization runs to share their best solutions.
-   **Comprehensive and Clear Results**: The tool provides detailed summaries of the problem definition and a full breakdown of the best solution found, including its structure, costs, and performance.

## Installation

To get started with EPTHenOpt, clone the repository and install the required dependencies.

1.  **Clone the repository:**

    ```bash
    git clone [https://github.com/your-username/EPTHenOpt.git](https://github.com/your-username/EPTHenOpt.git)
    cd EPTHenOpt
    ```

2.  **Install prerequisites:**
    The package requires `numpy` and `pytest` (for testing). Install them using the `requirements.txt` file.
    ```bash
    pip install -r requirements.txt
    ```

## Usage

The primary way to use EPTHenOpt is via the command-line interface (CLI) provided by the `run_problem.py` script.

### Running via Command Line (CLI)

You can run an optimization with default settings or customize it using a wide range of arguments.

**Basic Example (using Genetic Algorithm):**

This command runs the optimization using the default Genetic Algorithm (`GA`) model with the `streams.csv` and `utilities.csv` files in the root directory.

```bash
python run_problem.py
```

**Advanced Example (using TLBO with Parallel Workers):**

This command runs the optimization using the Teaching-Learning-Based Optimization (`TLBO`) model, utilizing 4 parallel workers for 50 epochs.

```bash
python run_problem.py --model TLBO --number_of_workers 4 --epochs 50 --generations_per_epoch 25
```

**Displaying Help:**

To see a full list of all available command-line arguments and their descriptions, use the `--help` or `-h` flag.

```bash
python run_problem.py --help
```

### Input Data Format

All input data is provided through CSV files. The default filenames are listed below, but you can specify different paths via CLI arguments.

-   **`streams.csv`**: Defines the process streams.

    -   `Name`: Unique identifier for the stream (e.g., H1, C1).
    -   `Type`: `hot` or `cold`.
    -   `TIN_spec`: Inlet temperature (K).
    -   `TOUT_spec`: Target outlet temperature (K).
    -   `Fcp`: Heat capacity flowrate (kW/K).

    _Example (`streams.csv`):_

    ```csv
    Name,Type,TIN_spec,TOUT_spec,Fcp
    H1,Hot,443,333,30
    H2,Hot,423,303,15
    C1,Cold,293,408,20
    C2,Cold,353,413,40
    ```

-   **`utilities.csv`**: Defines the available hot and cold utilities.

    -   `Name`: Unique identifier (e.g., S1, W1).
    -   `Type`: `hot_utility` or `cold_utility`.
    -   `TIN_utility`, `TOUT_utility`: Inlet and outlet temperatures (K).
    -   `Unit_Cost_Energy`: Cost per unit of energy ($/kW).
    -   `U_overall`, `Fixed_Cost_Unit`, `Area_Cost_Coeff`, `Area_Cost_Exp`: Parameters for cost calculation.

    _Example (`utilities.csv`):_

    ```csv
    Name,Type,TIN_utility,TOUT_utility,Unit_Cost_Energy,U_overall,Fixed_Cost_Unit,Area_Cost_Coeff,Area_Cost_Exp
    S1,Hot_Utility,450,450,80,1.2,0,1200,0.6
    W1,Cold_Utility,293,313,20,0.8,0,1000,0.6
    ```

-   **`matches_U_cost.csv`** (Optional): Specify custom U-values and costs for specific stream matches.
-   **`forbidden_matches.csv`** (Optional): List pairs of streams that are not allowed to be matched.
-   **`required_matches.csv`** (Optional): List pairs of streams that must be matched in the final solution.

### Output Interpretation

The script will print real-time progress and a final summary to the console. The summary includes:

-   A breakdown of the HEN problem (streams and utilities).
-   The Total Annualized Cost (TAC) of the best solution found.
-   A detailed cost breakdown (CapEx, OpEx, penalties).
-   The structure of the optimal network, including all heat exchangers, heaters, and coolers, with their respective duties, areas, and temperatures.

## File Descriptions

| File/Folder                   | Description                                                                        |
| ----------------------------- | ---------------------------------------------------------------------------------- |
| `run_problem.py`              | Main executable script for running HEN optimizations from the command line.        |
| `EPTHenOpt/`                  | The core Python package source code.                                               |
| `EPTHenOpt/__init__.py`       | Defines the package's public API.                                                  |
| `EPTHenOpt/hen_models.py`     | Contains the data models: `Stream`, `Utility`, `CostParameters`, and `HENProblem`. |
| `EPTHenOpt/base_optimizer.py` | Base class for all optimizers, containing the shared fitness calculation logic.    |
| `EPTHenOpt/ga_helpers.py`     | Implementation of the `GeneticAlgorithmHEN` optimizer.                             |
| `EPTHenOpt/tlbo_helpers.py`   | Implementation of the `TeachingLearningBasedOptimizationHEN` optimizer.            |
| `EPTHenOpt/cores.py`          | Manages parallel processing and inter-worker communication (migration).            |
| `EPTHenOpt/utils.py`          | Utility functions for loading data, calculating LMTD, and displaying results.      |
| `tests/`                      | Contains unit tests for the package components.                                    |
| `*.csv`                       | Default input data files for streams, utilities, and match constraints.            |
| `requirements.txt`            | A list of Python package dependencies.                                             |
| `usage.txt`                   | A text file containing the full command-line help message.                         |

## Example Results

Below is a sample of the output generated for the best solution found during an optimization run.

```
--- Summary of Multiple GA Runs ---
Run with Seed worker_0: True TAC = 154321.89 (Optimized Obj. = 154321.89)
Run with Seed worker_1: True TAC = 160123.45 (Optimized Obj. = 160123.45)

Best True TAC found across all runs (corresponding to best Optimized Objective): 154321.89
  (This solution had an Optimized Objective of: 154321.89)
  Best solution from Seed: worker_0

Cost Breakdown for the Best Overall Solution:
  True TAC: 154321.89, Optimized Obj.: 154321.89
  CapEx (Process Ex.): 85432.10
  CapEx (Heaters): 10000.00
  CapEx (Coolers): 12000.00
  OpEx (Hot Utility): 25400.90
  OpEx (Cold Utility): 21488.89
  No significant penalties applied...

Structure of the Best Overall Solution:

  Process Heat Exchangers:
    H1-C2 (Stage 1): Q=150.00 kW, Area=25.67 m^2
      H1: FlowCp_branch=10.00 (SplitFrac=1.000), Tin=200.0 K, Tout=185.0 K
      C2: FlowCp_branch=8.00 (SplitFrac=1.000), Tin=70.0 K, Tout=88.8 K
  Total Q Recovered (Process Exchangers): 150.00 kW
  Total Area (Process Exchangers): 25.67 m^2

  Utility Units:
    Heater for C1: Q=50.00 kW, Area=8.12 m^2, Tc_in=100.0 K, Tc_out=150.0 K
    Cooler for H2: Q=80.00 kW, Area=12.45 m^2, Th_in=95.0 K, Th_out=80.0 K

  Utility Duty Summary:
    Total Cold Utility Duty: 80.00 kW
    Total Hot Utility Duty: 50.00 kW
```

## Citation

If you use EPTHenOpt in your research or work, please cite it as follows:

```bibtex
@software{Lorprajuksiri_EPTHenOpt_2024,
  author = {Lorprajuksiri, Maetee},
  title = {{EPTHenOpt: A Python Package for Heat Exchanger Network Synthesis and Optimization}},
  month = {6},
  year = {2024},
  url = {[https://github.com/your-username/EPTHenOpt](https://github.com/your-username/EPTHenOpt)}
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
