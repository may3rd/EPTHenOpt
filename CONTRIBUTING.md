# Contributing to EPTHenOpt

First off, thank you for considering contributing to EPTHenOpt! We welcome any contributions, from fixing a typo to implementing new features. This document provides guidelines to help you get started.

## How Can I Contribute?

* **Reporting Bugs:** If you find a bug, please open an issue on our GitHub page. Include a clear description, a minimal reproducible example, and any relevant error messages.
* **Suggesting Enhancements:** Have an idea for a new feature or an improvement to an existing one? Open an issue to start a discussion.
* **Pull Requests:** If you want to contribute code, please submit a pull request.

## Development Setup

To get your development environment set up, please follow these steps. Since you're on a Mac, these commands should work well in your terminal.

1.  **Fork & Clone the Repository:**
    * Fork the repository on GitHub.
    * Clone your fork locally:
        ```bash
        git clone [https://github.com/YOUR_USERNAME/EPTHenOpt.git](https://github.com/YOUR_USERNAME/EPTHenOpt.git)
        cd EPTHenOpt
        ```

2.  **Set up a Virtual Environment:**
    It's highly recommended to work in a virtual environment to manage dependencies.
    ```bash
    # Create a virtual environment
    python3 -m venv .venv

    # Activate it
    source .venv/bin/activate
    ```

3.  **Install in Editable Mode:**
    Install the package in "editable" mode along with the development dependencies. This allows you to test your changes live without reinstalling.
    ```bash
    pip install -e .[dev]
    ```
    *(Note: We will add the `[dev]` dependencies in `setup.py` next)*

## Submitting a Pull Request

1.  Create a new branch for your feature or bug fix:
    ```bash
    git checkout -b feature/my-new-feature
    ```

2.  Make your changes. Follow the coding style guidelines below.

3.  Add and run tests for your changes.
    ```bash
    pytest
    ```

4.  Commit your changes with a clear commit message.
    ```bash
    git commit -m "feat: Add my new feature"
    git push origin feature/my-new-feature
    ```

5.  Open a pull request from your fork to the `main` branch of the `may3rd/EPTHenOpt` repository. Provide a clear title and description of the changes you've made.

## Coding Style

* Please follow the **PEP 8** style guide for Python code.
* Use clear and descriptive variable and function names.
* Add docstrings to all new modules, classes, and functions, explaining their purpose, arguments, and what they return.

Thank you for your contribution!

