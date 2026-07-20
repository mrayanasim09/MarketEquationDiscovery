# Contributing to GNNs for Macroeconomic Forecasting

Thank you for your interest in contributing to this research repository. As this project is primarily an academic research artifact, contributions are managed slightly differently from standard software projects.

## How to Contribute

### 1. Reporting Issues
If you encounter bugs, broken scripts, or inconsistencies in the data pipeline, please open an issue on the GitHub repository. Include:
- A clear description of the issue.
- The operating system and environment details.
- Steps to reproduce the bug.
- Any relevant logs or error messages.

### 2. Suggesting Improvements
We welcome suggestions for alternative modeling approaches or data sources. Please open an issue outlining your proposal before initiating substantial work, as the primary goal of this repository is to replicate a specific published study.

### 3. Submitting Pull Requests
If you wish to submit code changes (e.g., bug fixes or performance improvements):
1. Fork the repository.
2. Create a new branch for your feature (`git checkout -b feature/your-feature-name`).
3. Commit your changes (`git commit -m 'Add some feature'`).
4. Push to the branch (`git push origin feature/your-feature-name`).
5. Open a Pull Request.

## Code Style
- **Python**: Please adhere to PEP 8 standards. Use `black` and `flake8` for formatting and linting.
- **Documentation**: Ensure that any new functions or classes include appropriate docstrings detailing inputs, outputs, and purpose.

## Testing
When submitting changes that affect the data processing or modeling pipeline:
- Ensure that the reproduction scripts (`reproduce.sh` or `reproduce.ps1`) complete successfully.
- Verify that the final outputs maintain statistical equivalence to the published benchmark results (matching the published cryptographic hashes where applicable).

Thank you for helping improve the robustness and reproducibility of this research!
