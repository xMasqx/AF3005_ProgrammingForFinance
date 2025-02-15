# AF3005_ProgrammingForFinance_Assignment_1
# Loan Eligibility and Stock Risk Analysis

## Overview
This repository contains an implementation of loan eligibility assessment and stock risk evaluation using interactive widgets in Jupyter Notebook. The project utilizes `ipywidgets` to provide an interactive user interface for inputting data and receiving real-time feedback on loan qualification and stock investment risks.

## Features
- **Loan Eligibility Checker**: Determines whether a user qualifies for a loan based on employment status, monthly income, and credit rating.
- **Stock Risk Assessment**: Evaluates stock investments based on past returns to categorize them into high, medium, or low-risk stocks.
- **Interactive UI**: Utilizes `ipywidgets` for an engaging user experience, enabling real-time decision feedback.

## Prerequisites
Ensure you have the following dependencies installed in your Python environment:

```bash
pip install ipywidgets numpy matplotlib
```

## Implementation
### Loan Eligibility Checker
1. Users input their **employment status** (Employed/Unemployed).
2. Users enter their **monthly income** in PKR.
3. Users adjust the **credit score slider** (Range: 300-850).
4. The system evaluates and displays loan approval status with applicable interest rates.

### Stock Risk Assessment
1. Users input **stock returns** as a comma-separated list.
2. The system processes the values and categorizes the stock into High, Medium, or Low risk.

## Running the Notebook
1. Open Jupyter Notebook or JupyterLab.
2. Load the provided `.ipynb` file.
3. Execute the cells sequentially to initialize the interactive widgets.
4. Provide the necessary inputs and observe the system's response in real-time.

## Repository Structure
```
📂 Loan-and-Stock-Analysis
│── 📄 README.md  # Documentation
│── 📄 loan_stock_analysis.ipynb  # Jupyter Notebook implementation
│── 📄 requirements.txt  # List of dependencies
```

## Notes
- The project uses **interactive widgets**, so ensure Jupyter Notebook is running in a compatible environment.
- Errors related to missing packages can be resolved using the `pip install` command mentioned above.

## Author
- **Your Name**  
- Contact: [Your Email]
- GitHub: [Your GitHub Profile]

## License
This project is licensed under the MIT License.

