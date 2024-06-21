# Generating Tables from the Parametric Knowledge of Language Models

This repository contains **WikiTabGen** - a benchmark of 100 Wikipedia tables for the evaluation of LLMs 

📝 Details on the method can be found in our paper available under: https://arxiv.org/pdf/2406.10922

## Usage
Examples for GPT-3.5 for all prompting methods (full table, row-by-row and cell-by-cell) are available in _example_notebooks_ folder.
You need to set your open.api_key in the Imports section.
Upon successful run a results folder will be created with the _tables_ subfolder containing generated tables in CSV format and _result.json_ file with the logs of prompts and LLM responses.


## Evaluation
To produce the evaluation metrics of your experiment run the notebook _example_notebooks/Metrics_calculation.ipynb_ .
You need to set the vaue of tables_folder (path to CSV files generated by LLM) and result_folder (path to the folder where you want to save the metrics report).
The notebook will calculate the metrics and save the report in CSV format in result_folder.

## More
If you encounter any errors, or you observe unexpected behavior, please contact the authors.
