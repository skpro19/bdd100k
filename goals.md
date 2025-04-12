# Project Goals and Tasks Summary

## Overall Goal
The assignment aims to assess programming and analytical skills in the context of an end-to-end data science project, specifically focusing on object detection using the BDD dataset.

## Key Requirements
*   **GitHub Repository:** All code, documentation, analysis, and instructions must be submitted via a GitHub repository.
*   **Documentation:** Clear documentation is required for each task, including setup, usage instructions (especially for the container and model/evaluation), analysis findings, model choices, and evaluation results.
*   **Coding Standards:** Adhere to PEP8 standards, use docstrings, and optionally use tools like `black` or `pylint`.
*   **Containerization:** The data analysis task must be deployable and runnable within a self-contained Docker container.

## Tasks

### 1. Data Analysis (10 points)
*   Analyze the BDD object detection dataset (100k images + labels). Focus only on the 10 detection classes.
*   Analyze class distributions, train/validation split, anomalies, and patterns.
*   Visualize dataset statistics (e.g., in a dashboard) and interesting/unique samples.
*   Document the analysis and include the code in the repository.
*   Package the analysis code into a Docker container.

### 2. Model (5 points + 5 bonus points)
*   Choose an object detection model (pre-trained is allowed, or train your own).
*   Justify your model choice and explain its architecture in the documentation.
*   Include code snippets or working notebooks in the repository.
*   **Bonus:** Build a data loader and training pipeline for the BDD dataset. Train for at least one epoch on a subset of the data and include the code snippet.

### 3. Evaluation and Visualization (10 points)
*   Evaluate your chosen model on the BDD validation dataset.
*   Document quantitative performance, choosing appropriate metrics and justifying their selection.
*   Perform qualitative analysis: visualize ground truth vs. predictions, identify where the model fails, and potentially cluster failure cases.
*   Connect evaluation results back to the data analysis findings.
*   Suggest potential improvements to the model or data based on your evaluation. 