# 💡 Project Execution Flow Overview: 
Our codebase is structured as a modular execution pipeline. It operates on a configuration-driven approach, ensuring flexibility and scalability across multiple tasks. The core workflow consists of the following steps:

## 📊 [BLUEGLASS Pipeline Structure](docs/project_execution_overview.md)
#### 🏃 Runners: Executes training, evaluation, and inference.
#### 📦 Model Wrappers: Generic interface to integrate any model.
#### 📚 Dataloaders: Handles dataset processing.
#### 🛠 Evaluators: Different evaluation metrics for benchmarking.
#### ⚡ Structures: Core utilities for framework operations.

## 1️⃣ Configuration Loading

The system starts by loading configuration files defining the model, dataset, evaluation metrics, and execution parameters.
## 2️⃣ Runner Initialization

The runner module reads these configurations and initializes the necessary components, such as models, evaluators, and metrics.
## 3️⃣ Task-Specific Runners
Based on the defined task in the configuration, different specialized runners are activated:

- Benchmarking Runner: Evaluates model performance against standard benchmarks.
- Feature Extraction Runner: Extracts and processes feature representations.
- Feature Exploration Runner: Analyzes feature distributions and relationships.
- Linear Probing Runner: Conducts linear probing experiments.
- SAE (Sparse Autoencoder) Runner: Applies sparse autoencoders for interpretation.
- Interpreter Tools: Loads additional tools for model interpretability.

Additionally, if you need to perform a new task, you can easily create a custom runner by following the existing templates. Typically, you will need to:

- Define a configuration schema to specify task-specific parameters.
- Implement a new runner class by extending the base runner structure.
- Load the required model, dataset, and evaluation metrics similar to those of existing runners.
- Ensure proper logging and result handling for integration with the pipeline.

This modular design allows seamless extension of new functionalities while maintaining consistency with the execution flow. 🚀

## 4️⃣ Dataset and Intermediate Data Management

The system dynamically loads intermediate datasets based on the chosen model and dataset in the configuration, ensuring that each task receives the correct pre-processed data.
## 5️⃣ Execution and Result Processing

The runner executes the assigned task, collects results, and logs outputs, ensuring an organized workflow.
This structure enables seamless execution of various machine learning and interpretability tasks while maintaining modularity and extensibility."
