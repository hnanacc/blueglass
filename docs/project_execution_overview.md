# 💡 Project Execution Flow Overview: 
Our codebase is structured as a modular execution pipeline. It operates on a configuration-driven approach, ensuring flexibility and scalability across multiple tasks. The core workflow consists of the following steps:

## 📊 [BLUEGLASS Pipeline Structure](docs/project_execution_overview.md)
#### 🏃 Runners: Executes training, evaluation, and inference.
#### 📦 Model Wrappers: Generic interface to integrate any model.
#### 📚 Dataloaders: Handles dataset processing.
#### 🛠 Evaluators: Different evaluation metrics for benchmarking.
#### ⚡ Structures: Core utilities for framework operations.

## 1️⃣ Configuration Loading

The BlueGlass workflow starts by loading a configuration file that defines:
- The model
- The dataset
- The evaluation metrics
- Other execution parameters

To launch an experiment, use:

<pre> python launch.py --config-name modelstore.mmdet_detr.coco </pre>

In this example: **modelstore** is the name of the runner, **mmdet_detr** refers to the selected model, and **coco** is the dataset.

Each configuration is registered in the BlueGlass config module under a specific runner config file.
📄 This particular configuration is stored in: blueglass/blueglass/configs/modelstore_benchmarks.py

Below is how the configuration is registered using Hydra’s ConfigStore:
<pre>
cs.store(
    f"modelstore.mmdet_detr.coco",
    BLUEGLASSConf(
        runner=ModelStoreRunnerConf(),  # Defines execution logic
        dataset=ModelstoreDatasetConf(
            test=ds_test,
            label=ds_test
        ),  # Specifies test and label datasets
        model=ModelConf(
            name=Model.YOLO,
            checkpoint_path=osp.join(WEIGHTS_DIR, "yolo", "yolov8x-oiv7.pt"),
        ),  # Model details and weights path
        evaluator=LabelMatchEvaluatorConf(names=ev),  # Evaluation metrics
        experiment=ExperimentConf(name=f"modelstore_yolo_{ds_name}"),  # Experiment metadata
    ),
)
</pre>
This modular configuration structure allows you to easily switch models, datasets, or evaluators by changing only the config name.

## 2️⃣ Runner Initialization

The runner module reads these configurations and initializes the necessary components, such as models, evaluators, and metrics.
## 3️⃣ Task-Specific Runners
Based on the defined task in the configuration, different specialised runners are activated. A few examples are listed below:

- Benchmarking Runner: Evaluates model performance against standard benchmarks.
  <pre> python launch.py --config-name modelstore.mmdet_detr.coco </pre>
- Feature Extraction Runner: Extracts and processes feature representations.
  <pre> python launch.py --config-name features.mmdet_detr.coco </pre>
- Linear Probing Runner: Conducts linear probing experiments.
  <pre> python launch.py --config-name probe.mmdet_detr.coco </pre>
- SAE (Sparse Autoencoder) Runner: Applies sparse autoencoders for interpretation.
  <pre> python launch.py --config-name saes.mmdet_detr.coco </pre>

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
