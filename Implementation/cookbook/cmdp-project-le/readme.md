# CMDP Project

This project implements the CMDP (Codebook-Mediated Differential Privacy) methodology for NLP tasks. The goal is to enhance privacy in text data while maintaining model performance.

## Project Structure

- `data/`: Contains training and evaluation data.
- `src/`: Contains the source code for the project.
- `config/`: Contains configuration files.
- `notebooks/`: Contains Jupyter notebooks for analysis and experiments.
- `requirements.txt`: Lists the Python dependencies.
- `README.md`: Project documentation.
- `main.py`: Entry point for the project.

## Setup

1. Install the required dependencies:

    ```bash
    pip install -r requirements.txt
    ```

2. Place your training data in the `data/` directory.

3. Run the training script:

    ```bash
    python main.py
    ```

## Configuration

Configure the hyperparameters in `config/config.yaml`.

## Usage

- **Training**: Use the `train.py` script to train the CMDP model.
- **Inference**: Use the `inference.py` script to make predictions with the trained model.

## Contributing

Feel free to contribute to the project by opening issues or submitting pull requests.
