# Role-Lora Notebook

This project contains a Jupyter notebook for training a model using LoRA (Low-Rank Adaptation). The notebook has been modified for compatibility with Google Colab, ensuring that all code cells run smoothly in that environment.

## Project Structure

```
role-lora-notebook
├── role_lora_pipeline.ipynb  # Jupyter notebook for training the model
├── requirements.txt           # Required Python packages
├── .gitignore                 # Files and directories to ignore by Git
└── README.md                  # Project documentation
```

## Installation

To run this project, you need to install the required Python packages. You can do this by running the following command:

```bash
pip install -r requirements.txt
```

## Running the Notebook

### In Visual Studio Code

1. Clone the repository to your local machine.
2. Open the `role_lora_pipeline.ipynb` file in Visual Studio Code.
3. Ensure you have the necessary Python environment set up with the required packages installed.
4. Run the notebook cells sequentially.

### In Google Colab

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/edris6/rola_lora_pipeline/blob/main/role_lora_pipeline.ipynb)

Click the badge above to open this notebook directly in Google Colab. Once it's open, you can run all cells by navigating to **Runtime ▸ Run all**.

## Requirements

The following packages are required for this project:

- `transformers`
- `datasets`
- `peft`
- `accelerate`
- `bitsandbytes`

You can install these packages using the `requirements.txt` file provided in the project.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.