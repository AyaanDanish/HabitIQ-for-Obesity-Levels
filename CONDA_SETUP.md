# HabitIQ Conda Environment Setup

## Quick Start

1. **Create the conda environment:**

   ```bash
   conda env create -f environment.yml
   ```

2. **Activate the environment:**

   ```bash
   conda activate habitiq
   ```

3. **Run the application:**
   ```bash
   streamlit run source/app.py
   ```

## Environment Management

### Creating the Environment

If you don't have the `environment.yml` file, you can create the environment manually:

```bash
# Create environment with Python 3.11
conda create -n habitiq python=3.11

# Activate environment
conda activate habitiq

# Install dependencies
conda install -c conda-forge streamlit pandas numpy plotly altair joblib scikit-learn matplotlib seaborn pip
pip install shap>=0.42.0
```

### Updating the Environment

If you need to update dependencies:

```bash
# Activate environment
conda activate habitiq

# Update all packages
conda update --all

# Or install specific packages
conda install package_name
```

### Exporting Environment (for sharing)

To create a new environment.yml file from your current setup:

```bash
# Activate the environment first
conda activate habitiq

# Export environment
conda env export > environment.yml
```

### Removing the Environment

If you need to start fresh:

```bash
# Deactivate first if currently active
conda deactivate

# Remove environment
conda env remove -n habitiq
```

## Troubleshooting

### Common Issues

1. **Streamlit not found:**

   - Make sure you've activated the environment: `conda activate habitiq`
   - Verify installation: `streamlit --version`

2. **Model file not found:**

   - Ensure you're running from the project root directory
   - Check that `model/random_forest_model.pkl` or `model/obesity_model_adjusted.pkl` exists

3. **Permission errors on Windows:**

   - Run conda commands in an Administrator command prompt
   - Or use Anaconda Prompt

4. **Environment conflicts:**
   - Remove and recreate the environment if packages conflict
   - Use `conda clean --all` to clear conda cache

## Development Setup

For development with additional tools:

```bash
# Activate environment
conda activate habitiq

# Install development dependencies
conda install jupyter notebook ipykernel
pip install black flake8 pytest

# Register kernel for Jupyter
python -m ipykernel install --user --name habitiq --display-name "HabitIQ Environment"
```

## Application Structure

```
ethics/
├── source/
│   ├── app.py              # Main Streamlit application
│   └── model_training.ipynb # Model training notebook
├── model/
│   ├── random_forest_model.pkl     # Original trained model
│   └── obesity_model_adjusted.pkl  # Adjusted model (if exists)
├── data/
│   └── ObesityData.csv     # Training dataset
├── environment.yml         # Conda environment specification
├── requirements.txt        # Pip requirements (backup)
└── README.md              # Project documentation
```
