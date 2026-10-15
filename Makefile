
VENV = .venv
PYTHON = $(VENV)/bin/python
PIP = $(VENV)/bin/pip

TRAIN_CSV = datasets/dataset_train.csv
TEST_CSV = datasets/dataset_test.csv

MODEL = logreg_model.npz
PREDICTIONS = houses.csv

.PHONY: start train graph predict evaluate validate clean fclean restart

# Create the virtual environment and install dependencies
start:
	@echo "========================================"
	@echo "       DSLR - Project setup"
	@echo "========================================"
	@echo "[1/3] Creating virtual environment..."
	@python3 -m venv $(VENV)
	@echo "[2/3] Updating pip..."
	@$(PIP) install --upgrade pip --progress-bar on
	@echo "[3/3] Installing dependencies..."
	@$(PIP) install -r requirements.txt --progress-bar on
	@echo "Setup completed successfully!"

# Generate graph
graph:
	@$(PYTHON) -m scripts.histogram $(TRAIN_CSV)
	@$(PYTHON) -m scripts.scatter_plot $(TRAIN_CSV)
	@$(PYTHON) -m scripts.pair_plot $(TRAIN_CSV)

# Train the model
train:
	@$(PYTHON) -m scripts.logreg_train $(TRAIN_CSV)

# Generate predictions
predict:
	@$(PYTHON) -m scripts.logreg_predict $(TEST_CSV)

# Evaluate the model
evaluate:
	@$(PYTHON) -m scripts.logreg_evaluate $(TRAIN_CSV)

# Validate the model
validate:
	@$(PYTHON) -m scripts.logreg_validate $(TRAIN_CSV)

# Remove Python cache files
clean:
	@echo "Removing Python cache..."
	@find . -type d -name "__pycache__" -prune -exec rm -rf {} +
	@find . -type f -name "*.py[co]" -delete
	@echo "Cache removed."
	@echo "Removing generated files..."
	@rm -f $(MODEL) $(PREDICTIONS)
	@echo "Cleanup completed."

# Remove generated files and Python cache
fclean: clean
	@echo "Removing venv file & dependencies..."
	@rm -rf $(VENV)
	@echo "Full cleanup completed."

restart: fclean start
	@echo "DSLR is restarted !"
