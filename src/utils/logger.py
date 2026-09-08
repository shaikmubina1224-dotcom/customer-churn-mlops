import logging
import os


# Create logs directory if it does not exist
os.makedirs("artifacts/logs", exist_ok=True)


# Create logger
logger = logging.getLogger("customer_churn_mlops")
logger.setLevel(logging.INFO)


# Create file handler
file_handler = logging.FileHandler(
    "artifacts/logs/app.log"
)

# Create console handler
console_handler = logging.StreamHandler()


# Define log format
formatter = logging.Formatter(
    "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)


# Apply format to handlers
file_handler.setFormatter(formatter)
console_handler.setFormatter(formatter)


# Add handlers to logger
logger.addHandler(file_handler)
logger.addHandler(console_handler)