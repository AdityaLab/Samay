import logging
from pathlib import Path
import torch
from transformers import AutoModelForSeq2SeqLM, AutoModelForCausalLM
from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error
from chronos import ChronosPipeline
from json_logger import JsonFileHandler, JsonFormatter

# Configure logging
log_file = Path("evaluation_results.json")
json_handler = JsonFileHandler(log_file)
json_handler.setFormatter(JsonFormatter())

logging.basicConfig(format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
logger.addHandler(json_handler)

def load_model(model_dir: str, model_type: str = "seq2seq", device: str = "cuda", **kwargs):
    model_class = AutoModelForSeq2SeqLM if model_type == "seq2seq" else AutoModelForCausalLM
    model = ChronosPipeline.from_pretrained(model_dir, model_type=model_type)
    return model

def evaluate_model(model, fit_data, test_data, prediction_length, metrics, logger=None):
    predictions, targets = [], []
    context = torch.tensor(fit_data)
    predictions = model.predict(context, prediction_length=prediction_length).squeeze().tolist()

    results = {}
    results['num_samples'] = len(predictions)
    results['predictions'] = predictions[0]

    if 'RMSE' in metrics:
        results['RMSE'] = mean_squared_error(test_data, predictions[0], squared=False)
    if 'MAPE' in metrics:
        results['MAPE'] = mean_absolute_percentage_error(test_data, predictions[0])

    logger.info(f"Evaluation results: {results}")

    return results

# Example usage
if __name__ == "__main__":
    pass
    # model_dir = "./output/run-15/checkpoint-final"
    # model_type = "seq2seq"
    # metrics = ['RMSE', 'MAPE']
    # model = load_model(model_dir, model_type)
    # logger.info(f"Model loaded from {model_dir}")

    # # Assuming data_train and data_test are already defined
    # column_id = column_list[0]
    # results = evaluate_model(model, data_train[column_id].values, data_test[column_id].values, abs(offset), metrics, logger)
    # print(results)