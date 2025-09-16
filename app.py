from flask import Flask, request, jsonify, send_file
from flask_cors import CORS, cross_origin
import torch
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import io
import sys
import os
import traceback

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "src")))
sys.path.insert(0, "/nethome/sdas412/Samay/src/samay/models/chronosforecasting/chronos")

from src.samay.model import LPTMModel, TimesfmModel
from src.samay.dataset import LPTMDataset, TimesfmDataset

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "https://showmick119.github.io"}}, supports_credentials=True)

loaded_models = {}
datasets = {}

############################################
# 1) /load_model
############################################
@app.route("/load_model", methods=["POST"])
@cross_origin()
def load_model():
    data = request.get_json()
    model_name = data.get("model_name")
    if model_name not in ["LPTM", "TimesFM"]:
        return jsonify({"error": "Invalid model name"}), 400
    try:
        if model_name == "LPTM":
            config = {
                "task_name": "forecasting",
                "forecast_horizon": 192,
                "head_dropout": 0.1,
                "weight_decay": 0,
                "freeze_encoder": True,
                "freeze_embedder": True,
                "freeze_head": False,
            }
            model = LPTMModel(config)
        elif model_name == "TimesFM":
            repo = "google/timesfm-1.0-200m-pytorch"
            config = {
                "context_len": 512,
                "horizon_len": 192,
                "backend": "gpu",
                "per_core_batch_size": 32,
                "input_patch_len": 32,
                "output_patch_len": 128,
                "num_layers": 20,
                "model_dims": 1280,
                "quantiles": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9],
            }
            model = TimesfmModel(config=config, repo=repo)
        loaded_models[model_name] = model
        return jsonify({"message": f"Model {model_name} loaded successfully"})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

############################################
# 2) /load_dataset
############################################
# @app.route("/load_dataset", methods=["POST"])
# @cross_origin()
# def load_dataset():
#     data = request.get_json()
#     dataset_name = data.get("dataset_name")
#     dataset_path = f"/nethome/sdas412/Samay/data/data/{dataset_name}.csv"
#     try:
#         # LPTMDataset configured with default arguments from example/lptm.ipynb
#         train_dataset = LPTMDataset(
#             name="ett",
#             datetime_col="date",
#             path=dataset_path,
#             mode="train",
#             horizon=192,
#             task_name="forecasting",
#         )
#         datasets[dataset_name] = train_dataset
#         return jsonify({"message": "Dataset loaded successfully", "size": len(train_dataset)})
#     except Exception as e:
#         traceback_str = traceback.format_exc()
#         return jsonify({"error": traceback_str}), 500

############################################
# 2) /upload_dataset
############################################
@app.route("/upload_dataset", methods=["POST"])
@cross_origin()
def upload_dataset():
    if 'dataset' not in request.files:
        return jsonify({"error": "No file part in the request"}), 400
    file = request.files['dataset']
    model_name = request.form.get("model_name")
    if not model_name or model_name not in loaded_models:
        return jsonify({"error": "Model not loaded"}), 400
    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400
    try:
        save_path = "/nethome/sdas412/Samay/data/data/uploaded.csv"
        file.save(save_path)
        if model_name == "LPTM":
            train_dataset = LPTMDataset(
                name="uploaded",
                datetime_col="date",
                path=save_path,
                mode="train",
                horizon=192,
                task_name="forecasting",
            )
        elif model_name == "TimesFM":
            train_dataset = TimesfmDataset(
                name="uploaded",
                datetime_col="date",
                path=save_path,
                mode="train",
                context_len=512,
                horizon_len=128,
            )
        else:
            return jsonify({"error": f"Unsupported model for upload: {model_name}"}), 400
        datasets["uploaded"] = train_dataset
        return jsonify({"message": "Dataset uploaded and loaded successfully", "size": len(train_dataset)})
    except Exception as e:
        traceback_str = traceback.format_exc()
        return jsonify({"error": traceback_str}), 500

############################################
# 3) /finetune
############################################
@app.route("/finetune", methods=["POST"])
@cross_origin()
def finetune():
    data = request.get_json()
    model_name = data.get("model_name")
    dataset_name = "uploaded"
    if model_name not in loaded_models:
        return jsonify({"error": "Model not loaded"}), 400
    if dataset_name not in datasets:
        return jsonify({"error": "Dataset not loaded"}), 400
    try:
        model = loaded_models[model_name]
        dataset = datasets[dataset_name]
        model.finetune(dataset, task_name="forecasting")
        loaded_models[model_name] = model
        return jsonify({"message": f"Finetuning {model_name} on {dataset_name} complete"})
    except Exception as e:
        traceback_str = traceback.format_exc()
        return jsonify({"error": traceback_str}), 500

############################################
# 4) /run_inference
############################################
@app.route("/run_inference", methods=["POST"])
@cross_origin()
def run_inference():
    """
    Returns an actual PNG image that plots
    history (512 timesteps) and forecast (192 timesteps).
    """
    data = request.get_json()
    model_name = data.get("model_name")
    dataset_name = "uploaded"
    if model_name not in loaded_models:
        return jsonify({"error": "Model not loaded"}), 400
    if dataset_name not in datasets:
        return jsonify({"error": "Dataset not loaded"}), 400
    try:
        if model_name == "LPTM":
            model = loaded_models[model_name]
            dataset_path = f"/nethome/sdas412/Samay/data/data/{dataset_name}.csv"
            val_dataset = LPTMDataset(
                name="uploaded",
                datetime_col="date",
                path=dataset_path,
                mode="train",
                horizon=192,
                task_name="forecasting",
            )
            avg_loss, trues, preds, histories = model.evaluate(val_dataset, task_name="forecasting")

            # Convert inputs to numpy arrays
            trues = np.array(trues)
            preds = np.array(preds)
            histories = np.array(histories)

            # Randomly select channel and time index
            channel_idx = np.random.randint(0, 7)
            time_index = np.random.randint(0, trues.shape[0])

            # Extract time series
            history = histories[time_index, channel_idx, :]
            true = trues[time_index, channel_idx, :]
            pred = preds[time_index, channel_idx, :]

            # Create time axes
            offset = len(history)
            forecast_range = range(offset, offset + len(true))

            # Create plot
            plt.figure(figsize=(14, 5))

            # Plot history
            plt.plot(range(len(history)), history, label="History (512 timesteps)", color="#1f77b4", linewidth=2)

            # Plot ground truth
            plt.plot(forecast_range, true, label="Ground Truth (192 timesteps)", 
                    color="#1f77b4", linestyle="--", linewidth=2, alpha=0.6)

            # Plot prediction
            plt.plot(forecast_range, pred, label="Forecast (192 timesteps)", 
                    color="#d62728", linestyle="--", linewidth=2)

            # Titles and labels
            plt.title(f"ETTh1 (Hourly) — (idx={time_index}, channel={channel_idx})", fontsize=18, fontweight='bold')
            plt.xlabel("Time", fontsize=14)
            plt.ylabel("Value", fontsize=14)

            # Style and layout
            plt.grid(True, linestyle='--', alpha=0.4)
            plt.legend(fontsize=12)
            plt.tight_layout()


            # trues = np.array(trues)
            # preds = np.array(preds)
            # histories = np.array(histories)
            # channel_idx = np.random.randint(0, 7)
            # time_index = np.random.randint(0, trues.shape[0])

            # history = histories[time_index, channel_idx, :]
            # true = trues[time_index, channel_idx, :]
            # pred = preds[time_index, channel_idx, :]

            # plt.figure(figsize=(12, 4))
            # plt.plot(range(len(history)), history, label="History (512 timesteps)", c="darkblue")
            # num_forecasts = len(true)
            # offset = len(history)
            # plt.plot(
            #     range(offset, offset + len(true)),
            #     true,
            #     label="Ground Truth (192 timesteps)",
            #     color="darkblue",
            #     linestyle="--",
            #     alpha=0.5,
            # )
            # plt.plot(
            #     range(offset, offset + len(pred)),
            #     pred,
            #     label="Forecast (192 timesteps)",
            #     color="red",
            #     linestyle="--",
            # )
            # plt.title(f"ETTh1 (Hourly) -- (idx={time_index}, channel={channel_idx})", fontsize=18)
            # plt.xlabel("Time", fontsize=14)
            # plt.ylabel("Value", fontsize=14)
            # plt.legend(fontsize=14)
            # plt.tight_layout()

        elif model_name == "TimesFM":
            model = loaded_models[model_name]
            dataset_path = f"/nethome/sdas412/Samay/data/data/{dataset_name}.csv"
            val_dataset = TimesfmDataset(
                name="uploaded",
                datetime_col="date",
                path=dataset_path,
                mode="test",
                context_len=512,
                horizon_len=192,
            )
            avg_loss, trues, preds, histories = model.evaluate(val_dataset)

            trues = np.array(trues)
            preds = np.array(preds)
            histories = np.array(histories)

            # Random selection
            channel_idx = np.random.randint(0, 7)
            time_index = np.random.randint(0, trues.shape[0])

            # Slice time series
            history = histories[time_index, channel_idx, :]
            true = trues[time_index, channel_idx, :]
            pred = preds[time_index, channel_idx, :]

            # Create time ranges
            offset = len(history)
            forecast_range = range(offset, offset + len(true))

            # Create figure
            plt.figure(figsize=(14, 5))

            # Plot history
            plt.plot(range(len(history)), history, label="History (512 timesteps)", 
                    color="#1f77b4", linewidth=2)

            # Plot ground truth
            plt.plot(forecast_range, true, label="Ground Truth (192 timesteps)", 
                    color="#1f77b4", linestyle="--", linewidth=2, alpha=0.6)

            # Plot prediction
            plt.plot(forecast_range, pred, label="Forecast (192 timesteps)", 
                    color="#d62728", linestyle="--", linewidth=2)

            # Labels and formatting
            plt.title(f"ETTh1 (Hourly) — (idx={time_index}, channel={channel_idx})", 
                    fontsize=18, fontweight='bold')
            plt.xlabel("Time", fontsize=14)
            plt.ylabel("Value", fontsize=14)
            plt.legend(fontsize=12)
            plt.grid(True, linestyle='--', alpha=0.4)
            plt.tight_layout()

        img = io.BytesIO()
        plt.savefig(img, format='png')
        plt.close()
        img.seek(0)
        return send_file(img, mimetype='image/png', as_attachment=False, download_name='forecast.png')
    except Exception as e:
        traceback_str = traceback.format_exc()
        return jsonify({"error": traceback_str}), 500

############################################
# Start the Flask server
############################################
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5002)