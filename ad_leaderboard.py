import datetime
import gc
import os
import sys
import time

import pandas as pd
import torch
from sklearn.metrics import roc_auc_score, precision_score, recall_score, f1_score

src_path = os.path.abspath(os.path.join("src"))
if src_path not in sys.path:
    sys.path.insert(0, src_path)

from samay.utils import get_tsb_ad_datasets, load_args 
from samay.model import MomentModel, LPTMModel
from samay.dataset import MomentDataset, LPTMDataset 

start = time.time()
NAMES = get_tsb_ad_datasets("data/TSB-AD-U")
end = time.time()
print(f"Time taken to load dataset names: {end - start:.2f} seconds")

if __name__ == "__main__":
    very_start = time.time()
    for model_name in ["MomentModel", "LPTMModel"]:
        csv_path = f"leaderboard/AD_{model_name}.csv"
        if not os.path.exists(csv_path):
            df = pd.DataFrame(columns=["dataset", "size_in_MB", "eval_time", "AUC_ROC", "Precision", "Recall", "F1"])
            df.to_csv(csv_path, index=False)
        mod_start = time.time()
        mod_timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        for fpath, (freq, fsize) in NAMES:
            dataset_name = fpath.split("/")[-1].split(".")[0]
            print(f"Model eval started at: {mod_timestamp}")
            print(f"Running {model_name} on {dataset_name}")
            train_size = fpath.split(".")[0].split("_")[-3]
            if model_name == "MomentModel":
                args = load_args("configs/moment_detection.json")
                model = MomentModel(args)
                train_set = MomentDataset(path=fpath, boundaries=[train_size, train_size, 0], task_name="detection", mode="train")
                test_set = MomentDataset(path=fpath, boundaries=[train_size, train_size, 0], task_name="detection", mode="test")
            elif model_name == "LPTMModel":
                args = load_args("configs/lptm.json")
                model = LPTMModel(args)
                train_set = LPTMDataset(path=fpath, boundaries=[train_size, train_size, 0], task_name="detection", mode="train")
                test_set = LPTMDataset(path=fpath, boundaries=[train_size, train_size, 0], task_name="detection", mode="test")

            start = time.time()
            model.finetune(dataset=train_set, task_name="detection")
            trues, preds, labels = model.evaluate(test_set, task_name="detection")
            anomaly_score = ((preds - trues) ** 2).flatten()
            # set anomalies to be outside mean + 3*std
            threshold = anomaly_score.mean() + 3 * anomaly_score.std()
            pred_labels = (anomaly_score > threshold).astype(int)
            # AUC_ROC, Precision, Recall, F1 as metrics
            auc_roc = roc_auc_score(labels, anomaly_score)
            precision = precision_score(labels, pred_labels)
            recall = recall_score(labels, pred_labels)
            f1 = f1_score(labels, pred_labels)
            print(f"AUC_ROC: {auc_roc:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}, F1: {f1:.4f}")
            
            end = time.time()
            print(f"Time taken to evaluate {model_name} on {dataset_name}: {end - start:.2f} seconds")

            eval_time = end - start
            unit = 's'
            if eval_time > 1000:
                eval_time /= 60
                unit = 'm'

            df = pd.read_csv(csv_path)
            row_name = f"{dataset_name}"
            if row_name in df['dataset'].values:
                print(f"Dataset {dataset_name} already exists in {csv_path}, skipping...")
                continue
            else:
                new_row = {
                    "dataset": dataset_name,
                    "size_in_MB": fsize,
                    "eval_time": f"{eval_time:.2f} {unit}",
                    "AUC_ROC": f"{auc_roc:.4f}",
                    "Precision": f"{precision:.4f}",
                    "Recall": f"{recall:.4f}",
                    "F1": f"{f1:.4f}"
                }
                df = df.append(new_row, ignore_index=True)
                df.to_csv(csv_path, index=False)
                print(f"Results saved to {csv_path}")

        mod_end = time.time()
        print(f"Total time taken to evaluate {model_name} on all datasets: {mod_end - mod_start:.2f} seconds")

    very_end = time.time()
    print("All evaluations completed.")
    print(f"Total time taken to evaluate all models on all datasets: {very_end - very_start:.2f} seconds")
