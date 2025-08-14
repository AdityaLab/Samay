def detect_anomalies_in_data(epochs, data_path, image_name, train_end, anomaly_start, anomaly_end):
    import sys
    import os

    sys.path.insert(0, os.path.abspath('..'))

    from src.samay.model import LPTMModel
    import src.samay.model
    import samay.model
    import importlib

    importlib.reload(samay.model)

    #-----------------------------------------
    #LOADING THE MODEL

    print("Using model.py from:", src.samay.model.__file__)


    config = {
        "task_name": "forecasting",
        "forecast_horizon": 192,
        "head_dropout": 0,
        "weight_decay": 0,
        "max_patch": 16,
        "freeze_encoder": True,  # Freeze the patch embedding layer
        "freeze_embedder": True,  # Freeze the transformer encoder
        "freeze_head": False,  # The linear forecasting head must be trained
        "freeze_segment": True,  # Freeze the segmention module
    }
    model = LPTMModel(config)

    #-----------------------------------------
    #TRAIN THE MODEL

    from src.samay.anomaly_dataset_script import LPTMDataset
    dataset_path = data_path
    train_len = train_end
    train_dataset = LPTMDataset(
        name="ett",
        datetime_col=None,
        path=dataset_path,
        mode="train",
        horizon=192,
        boundaries=[train_len, 0, 0],
        bypass=2
    )
    if (epochs != 0):
        finetuned_model = model.finetune(train_dataset, epoch = epochs)

    #-----------------------------------------
    #TEST THE MODEL

    import pandas as pd
    import plotly.graph_objects as go
    import plotly.io as pio
    pio.renderers.default = 'browser'



    df = pd.read_csv(dataset_path)
    n = len(df)
    print(n)

    dataset = LPTMDataset(
        name="ett",
        datetime_col=None,
        path=dataset_path,
        mode="test",
        horizon=192,
        boundaries=[train_len, 0, n], 
        stride=10,
        #seq_len=512,
        task_name="forecasting",
        bypass=2
    )

    avg_loss, trues, preds, histories = model.evaluate(dataset, task_name="forecasting")

    #-----------------------------------------
    #PLOT AND SAVE THE RESULTS

    import numpy as np
    import matplotlib.pyplot as plt

    trues = np.array(trues)
    preds = np.array(preds)
    histories = np.array(histories)

    '''for i in range(trues.shape[0]):  # num_windows
        for j in range(trues.shape[1]):  # num_channels
            trues[i, j, :] = scaler.inverse_transform(trues[i, j, :].reshape(-1, 1)).flatten()
            preds[i, j, :] = scaler.inverse_transform(preds[i, j, :].reshape(-1, 1)).flatten()
            histories[i, j, :] = scaler.inverse_transform(histories[i, j, :].reshape(-1, 1)).flatten()'''


    # --- Parameters ---
    stride = 10
    num_windows, num_channels, forecast_len = preds.shape
    first_k = stride  
    total_len = (num_windows - 1) * stride + first_k + histories.shape[-1]

    stitched_true = np.zeros((num_channels, total_len))
    stitched_pred = np.zeros((num_channels, total_len))

    for i in range(num_windows):
        start = i * stride
        stitched_true[:, start + histories.shape[-1] : start + histories.shape[-1] + first_k] = trues[i, :, :first_k]
        stitched_pred[:, start + histories.shape[-1] : start + histories.shape[-1] + first_k] = preds[i, :, :first_k]

    channel_idx = 0
    x = np.arange(stitched_true.shape[1])

    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=x,
        y=stitched_true[channel_idx],
        mode='lines',
        name='Ground Truth',
        line=dict(color='darkblue')
    ))

    fig.add_trace(go.Scatter(
        x=x,
        y=stitched_pred[channel_idx],
        mode='lines',
        name='Forecast',
        line=dict(color='red', dash='dash')
    ))

    fig.update_layout(
        title=f"Stitched Full Time-Series Forecast (Channel {channel_idx})",
        xaxis_title="Time",
        yaxis_title="Value",
        legend=dict(font=dict(size=12)),
        hovermode="x unified",
        height=500,
        width=1000,
    )

    #fig.show()
    '''import os
    from datetime import datetime

    image_folder = "saved_plots"
    os.makedirs(image_folder, exist_ok=True)

    image_path = os.path.join(image_folder, f"{image_name}.png")
    fig.write_image(image_path)
    print(f"Plot saved to: {image_path}")'''

    #-----------------------------------------
    #LIST OUT FINAL ANOMALIES AND SAVE TO CSV

    import numpy as np

    channel_idx = 0

    true_flat = stitched_true[channel_idx]
    pred_flat = stitched_pred[channel_idx]

    epsilon = 1e-6

    relative_errors = ((true_flat - pred_flat) ** 2) / (np.abs(true_flat) + epsilon)

    # Get indices of top 10 anomaly points (sorted descending)
    top_10_indices = np.argsort(relative_errors)[-10:][::-1]



    print("Top 10 anomaly indices:", top_10_indices.tolist())

    L = anomaly_end - anomaly_start + 1
    correct_anomalies = []
    for index in top_10_indices:
        if (min(anomaly_start-100, anomaly_start-L) < index and max(anomaly_end+100, anomaly_end+L) > index):
            correct_anomalies.append(index)

    #Save to csv
    import pandas as pd
    import os

    anomaly_folder = "saved_anomalies"
    os.makedirs(anomaly_folder, exist_ok=True)

    anomaly_file = os.path.join(anomaly_folder, "anomalies_log.csv")

    df = pd.DataFrame([top_10_indices.tolist()])
    df['correct_anomalies'] = [correct_anomalies]


    if not os.path.exists(anomaly_file):
        headers = [f"anomaly_{i+1}" for i in range(10)] + ['correct_anomalies']
        df.to_csv(anomaly_file, index=False, header=headers)
    else:
        df.to_csv(anomaly_file, mode='a', index=False, header=False)

    print(f"Anomalies saved to: {anomaly_file}")
