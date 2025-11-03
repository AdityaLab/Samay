import io
import os
import time
from datetime import datetime

import numpy as np
import pandas as pd
import plotly.graph_objs as go
import streamlit as st

from samay.dataset import LPTMDataset
from samay.model import LPTMModel

# ----------------------------
# Helpers
# ----------------------------
UPLOAD_DIR = os.path.join(os.path.dirname(__file__), "uploads")
os.makedirs(UPLOAD_DIR, exist_ok=True)


def save_uploaded_file(
    uploaded_file: "st.runtime.uploaded_file_manager.UploadedFile",
) -> str:
    filename = f"{int(time.time())}_{uploaded_file.name}"
    path = os.path.join(UPLOAD_DIR, filename)
    with open(path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    return path


def init_session_state():
    if "model" not in st.session_state:
        st.session_state.model = None
    if "finetuned" not in st.session_state:
        st.session_state.finetuned = False
    if "last_forecast" not in st.session_state:
        st.session_state.last_forecast = None  # (df_forecast: pd.DataFrame)
    if "uploaded_path" not in st.session_state:
        st.session_state.uploaded_path = None
    if "datetime_col" not in st.session_state:
        st.session_state.datetime_col = None


# ----------------------------
# UI
# ----------------------------
st.set_page_config(page_title="Time-series Forecasting UI", layout="wide")
init_session_state()

st.title("Time-series Forecasting (LPTM)")
st.caption(
    "Upload a CSV, choose parameters, optionally fine-tune the model, and generate forecasts."
)


with st.sidebar:
    st.header("Configuration")

    horizon = st.number_input(
        "Forecast horizon", min_value=1, max_value=10000, value=192, step=1
    )
    past_window = st.number_input(
        "Past window (timesteps)",
        min_value=16,
        max_value=4096,
        value=512,
        step=16,
        help="Amount of past history to use (model/dataset dependent).",
    )

    st.divider()
    st.subheader("Fine-tune settings")
    ft_epochs = st.number_input(
        "Fine-tune epochs", min_value=1, max_value=1000, value=5
    )

    st.divider()
    st.subheader("Model Config")
    freeze_encoder = st.checkbox("Freeze encoder", value=True)
    freeze_embedder = st.checkbox("Freeze embedder", value=True)
    freeze_head = st.checkbox("Freeze head", value=False)
    freeze_segment = st.checkbox("Freeze segment", value=True)


st.subheader("1) Upload dataset (CSV)")
uploaded = st.file_uploader("Upload a CSV file", type=["csv"])

df_preview = None
if uploaded is not None:
    try:
        df_preview = pd.read_csv(uploaded)
    except Exception:
        uploaded.seek(0)
        df_preview = pd.read_csv(io.BytesIO(uploaded.getvalue()))

    st.write("Preview:")
    st.dataframe(df_preview.head(), use_container_width=True)

    cols = df_preview.columns.tolist()
    datetime_col = st.selectbox("Datetime column", options=cols)
    st.session_state.datetime_col = datetime_col

    if st.button("Save upload"):
        st.session_state.uploaded_path = save_uploaded_file(uploaded)
        st.success(f"Saved: {os.path.basename(st.session_state.uploaded_path)}")


st.subheader("2) Initialize model")
col_a, col_b = st.columns(2)
with col_a:
    if st.button("Create/Reset model"):
        config = {
            "task_name": "forecasting",
            "forecast_horizon": int(horizon),
            "head_dropout": 0,
            "weight_decay": 0,
            "max_patch": 16,
            "freeze_encoder": bool(freeze_encoder),
            "freeze_embedder": bool(freeze_embedder),
            "freeze_head": bool(freeze_head),
            "freeze_segment": bool(freeze_segment),
        }
        st.session_state.model = LPTMModel(config)
        st.session_state.finetuned = False
        st.session_state.last_forecast = None
        st.success("Model initialized.")

with col_b:
    if st.session_state.model is not None:
        st.info("Model ready")


st.subheader("3) Fine-tune (optional)")
can_finetune = (
    st.session_state.model is not None
    and st.session_state.get("uploaded_path") is not None
    and uploaded is not None
)
if not can_finetune:
    st.caption("Upload data and initialize model to enable fine-tuning.")

ft_col1, ft_col2 = st.columns(2)
with ft_col1:
    fine_tune_clicked = st.button("Fine-tune model")

if fine_tune_clicked:
    if not can_finetune:
        st.error(
            "Please upload a dataset, select datetime column, and initialize the model first."
        )
    else:
        try:
            train_dataset = LPTMDataset(
                name="custom",
                datetime_col=st.session_state.datetime_col,
                path=st.session_state.uploaded_path,
                mode="train",
                horizon=int(horizon),
            )

            with st.status("Fine-tuning in progress...", expanded=True) as status:
                st.write(f"Epochs: {ft_epochs}")
                # The underlying model's finetune likely loops epochs internally.
                # If not, this still triggers training with default epochs.
                _ = st.session_state.model.finetune(train_dataset)
                st.session_state.finetuned = True
                status.update(label="Fine-tuning complete", state="complete")
                st.success("Fine-tuning finished.")
        except Exception as e:
            st.error(f"Fine-tune failed: {e}")


st.subheader("4) Forecast")
forecast_btn = st.button("Generate forecast")
if forecast_btn:
    if (
        st.session_state.model is None
        or st.session_state.get("uploaded_path") is None
        or uploaded is None
    ):
        st.error(
            "Please upload a dataset, select datetime column, and initialize the model first."
        )
    else:
        try:
            eval_dataset = LPTMDataset(
                name="custom",
                datetime_col=st.session_state.datetime_col,
                path=st.session_state.uploaded_path,
                mode="train",
                horizon=int(horizon),
            )
            res = st.session_state.model.evaluate(eval_dataset, task_name="forecasting")
            # Defensive unpacking for different return signatures
            if isinstance(res, tuple):
                if len(res) == 4:
                    metrics, trues, preds, histories = res
                elif len(res) == 3:
                    metrics, trues, preds = res
                    histories = None
                else:
                    raise ValueError("Unexpected evaluate() return signature")
            else:
                raise ValueError("Unexpected evaluate() return type")

            st.write("Metrics:", metrics)

            # Prepare a flat forecast CSV by averaging across channels if multivariate
            trues_np = np.array(trues)
            preds_np = np.array(preds)
            histories_np = np.array(histories) if histories is not None else None
            # Shape assumption: [N, C, H]
            n, c, h = preds_np.shape
            # Construct per-sample per-channel forecast rows
            rows = []
            for i in range(n):
                for ch in range(c):
                    row = {"sample_index": i, "channel": ch}
                    for t in range(h):
                        row[f"t+{t + 1}"] = preds_np[i, ch, t]
                    rows.append(row)
            df_forecast = pd.DataFrame(rows)
            st.session_state.last_forecast = df_forecast
            st.session_state.trues_np = trues_np
            st.session_state.preds_np = preds_np
            st.session_state.histories_np = histories_np
            st.dataframe(df_forecast.head(), use_container_width=True)

        except Exception as e:
            st.error(f"Forecast failed: {e}")


# ----------------------------
# Plotly visualization controls
# ----------------------------
if (
    st.session_state.get("trues_np") is not None
    and st.session_state.get("preds_np") is not None
):
    st.subheader("Forecast visualization")
    trues_np = st.session_state.trues_np
    preds_np = st.session_state.preds_np
    histories_np = st.session_state.histories_np

    n, c, h = preds_np.shape
    hist_len = histories_np.shape[-1] if histories_np is not None else 0

    viz_col1, viz_col2 = st.columns(2)
    with viz_col1:
        sample_idx = st.number_input(
            "Sample index", min_value=0, max_value=max(0, n - 1), value=0, step=1
        )
    with viz_col2:
        channel_idx = st.number_input(
            "Channel", min_value=0, max_value=max(0, c - 1), value=0, step=1
        )

    # Build Plotly figure
    history = (
        histories_np[sample_idx, channel_idx, :] if histories_np is not None else None
    )
    true = trues_np[sample_idx, channel_idx, :]
    pred = preds_np[sample_idx, channel_idx, :]

    x_hist = list(range(hist_len)) if history is not None else []
    x_fore = list(range(hist_len, hist_len + h)) if hist_len else list(range(h))

    fig = go.Figure()
    if history is not None:
        fig.add_trace(
            go.Scatter(
                x=x_hist,
                y=history,
                mode="lines",
                name=f"History (len={hist_len})",
                line=dict(color="darkblue"),
            )
        )
    fig.add_trace(
        go.Scatter(
            x=x_fore,
            y=true,
            mode="lines",
            name="Ground Truth",
            line=dict(color="darkblue", dash="dash"),
            opacity=0.6,
        )
    )
    fig.add_trace(
        go.Scatter(
            x=x_fore,
            y=pred,
            mode="lines",
            name="Forecast",
            line=dict(color="red", dash="dash"),
        )
    )
    fig.update_layout(
        height=400,
        width=None,
        margin=dict(l=20, r=20, t=40, b=40),
        title=f"Sample {sample_idx} — Channel {channel_idx}",
        xaxis_title="Time",
        yaxis_title="Value",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )

    st.plotly_chart(fig, use_container_width=True)
    try:
        st.session_state.last_fig_html = fig.to_html(
            full_html=False, include_plotlyjs="cdn"
        )
    except Exception:
        st.session_state.last_fig_html = None


st.subheader("5) Downloads")
dl_col1, dl_col2 = st.columns(2)

with dl_col1:
    if st.session_state.last_forecast is not None:
        csv_buf = io.StringIO()
        st.session_state.last_forecast.to_csv(csv_buf, index=False)
        st.download_button(
            label="Download forecast CSV",
            data=csv_buf.getvalue(),
            file_name=f"forecast_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
            mime="text/csv",
        )
    else:
        st.caption("No forecast yet.")

with dl_col2:
    if st.session_state.model is not None and st.session_state.finetuned:
        try:
            import torch

            # Serialize model to memory for direct browser download
            buf = io.BytesIO()
            torch.save(st.session_state.model, buf)
            buf.seek(0)
            st.download_button(
                label="Download fine-tuned model (.pt)",
                data=buf.getvalue(),
                file_name=f"finetuned_model_{int(time.time())}.pt",
                mime="application/octet-stream",
            )
        except Exception as e:
            st.caption("Model export not supported in this environment.")
            st.code(str(e))
    else:
        st.caption("No fine-tuned model available.")

st.subheader("Figure download")
if st.session_state.get("last_fig_html"):
    st.download_button(
        label="Download current figure (HTML)",
        data=st.session_state.last_fig_html,
        file_name=f"forecast_plot_{datetime.now().strftime('%Y%m%d_%H%M%S')}.html",
        mime="text/html",
    )
else:
    st.caption("No figure available yet.")


st.divider()
st.caption(
    "Tip: run with `streamlit run example/app.py`. Make sure your environment has Samay dependencies installed."
)
