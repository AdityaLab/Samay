import numpy as np
import matplotlib.pyplot as plt

class BaseVisualization:
    def __init__(self, trues, preds, labels=None):
        self.trues = trues
        self.preds = preds
        # self.labels = np.array(labels) if labels is not None else None

    def plot(self):
        raise NotImplementedError("Subclasses should implement this method")

class ForecastVisualization(BaseVisualization):
    def __init__(self, trues, preds, histories):
        super().__init__(trues, preds)
        self.histories = histories

    def plot(self, channel_idx=1, time_idx=1, crop_history=False):
        #channel_idx = np.random.randint(0, self.trues.shape[1])
        # time_idx = np.random.randint(0, self.trues.shape[0])

        history = self.histories[time_idx][channel_idx, :]
        true = self.trues[time_idx][channel_idx, :]
        pred = self.preds[time_idx][channel_idx, :]

        if crop_history:
            length_to_crop = 3*len(true) if len(history) > 3*len(true) else len(history)
            history = history[-length_to_crop:]

        plt.figure(figsize=(12, 4))
        plt.plot(range(len(history[])), history[], label='History', c='darkblue')
        offset = len(history[]) 
        plt.plot(range(offset, offset + len(true)), true, label='Ground Truth', color='darkblue', linestyle='--', alpha=0.5)
        plt.plot(range(offset, offset + len(pred)), pred, label='Forecast', color='red', linestyle='--')
        plt.title(f"Forecast Visualization -- (idx={time_idx}, channel={channel_idx})", fontsize=18)
        plt.xlabel('Time', fontsize=14)
        plt.ylabel('Value', fontsize=14)
        plt.legend(fontsize=14)
        plt.show()

class ImputationVisualization(BaseVisualization):
    def __init__(self, trues, preds, masks):
        super().__init__(trues, preds)
        self.masks = np.array(masks)

    def plot(self):
        idx = np.random.randint(self.trues.shape[0])
        channel_idx = np.random.randint(self.trues.shape[1])

        fig, axs = plt.subplots(2, 1, figsize=(10, 5))
        axs[0].set_title(f"Channel={channel_idx}")
        axs[0].plot(self.trues[idx, channel_idx, :].squeeze(), label='Ground Truth', c='darkblue')
        axs[0].plot(self.preds[idx, channel_idx, :].squeeze(), label='Predictions', c='red')
        axs[0].legend(fontsize=16)
        axs[1].imshow(np.tile(self.masks[np.newaxis, idx, channel_idx], reps=(8, 1)), cmap='binary')
        plt.show()

class AnomalyDetectionVisualization(BaseVisualization):
    def __init__(self, trues, preds, labels):
        super().__init__(trues, preds, labels)

    def plot(self):
        anomaly_scores = (self.trues - self.preds) ** 2
        anomaly_start = 74158
        anomaly_end = 74984
        start = anomaly_start - 512
        end = anomaly_end + 512

        plt.plot(self.trues[start:end], label="Observed", c='darkblue')
        plt.plot(self.preds[start:end], label="Predicted", c='red')
        plt.plot(anomaly_scores[start:end], label="Anomaly Score", c='black')
        plt.legend(fontsize=16)
        plt.show()

class ClassificationVisualization(BaseVisualization):
    def __init__(self, embeddings, labels):
        self.embeddings = np.array(embeddings)
        self.labels = np.array(labels)

    def plot(self):
        from sklearn.decomposition import PCA

        embeddings_manifold = PCA(n_components=2).fit_transform(self.embeddings)

        plt.title("ECG5000 Test Embeddings", fontsize=20)
        plt.scatter(
            embeddings_manifold[:, 0],
            embeddings_manifold[:, 1],
            c=self.labels.squeeze()
        )
        plt.xticks(fontsize=16)
        plt.yticks(fontsize=16)
        plt.show()