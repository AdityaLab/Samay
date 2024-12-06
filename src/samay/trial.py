from .model import MomentModel
from .dataset import MomentDataset


def main():
    # dataset = TimesfmDataset(name="tycho", path='/nethome/sli999/data/Tycho/timesfm_US_covid_pivot.csv')
    dataset = MomentDataset(name="ett", datetime_col='date', path='/nethome/sli999/TSFMProject/src/tsfmproject/models/moment/data/ETTh1.csv')

    # repo = "google/timesfm-1.0-200m-pytorch"
    repo = "AutonLab/MOMENT-1-large"
    # config = {
    #     "context_len": 128,
    #     "horizon_len": 32,
    #     "backend": "gpu",
    #     "per_core_batch_size": 32,
    #     "input_patch_len": 32,
    #     "output_patch_len": 128,
    #     "num_layers": 20,
    #     "model_dims": 1280,
    #     "quantiles": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9],
    # }
    config = {
        'task_name': 'forecasting',
        'forecast_horizon': 192,
        'head_dropout': 0.1,
        'weight_decay': 0,
        'freeze_encoder': True, # Freeze the patch embedding layer
        'freeze_embedder': True, # Freeze the transformer encoder
        'freeze_head': False, # The linear forecasting head must be trained
    }
    
    # tfm = TimesfmModel(config=config, repo=repo)
    mmt = MomentModel(config=config, repo=repo)
    # finetuned_model = tfm.finetune(dataset)
    finetuned_model = mmt.finetune(dataset)


if __name__ == "__main__":
    main()