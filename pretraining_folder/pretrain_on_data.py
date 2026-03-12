#from process_dataset_files import process_dataset_files

import torch
from torch.utils.data import Dataset, DataLoader
from torch import nn
from torch.optim import AdamW
import numpy as np
from src.samay.models.lptm.model.backbone import LPTMPipeline, TASKS
from src.samay.models.lptm.model.masktrain import Masking
from src.samay.models.lptm.utils import parse_config

import pandas as pd


# Created this class because LPTM dataset class in dataset.py can't process reconstruction (was throwing errors)
class SimpleLPTMDataset(Dataset):
    def __init__(self, path, seq_len=128, horizon=0, stride=10, batchsize=16, mode="train"):
        self.data_path = path
        self.seq_len = seq_len
        self.forecast_horizon = horizon
        self.stride = stride
        self.batchsize = batchsize
        self.mode = mode
        
        self._read_data()
        
        self.one_chunk_num = max(1, (
            self.length_timeseries - self.seq_len - self.forecast_horizon
        ) // self.stride + 1)
        self.num_chunks = 1 

    def _read_data(self):
        self.df = pd.read_csv(self.data_path, header=None)
        self.data = self.df.values
        
        self.data = (self.data - self.data.mean()) / self.data.std()
        
        self.length_timeseries = self.data.shape[0]
        self.n_channels = self.data.shape[1]

    def __len__(self):
        return self.num_chunks * self.one_chunk_num

    def __getitem__(self, idx):
        seq_start = (idx % self.one_chunk_num) * self.stride
        seq_end = seq_start + self.seq_len
        
        input_seq = self.data[seq_start:seq_end, :].T  
        input_mask = torch.ones(self.seq_len, dtype=torch.float)
        
        return input_seq, input_mask

    def get_data_loader(self):
        return DataLoader(self, batch_size=self.batchsize, shuffle=True)


#Just processed five datasets to test the program itself. May need to edit paths slightly
dataset_list = ["csv_output/bull/sequence_1.csv", "csv_output/kdd2022/sequence_1.csv", "csv_output/cockatoo/sequence_1.csv", "csv_output/elecdemand/sequence_1.csv", "csv_output/bdg-2_fox/sequence_1.csv"]
epochs = 5
lr = 1e-4
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

for i in range (len(dataset_list)):

    dataset = SimpleLPTMDataset(
        path=dataset_list[i],
        mode="train"
    )

    dataloader = dataset.get_data_loader()


    model_config = {
        "task_name": TASKS.RECONSTRUCTION, 
        "d_model": 768,  
        "seq_len": 128,
        "patch_len": 8,
        "patch_stride_len": 8,
        "transformer_backbone": "google/flan-t5-base",
        "transformer_type": "encoder_only",
        "t5_config": {
            "d_model": 768,  
            "num_layers": 12,
            "num_heads": 12,
            "dropout": 0.1
        }
    }

    config_obj = parse_config(model_config)
    model = LPTMPipeline(config=config_obj)
    model.init()  
    model = model.to(DEVICE)

    mask_generator = Masking(mask_ratio=0.25, patch_len=8)
    criterion = nn.MSELoss()
    optimizer = AdamW(model.parameters(), lr=lr)


    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for batch_idx, batch in enumerate(dataloader):
            input_seq, input_mask = batch
            x = input_seq.float().to(DEVICE) 
            input_mask = input_mask.float().to(DEVICE) 
            
            mask = mask_generator.generate_mask(x=x, input_mask=input_mask).to(DEVICE)

            optimizer.zero_grad()

            outputs = model(x_enc=x, input_mask=input_mask, mask=mask)  
            preds = outputs.reconstruction         

            loss = criterion(preds, x)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(dataloader)
        print(f"Epoch {epoch+1} - Loss: {avg_loss}")


    print("Pretraining completed")