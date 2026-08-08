import torch

# 10 min 1 hr window
model_kwargs_10min_1hr_full = {
    'num_features': 12,
    'forecast_horizon': 6,
    'dim_model': 256,
    'num_heads': 8,
    'num_encoder_layers': 3,
    'num_decoder_layers': 2,
    'encoder_dropout': 0.1,
    'decoder_dropout': 0,
    'max_len': 100,
    'device': torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
}

model_configs = {
    "10min_1hr_all_data": model_kwargs_10min_1hr_full,
}