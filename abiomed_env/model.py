import pickle
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
try:
    import matplotlib.pyplot as plt
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False
import copy

class TimeSeriesDataset(Dataset):
    def __init__(self, data_all, input_horizon=10, output_horizon=11):
        super().__init__()
        self.input_horizon = input_horizon
        self.output_horizon = output_horizon
        data, pl, labels = self.prep_transformer_world(data_all)
        self.data = data
        self.pl = pl
        self.labels = labels

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.pl[idx], self.labels[idx]

    def prep_transformer_world(self, x_n):
        n = x_n.shape[0]
        # horizon = int(x_n.shape[1]/2)
        # originally we dont take the p-level into the observation space
        # now we do. so x is (N, horizon, columns), pl is (N, horizon) and y is (N, horizon, columns-1)
        x = x_n[:, : self.input_horizon, :]
        y = x_n[
            :, self.input_horizon : self.input_horizon + self.output_horizon, :-1
        ].reshape((n, -1))
        pl = x_n[:, self.input_horizon : self.input_horizon + self.output_horizon, -1]
        return x, pl, y


class Decoder(nn.Module):
    def __init__(
        self, num_layers=2, input_size=82, hidden_size=512, output_size=66, dropout=0
    ):
        # output of the encoder+controlled variable (p-level)
        super(Decoder, self).__init__()
        # print(input_size)
        self.layers = num_layers
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size

        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)

        # Add middle layers if there are more than 2 layers
        if self.layers > 2:
            self.middle_layers = nn.ModuleList()
            for _ in range(self.layers - 2):
                self.middle_layers.append(nn.Linear(hidden_size, hidden_size))
                self.middle_layers.append(nn.ReLU())
                self.middle_layers.append(nn.Dropout(dropout))

        self.final_layer = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        if self.layers > 2:
            for layer in self.middle_layers:
                x = layer(x)
                x = self.relu(x)
                x = self.dropout(x)
        x = self.final_layer(x)

        return x


class TimeSeriesTransformer(nn.Module):
    def __init__(
        self,
        input_dim,
        output_dim,
        dim_model=512,
        num_heads=8,
        num_encoder_layers=3,
        num_decoder_layers=2,
        encoder_dropout=0.1,
        decoder_dropout=0,
        max_len=100,
        forecast_horizon=11,
        device="cpu",
    ):
        super(TimeSeriesTransformer, self).__init__()

        self.device = torch.device(device)
        print("Using device:", self.device)

        self.dim_model = dim_model
        self.input_embedding = nn.Linear(input_dim, dim_model).to(self.device)
        self.transformer_encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=dim_model, nhead=num_heads, dropout=encoder_dropout
            ).to(self.device),
            num_layers=num_encoder_layers,
        ).to(self.device)
        self.dim_model = dim_model

        self.positional_encoding = self.create_positional_encoding(
            max_len, dim_model
        ).to(self.device)

        self.decoder = Decoder(
            num_layers=num_decoder_layers,
            input_size=dim_model + forecast_horizon,
            output_size=output_dim,
            dropout=decoder_dropout,
        ).to(self.device)

    def create_positional_encoding(self, max_len, dim_model):
        pe = torch.zeros(max_len, dim_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, dim_model, 2).float() * -(np.log(10000.0) / dim_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        return nn.Parameter(pe, requires_grad=False)

    def forward(self, src, pl):
        src = self.input_embedding(src.to(self.device)) * np.sqrt(self.dim_model)

        src += self.positional_encoding[:, : src.size(1)].clone().detach()

        src = src.permute(1, 0, 2)  # Transformer expects (seq_len, batch, features)
        encoded_src = self.transformer_encoder(src)
        encoded_src = encoded_src.permute(1, 0, 2)  # Back to (batch, seq_len, features)

        pp = torch.cat([encoded_src[:, -1, :], pl], 1)

        output = self.decoder(pp)  # Use only the last time step
        # dimension of the dec input last dim of encoder
        return output

    def sample_multiple(self, src, pl, num_samples=10):
        self.train()  # Enable dropout during inference
        samples = []
        for _ in range(num_samples):
            output = self.forward(src, pl)
            samples.append(output)
        self.eval() 
        return torch.stack(samples)


def model_factory(input_dim, output_dim,
                dim_model=512, num_heads=8, 
                num_encoder_layers=3, num_decoder_layers=2, 
                encoder_dropout=0.1, decoder_dropout=0, 
                max_len=100, forecast_horizon=12, 
                model_type='transformer', device='cpu'):

    if model_type == 'transformer':
        return TimeSeriesTransformer(input_dim, output_dim, dim_model, num_heads, num_encoder_layers, num_decoder_layers, encoder_dropout, decoder_dropout, max_len, forecast_horizon, device)
    else:
        raise ValueError(f"Invalid model type: {model_type}")


class SoftDTW(nn.Module):
    """Differentiable Soft-DTW loss for batched multivariate time series.

    gamma: smoothing parameter — lower values approximate hard DTW,
           higher values produce a smoother, more forgiving loss.
    """
    def __init__(self, gamma=1.0):
        super().__init__()
        self.gamma = gamma

    def forward(self, pred, target):
        # pred, target: (batch, seq_len, features)
        batch_size, n, _ = pred.shape
        m = target.shape[1]

        # Local cost: squared L2 distance between every pair of frames
        D = torch.cdist(pred, target) ** 2  # (batch, n, m)

        # Soft-DTW recurrence; sentinel inf values at borders handle boundary conditions
        R = torch.full((batch_size, n + 2, m + 2), 1e9, device=pred.device, dtype=pred.dtype)
        R[:, 0, 0] = 0.0

        for i in range(1, n + 1):
            for j in range(1, m + 1):
                r0 = R[:, i - 1, j - 1]
                r1 = R[:, i - 1, j]
                r2 = R[:, i, j - 1]
                # Soft-min via log-sum-exp
                softmin = -self.gamma * torch.logsumexp(
                    torch.stack([-r0, -r1, -r2], dim=1) / self.gamma, dim=1
                )
                R[:, i, j] = D[:, i - 1, j - 1] + softmin

        return R[:, n, m].mean()


class WorldModel(nn.Module):
    def __init__(
        self,
        num_features,
        dim_model=512,
        num_heads=8,
        num_encoder_layers=3,
        num_decoder_layers=2,
        encoder_dropout=0.1,
        decoder_dropout=0,
        max_len=100,
        forecast_horizon=12,
        columns=None,
        model_type='transformer',
        device="cpu",
    ):
        super(WorldModel, self).__init__()
        self.device = torch.device(device)
        self.num_features = num_features

        # outputs everything except for the last feature (P-level), times the forecast horizon
        self.input_dim = num_features
        self.num_outputs = (num_features - 1) * forecast_horizon
        self.dim_model = dim_model
        self.num_heads = num_heads
        self.num_encoder_layers = num_encoder_layers
        self.num_decoder_layers = num_decoder_layers
        self.encoder_dropout = encoder_dropout
        self.decoder_dropout = decoder_dropout
        self.max_len = max_len
        self.forecast_horizon = forecast_horizon
        self.device = torch.device(device)
        self.columns = columns if columns is not None else [i for i in range(0, 13) if i != 11]
        
        print('time series transformer device', device)
        # --- Model ---
        model = model_factory(
            input_dim=self.input_dim,
            output_dim=self.num_outputs,
            dim_model=dim_model,
            num_heads=num_heads,
            num_encoder_layers=num_encoder_layers,
            num_decoder_layers=num_decoder_layers,
            encoder_dropout=encoder_dropout,
            decoder_dropout=decoder_dropout,
            max_len=max_len,
            forecast_horizon=forecast_horizon,
            model_type=model_type,
            device=device,
        )
        self.model = model

    def load_model(self, path):
        # when load model, default state is eval.
        
        state_dict = torch.load(path, map_location=self.device)
        self.model.load_state_dict(state_dict, strict=False)
        self.model.to(self.device)
        self.model.eval()


    def forward(self, src, pl):
        return self.model(src, pl)

    def load_data(self, path): 
        
        with open(path, "rb") as f:
            data = pickle.load(f)
        mean = data["mean"]
        std = data["std"]
        self.mean, self.std = mean, std

        self.data_train = TimeSeriesDataset(
            ((data["train"] - mean) / std)[:, :, self.columns],
            input_horizon = self.forecast_horizon,
            output_horizon=self.forecast_horizon,
        )
        self.data_val = TimeSeriesDataset(
            ((data["val"] - mean) / std)[:, :, self.columns],
            input_horizon=self.forecast_horizon,
            output_horizon=self.forecast_horizon,
        )
        self.data_test = TimeSeriesDataset(
            ((data["test"] - mean) / std)[:, :, self.columns],
            input_horizon=self.forecast_horizon,
            output_horizon=self.forecast_horizon,
        )

        print(
            "loaded datasets with length \n train: ",
            len(self.data_train),
            "\n val: ",
            len(self.data_val),
            "\n test: ",
            len(self.data_test),
        )   

    def train_model(self, num_epochs=50, batch_size=64, learning_rate=0.001, loss_fn = 'mse'):
        self.model.train()

        # --- Loss and Optimizer ---
        optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate)
        if loss_fn == 'mse':
            criterion = torch.nn.MSELoss()
        elif loss_fn == 'mae':
            criterion = torch.nn.L1Loss()
        else:
            raise ValueError(f"Invalid loss function: {loss_fn}")
        
        train_loader = DataLoader(self.data_train, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(self.data_val, batch_size=batch_size, shuffle=False)
        test_loader = DataLoader(self.data_test, batch_size=batch_size, shuffle=False)
        # --- Training Loop ---
        best_model = None
        best_val_loss = float("inf")

        train_loss_list = []
        val_loss_list = []
        val_map_mae_list = []

        for epoch in range(num_epochs):
            self.model.train()
            train_loss = 0
            for x, pl, y in train_loader:
                x, pl, y = (
                    x.to(self.device).float(),
                    pl.to(self.device).float(),
                    y.to(self.device).float(),
                )
                optimizer.zero_grad()
                output = self.model(x, pl)
                loss = criterion(output, y)
                loss.backward()
                optimizer.step()
                train_loss += loss.item() * x.size(0)
            train_loss /= len(train_loader.dataset)

            # --- Validation ---
            self.model.eval()
            val_loss = 0
            val_map_mae = 0
            with torch.no_grad():
                for x, pl, y in val_loader:
                    x, pl, y = (
                        x.to(self.device).float(),
                        pl.to(self.device).float(),
                        y.to(self.device).float(),
                    )
                    output = self.model(x, pl)
                    loss = criterion(output, y)

                    output_formated = output.reshape(
                        -1, self.forecast_horizon, self.num_features - 1
                    )
                    y_formated = y.reshape(
                        -1, self.forecast_horizon, self.num_features - 1
                    )
                    map_mae = criterion(output_formated[:, :, 0], y_formated[:, :, 0])
                    val_loss += loss.item() * x.size(0)
                    val_map_mae += map_mae.item() * x.size(0)
            val_loss /= len(val_loader.dataset)
            val_map_mae /= len(val_loader.dataset)
            train_loss_list.append(train_loss)
            val_loss_list.append(val_loss)
            val_map_mae_list.append(val_map_mae)

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_model = self.model.state_dict()
                print(f"New best model with val loss: {val_loss:.4f}")

            print(
                f"Epoch {epoch+1}/{num_epochs} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Val MAPE: {val_map_mae:.4f}"
            )
            if epoch % 10 == 0:
                self.test(loss_fn=loss_fn)

        self.model.load_state_dict(best_model)
        print("Best model validation loss: {:.4f}".format(best_val_loss))
        print(
            f"Training loss: {np.mean(train_loss_list[-15:]):.3f} with std {np.std(train_loss_list[-15:]):.3f}"
        )
        print(
            f"Validation loss: {np.mean(val_loss_list[-15:]):.3f} with std {np.std(val_loss_list[-15:]):.3f}"
        )
        print(
            f"Validation MAPE: {np.mean(val_map_mae_list[-15:])*self.std[0]:.3f} \
              with std {np.std(val_map_mae_list[-15:])*self.std[0]:.3f}"
        )

        return best_model

    def train_model_softdtw(self, num_epochs=50, batch_size=64, learning_rate=0.001, gamma=1.0, sdtw_weight=0.008, mse_weight=1.0):
        """Train with a combined Soft-DTW + MSE loss.

        loss = sdtw_weight * softdtw(pred, target) + mse_weight * mse(pred, target)

        gamma controls SoftDTW smoothing (lower ≈ hard DTW).
        Tune sdtw_weight and mse_weight independently to balance shape vs. pointwise accuracy.
        Note: SoftDTW values are ~seq_len times larger than MSE, so sdtw_weight ~0.1 and mse_weight=1.0 roughly balances them.
        """
        sdtw_criterion = SoftDTW(gamma=gamma)
        mse_criterion = torch.nn.MSELoss()
        mae_criterion = torch.nn.L1Loss()
        optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate)

        train_loader = DataLoader(self.data_train, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(self.data_val, batch_size=batch_size, shuffle=False)

        best_model = None
        best_val_loss = float("inf")
        train_loss_list = []
        val_loss_list = []
        val_map_mae_list = []

        for epoch in range(num_epochs):
            self.model.train()
            train_loss = 0
            for x, pl, y in train_loader:
                x, pl, y = (
                    x.to(self.device).float(),
                    pl.to(self.device).float(),
                    y.to(self.device).float(),
                )
                optimizer.zero_grad()
                output = self.model(x, pl)
                output_seq = output.reshape(-1, self.forecast_horizon, self.num_features - 1)
                y_seq = y.reshape(-1, self.forecast_horizon, self.num_features - 1)
                loss = sdtw_weight * sdtw_criterion(output_seq, y_seq) + mse_weight * mse_criterion(output_seq, y_seq)
                loss.backward()
                optimizer.step()
                train_loss += loss.item() * x.size(0)
            train_loss /= len(train_loader.dataset)

            # Validation — report combined val loss and MAE on MAP column
            self.model.eval()
            val_loss = 0
            val_map_mae = 0
            with torch.no_grad():
                for x, pl, y in val_loader:
                    x, pl, y = (
                        x.to(self.device).float(),
                        pl.to(self.device).float(),
                        y.to(self.device).float(),
                    )
                    output = self.model(x, pl)
                    output_seq = output.reshape(-1, self.forecast_horizon, self.num_features - 1)
                    y_seq = y.reshape(-1, self.forecast_horizon, self.num_features - 1)
                    loss = sdtw_weight * sdtw_criterion(output_seq, y_seq) + mse_weight * mse_criterion(output_seq, y_seq)
                    map_mae = mae_criterion(output_seq[:, :, 0], y_seq[:, :, 0])
                    val_loss += loss.item() * x.size(0)
                    val_map_mae += map_mae.item() * x.size(0)
            val_loss /= len(val_loader.dataset)
            val_map_mae /= len(val_loader.dataset)

            train_loss_list.append(train_loss)
            val_loss_list.append(val_loss)
            val_map_mae_list.append(val_map_mae)

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_model = self.model.state_dict()
                print(f"New best model with val loss: {val_loss:.4f}")

            print(
                f"Epoch {epoch+1}/{num_epochs} | Train Loss: {train_loss:.4f} | "
                f"Val Loss: {val_loss:.4f} | Val MAP MAE: {val_map_mae:.4f}"
            )
            if epoch % 10 == 0:
                self.test(loss_fn='mae')

        self.model.load_state_dict(best_model)
        print("Best model validation loss: {:.4f}".format(best_val_loss))
        print(
            f"Training loss: {np.mean(train_loss_list[-15:]):.3f} with std {np.std(train_loss_list[-15:]):.3f}"
        )
        print(
            f"Validation loss: {np.mean(val_loss_list[-15:]):.3f} with std {np.std(val_loss_list[-15:]):.3f}"
        )
        print(
            f"Validation MAP MAE: {np.mean(val_map_mae_list[-15:])*self.std[0]:.3f} "
            f"with std {np.std(val_map_mae_list[-15:])*self.std[0]:.3f}"
        )
        return best_model

    def test_output(self):
        test_loader = DataLoader(self.data_test, batch_size=64, shuffle=False)
        outputs = []
        ys = []
        pls = []
        for batch in test_loader:
            x, pl, y = batch
            x, pl, y = x.to(self.device).float(), pl.to(self.device).float(), y.to(self.device).float()
            output = self.model(x, pl)
            output_formated = output.reshape(
                -1, self.forecast_horizon, self.num_features - 1)
            outputs.append(output_formated)
            ys.append(y.reshape(-1, self.forecast_horizon, self.num_features - 1))
            pls.append(pl)
        return outputs, ys, pls
    
    def test_output_multiple(self, num_samples=10):
        test_loader = DataLoader(self.data_test, batch_size=64, shuffle=False)
        outputs = []
        ys = []
        pls = []
        self.model.train()
        with torch.no_grad():
            for batch in test_loader:
                x, pl, y = batch
                x, pl, y = x.to(self.device).float(), pl.to(self.device).float(), y.to(self.device).float()
                outputs_samples= []
                for sample in range(num_samples):
                    output = self.model(x, pl)
                    output_formated = output.reshape(
                        -1, self.forecast_horizon, self.num_features - 1)
                    outputs_samples.append(output_formated)
                outputs.append(torch.stack(outputs_samples, dim=1))
                ys.append(y.reshape(-1, self.forecast_horizon, self.num_features - 1))
                pls.append(pl)
        self.model.eval()
        return outputs, ys, pls
    
    def test(self, loss_fn='mse'):
        self.model.eval()
        outputs, ys, pls = self.test_output()
        if loss_fn == 'mse':
            criterion = torch.nn.MSELoss()
        elif loss_fn == 'mae':
            criterion = torch.nn.L1Loss()
        else:
            raise ValueError(f"Invalid loss function: {loss_fn}")
        mse = criterion(torch.cat(outputs, dim=0), torch.cat(ys, dim=0))
        print(f"Final test {loss_fn}: {mse.item()}")
        map_mae = criterion(outputs[0][:, :, 0], ys[0][:, :, 0])
        print(f"Final test MAP {loss_fn}: {map_mae.item() * self.std[0]:.3f}")
        return mse.detach().item(), map_mae.detach().item() * self.std[0]
    
    def save_model(self, path):
        #deepcopy model to cpu before saving
        model_copy = copy.deepcopy(self.model)
        torch.save(model_copy.to('cpu').state_dict(), path)
        del model_copy

    def step(self, x, pl):
        """
        x is (N, horizon, columns)
        pl is (N, horizon)
        """

        if isinstance(pl, int):
            # step takes int pl as action
            # we need to convert it to a tensor of shape (N, horizon)
            new_pl = torch.tensor([pl] * self.forecast_horizon).to(self.device).unsqueeze(0)
            pl = (new_pl - self.mean[-1]) / self.std[-1]

        x, pl = x.to(self.device).float(), pl.to(self.device).float()
        
        output = self.model(x, pl)
        output_formated = output.reshape(
            -1, self.forecast_horizon, self.num_features - 1
        )
        full_output = torch.cat([output_formated, pl.unsqueeze(2)], dim=2)
    
        return full_output

    def sample_autoregressive(
        self, x, steps, custom_pl=None, batch_data=None
    ):
        """
        batch_data: (x, pl, y)
        steps: int, total number of steps to predict
        custom_pl: list of pls, pls can be a single int or a tensor of shape (1, forecast_horizon)
        """
        x = x.to(self.device).float()

        pl_control = (
            [batch_data[i][1] for i in range(len(batch_data))]
            if custom_pl is None
            else custom_pl
        )

        output = []
        x_in = x
        for i in range(steps):
            # print(i, x_in.shape)
            pl_here = pl_control[i]
            pred = self.step(x_in, pl_here)
            x_in = pred
            output.append(x_in.detach())
        
        return output
    
    def sample_autoregressive_multiple(self, x, steps, custom_pl=None, batch_data=None, sample_size=1):
        self.model.train()
        outputs = []
        for sample in range(sample_size):
            output = self.sample_autoregressive(x, steps, custom_pl, batch_data)
            outputs.append(output)
        self.model.eval()
        return outputs

    def unnorm_state_col(self, col_idx, state_vectors):
        mean_val_tensor = self.mean[col_idx]
        std_val_tensor = self.std[col_idx]
        mean_val = mean_val_tensor.detach().cpu().numpy()
        std_val = std_val_tensor.detach().cpu().numpy()

        state_vectors_np = np.array(state_vectors)
        normalized_col = state_vectors_np[:, :, col_idx]
        unnormalized_col = (normalized_col * std_val) + mean_val
        return unnormalized_col
    
    def unnorm_state_vectors(self, state_vectors):

        normalized_states_np = np.array(state_vectors)
        original_shape = normalized_states_np.shape
        num_features = normalized_states_np.shape[-1]
        reshaped_states = normalized_states_np.reshape(-1, num_features)
        means = self.mean.detach().cpu().numpy()
        stds = self.std.detach().cpu().numpy()
        #first 12 cols
        means_sliced = means[:num_features]
        stds_sliced = stds[:num_features]
        unnormalized_reshaped_states = (reshaped_states * stds_sliced) + means_sliced
        unnormalized_states = unnormalized_reshaped_states.reshape(original_shape)
        return unnormalized_states
    
    def unnorm_output(self, output, ignore_pl=False):
        if not isinstance(output, torch.Tensor):
            output = torch.tensor(output).to(self.device)
        if ignore_pl:
            return output.cpu() * self.std[self.columns[:-1]] + self.mean[self.columns[:-1]]
        else:
            return output.cpu() * self.std[self.columns] + self.mean[self.columns]
    
    def unnorm_y(self, batch_data):
        all_y = []
        for i in range(len(batch_data)):
            y = batch_data[i][2]
            y_reshape = y.reshape(-1, self.forecast_horizon, self.num_features - 1)
            y_full = torch.cat([y_reshape, batch_data[i][1].unsqueeze(2)], dim=2)
            y_full_unnorm = y_full.cpu() * self.std[self.columns] + self.mean[self.columns]
            all_y.append(y_full_unnorm)
        return all_y
    

    def unnorm_pl(self, pl):
        if not isinstance(pl, torch.Tensor):
            pl = torch.tensor(pl).to(self.device)
        return pl.cpu() * self.std[-1] + self.mean[-1]
    
    def normalize_pl(self, pl):
        return (pl - self.mean[-1]) / self.std[-1]



    def plot_output(self, batch_data, pred, need_unnorm=True):
        if not MATPLOTLIB_AVAILABLE:
            print("Warning: matplotlib not available. Plotting skipped.")
            return
        
        # unnorm x
        x_unnorm = batch_data[0][0].cpu() * self.std[self.columns] + self.mean[self.columns]
        all_y = torch.cat(self.unnorm_y(batch_data), dim=1)

        if not isinstance(pred[0], list):
            pred = [pred]
        
        pred_example = pred[0]
        pred_example = torch.cat(pred_example, dim=1).detach().cpu()
        if need_unnorm: 
            pred_full_unnorm = pred_example.cpu() * self.std[self.columns] + self.mean[self.columns]
        else:
            pred_full_unnorm = pred_example.cpu()
            
        in_len = x_unnorm.shape[1]
        out_len = pred_full_unnorm.shape[1]

        fig, ax1 = plt.subplots()
        # fig.suptitle("MSE: " + str(loss.item()))

        ax1.xaxis.set_major_locator(plt.MaxNLocator(6))
        ax1.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f"{x/6:.1f}"))

        color = "tab:red"
        ax1.set_xlabel("time (hr)")
        ax1.set_ylabel("MAP", color=color)
        ax1.plot(np.arange(0, in_len, 1), x_unnorm[0, :, 0], color=color, linewidth=0.5)

        # multiple samples
        for i in range(len(pred)):
            if i == 0:
                # Only add label for the first prediction
                self.plot_only_output(in_len, pred[i], ax1, need_unnorm=need_unnorm, alpha=np.sqrt(1.0/len(pred)))
                ax1.lines[-1].set_label("prediction")
            else:
                self.plot_only_output(in_len, pred[i], ax1, need_unnorm=need_unnorm, alpha=np.sqrt(1.0/len(pred)))

        ax1.plot(
            np.arange(in_len, in_len + all_y.shape[1], 1),
            all_y[0, :, 0],
            color="red",
            label="true",
            linestyle="--",
            linewidth=0.5,
        )
        ax1.legend()
        ax1.tick_params(axis="y", labelcolor=color)

        ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis

        color = "tab:blue"
        ax2.set_ylabel(
            "P-Level", color=color
        )  # we already handled the x-label with ax1
        ax2.plot(np.arange(0, in_len, 1), x_unnorm[0, :, 11], color=color)
        ax2.plot(
            np.arange(in_len, in_len + out_len, 1),
            pred_full_unnorm[0, :, 11],
            color=color,
        )
        ax2.tick_params(axis="y", labelcolor=color)
        ax2.set_ylim(0, 10)

        fig.tight_layout()  # otherwise the right y-label is slightly clipped
        plt.show()

