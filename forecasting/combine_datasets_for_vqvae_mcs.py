"""
Variant of combine_datasets_for_vqvae.py for a compression_factor=2, length-6
generalist pool sized for Abiomed/MCS's clinical window (6 steps @ 10-min
resolution), instead of the standard length-96 pool used by the public
CF=4 generalist. Combines the length-6-rewindowed public datasets (see
save_revin_data.py --seq_len 6 --pred_len 6) with Abiomed's own length-6 data.
"""
import argparse
import numpy as np
import os


def create_full_data(args, split='train'):
    weather = np.load(f'{args.root_path}/weather_Tin6/{split}_data_x.npy')
    electricity = np.load(f'{args.root_path}/electricity_Tin6/{split}_data_x.npy')
    traffic = np.load(f'{args.root_path}/traffic_Tin6/{split}_data_x.npy')
    ETTm1 = np.load(f'{args.root_path}/ETTm1_Tin6/{split}_data_x.npy')
    ETTm2 = np.load(f'{args.root_path}/ETTm2_Tin6/{split}_data_x.npy')
    ETTh1 = np.load(f'{args.root_path}/ETTh1_Tin6/{split}_data_x.npy')
    ETTh2 = np.load(f'{args.root_path}/ETTh2_Tin6/{split}_data_x.npy')
    abiomed = np.load(f'{args.root_path}/abiomed/{split}_data_x.npy')

    data = np.concatenate(
        (weather, electricity, traffic, ETTm1, ETTm2, ETTh1, ETTh2, abiomed), axis=0
    )

    if not os.path.exists(args.save_path):
        os.makedirs(args.save_path)
    np.save(os.path.join(args.save_path, f'{split}_data_x.npy'), data)
    print(split, data.shape)


def main(args):
    for split in ['train', 'val', 'test']:
        create_full_data(args, split)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--root_path', type=str, default='forecasting/data', help='root path containing the per-dataset dirs')
    parser.add_argument('--save_path', type=str, default='forecasting/data/all_mcs', help='where to save the combined data')
    args = parser.parse_args()
    main(args)
