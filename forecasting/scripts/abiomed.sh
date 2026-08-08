# Abiomed world-model dataset: 12-step episodes (6 in + 6 out) at 10-min resolution, 12 channels.
# Run all commands from the repo root.
#
# Generate forecasting/data_raw/abiomed/{train,val,test}_data.npy first:
#   python forecasting/extract_abiomed_data.py

seq_len=6
root_path_name=forecasting/data_raw/abiomed
data_path_name=unused.npy
data_name=abiomed
random_seed=2021
pred_len=6
gpu=0

python -u forecasting/save_revin_data.py \
  --random_seed $random_seed \
  --data $data_name \
  --root_path $root_path_name \
  --data_path $data_path_name \
  --features M \
  --seq_len $seq_len \
  --pred_len $pred_len \
  --label_len 0 \
  --enc_in 12 \
  --gpu $gpu\
  --save_path "forecasting/data/abiomed"

gpu=0
python forecasting/train_vqvae.py \
  --config_path forecasting/scripts/abiomed.json \
  --model_init_num_gpus $gpu \
  --data_init_cpu_or_gpu cpu \
  --comet_log \
  --comet_tag pipeline \
  --comet_name vqvae_abiomed \
  --save_path "forecasting/saved_models/abiomed/"\
  --base_path "forecasting/data"\
  --batchsize 1024


seq_len=6
random_seed=2021
root_path_name=forecasting/data_raw/abiomed
data_path_name=unused.npy
model_id_name=abiomed
data_name=abiomed
gpu=0
pred_len=6
python -u forecasting/extract_forecasting_data.py \
  --random_seed $random_seed \
  --data $data_name \
  --root_path $root_path_name \
  --data_path $data_path_name \
  --features M \
  --seq_len $seq_len \
  --pred_len $pred_len \
  --label_len 0 \
  --enc_in 12\
  --gpu $gpu\
  --save_path "forecasting/data/abiomed/Tin"$seq_len"_Tout"$pred_len"/"\
  --trained_vqvae_model_path 'forecasting/saved_models/abiomed/CD64_CW256_CF2_BS1024_ITR15000/checkpoints/final_model.pth'\
  --compression_factor 2 \
  --classifiy_or_forecast "forecast"

# ---- Stage 3: LLM-based autoregressive forecaster head ---- #
gpu=0
Tin=6
Tout=6
datatype=abiomed
trained_vqvae_path='forecasting/saved_models/abiomed/CD64_CW256_CF2_BS1024_ITR15000/checkpoints/final_model.pth'
for seed in 2021 1 13
do
python forecasting/train_llm_forecaster.py \
  --data-type $datatype \
  --Tin $Tin \
  --Tout $Tout \
  --cuda-id $gpu \
  --seed $seed \
  --data_path "forecasting/data/"$datatype"/Tin"$Tin"_Tout"$Tout"" \
  --codebook_size 256 \
  --compression 2 \
  --trained_vqvae_model_path $trained_vqvae_path \
  --llm_name "Qwen/Qwen2.5-0.5B" \
  --lora_r 8 \
  --lora_alpha 16 \
  --checkpoint \
  --checkpoint_path "forecasting/saved_models/"$datatype"/llm_forecaster_checkpoints/"$datatype"_Tin"$Tin"_Tout"$Tout"_seed"$seed""\
  --file_save_path "forecasting/results_llm/"$datatype"/"
done
