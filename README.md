# Foundational-MTS

Commands to run:

Channel Temporal attention model

Electricity
```
python run_train_exp.py --model chtp --dataset electricity --seq_len 96 --pred_len 96 --num_epochs 500 --ch_proj_len 128 
```
EttH1
```
python run_train_exp.py --model chtp --dataset etth1 --seq_len 96 --pred_len 96 --num_epochs 500 --ch_proj_len 128 
```
EttH2
```
python run_train_exp.py --model chtp --dataset etth2 --seq_len 96 --pred_len 96 --num_epochs 500 --ch_proj_len 128 
```
EttM1
```
python run_train_exp.py --model chtp --dataset ettm1 --seq_len 96 --pred_len 96 --num_epochs 500 --ch_proj_len 128 
```
Traffic
```
python run_train_exp.py --model chtp --dataset traffic --seq_len 96 --pred_len 96 --num_epochs 500 --ch_proj_len 128 
```
iTransformer

EttH2

```
/scratch/vg2523/mts/bin/python run_train_exp.py --model iTransformer --dataset etth2 --seq_len 96 --pred_len 96 --num_epochs 500 
```
Traffic
```
/scratch/vg2523/mts/bin/python run_train_exp.py --model iTransformer --dataset traffic --seq_len 96 --pred_len 96 --num_epochs 150
```

PatchTST

ETTH2
```
python run_longExp.py --random_seed 2021 --is_training 1 --root_path datasets/ETT-small --data_path ETTh2.csv --model_id ETTh296_96 --model PatchTST --data ETTh2 --features M --seq_len 96 --pred_len 96 --enc_in 7 --e_layers 3 --n_heads 4 --d_model 16 --d_ff 128 --dropout 0.3 --fc_dropout 0.3 --head_dropout 0 --patch_len 16 --stride 8 --des 'Exp' --train_epochs 100 --itr 1 --batch_size 128 --learning_rate 0.0001
```
Electricty
```
python run_longExp.py --random_seed 2021 --model_id PatchTST_attn_Electricity_96_96 --is_training 1 --root_path datasets/electricity --data_path electricity.csv --model PatchTST --data custom --features M --seq_len 96 --pred_len 96 --enc_in 321 --e_layers 3 --n_heads 16 --d_model 128 --d_ff 256 --dropout 0.2 --fc_dropout 0.2 --head_dropout 0 --patch_len 16 --stride 8 --des 'Exp' --train_epochs 100 --patience 10 --lradj 'TST' --pct_start 0.2 --itr 1 --batch_size 32 --learning_rate 0.0001
```
Traffic
```
python run_longExp.py --random_seed 2021 --model_id PatchTST_attn_Traffic_96_96 --is_training 1 --root_path datasets/traffic --data_path traffic.csv --model PatchTST --data custom --features M --seq_len 96 --pred_len 96 --enc_in 862 --e_layers 3 --n_heads 16 --d_model 128 --d_ff 256 --dropout 0.2 --fc_dropout 0.2 --head_dropout 0 --patch_len 16 --stride 8 --des 'Exp' --train_epochs 100 --patience 10 --lradj 'TST' --pct_start 0.2 --itr 1 --batch_size 24 --learning_rate 0.0001
```