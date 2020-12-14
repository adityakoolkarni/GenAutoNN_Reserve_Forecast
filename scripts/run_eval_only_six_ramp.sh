#python train_and_eval.py --num_epochs 30 --learning_rate 4e-3 --num_hidden 300  --seasonality 'HD' --num_layers 1 --six_ramps 1 --context_len 168 --pred_len 12

#python train_and_eval.py --num_epochs 30 --learning_rate 4e-3 --num_hidden 300  --seasonality 'H' --num_layers 1 --six_ramps 1 --context_len 168 --pred_len 12

####
python train_and_eval.py --num_epochs 70 --learning_rate 4e-3 --num_hidden 300  --seasonality 'HD' --num_layers 1 --six_ramps 1 --context_len 168 --pred_len 24 --run_eval_only True --simulation_folder_name 'six_ramps_final'
####
###
python train_and_eval.py --num_epochs 70 --learning_rate 4e-3 --num_hidden 300  --seasonality 'HM' --num_layers 1 --six_ramps 1 --context_len 168 --pred_len 24 --run_eval_only True --simulation_folder_name 'seasonality_hm_exp'
###

python train_and_eval.py --num_epochs 70 --learning_rate 4e-3 --num_hidden 300  --seasonality 'HD' --num_layers 1 --six_ramps 1 --context_len 48 --pred_len 24 --run_eval_only True --simulation_folder_name 'context_len_exp'

python train_and_eval.py --num_epochs 70 --learning_rate 4e-3 --num_hidden 300  --seasonality 'HD' --num_layers 1 --six_ramps 0 --context_len 168 --pred_len 24 --run_eval_only True --simulation_folder_name 'simulation_num_10'
python train_and_eval.py --num_epochs 70 --learning_rate 4e-3 --num_hidden 300  --seasonality 'D' --num_layers 1 --six_ramps 1 --context_len 168 --pred_len 24 --run_eval_only True --simulation_folder_name 'season_d_alone'
