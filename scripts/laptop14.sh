output_dir=model_weight/roberta_laptop14
mkdir -p $output_dir 

CUDA_VISIBLE_DEVICES=1 python -m main_total -ds_name 14semeval_laptop -bs 32 -learning_rate 0.00003 -n_epoch 20 -model KGNN -dim_w 768 -dim_k 400 -kge analogy  -gcn 0  -is_test 0 -is_bert 2 2>&1 | tee $output_dir/train-roberta-laptop.log

output_dir=model_weight/bert_laptop14
mkdir -p $output_dir 

CUDA_VISIBLE_DEVICES=1 python -m main_total -ds_name 14semeval_laptop -bs 32 -learning_rate 0.00003 -n_epoch 20 -model KGNN -dim_w 768 -dim_k 400 -kge analogy -gcn 0  -is_test 0 -is_bert 1 -save_dir $output_dir 2>&1 | tee $output_dir/train-bert-laptop.log


output_dir=model_weight/glove_laptop14
mkdir -p $output_dir

CUDA_VISIBLE_DEVICES=1 python -m main_total -ds_name 14semeval_laptop -bs 32 -learning_rate 0.001 -dropout_rate 0.5 -n_epoch 20 -model KGNN -dim_w 300 -dim_k 400 -kge analogy  -gcn 0  -is_test 0 -is_bert 0 -save_dir $output_dir 2>&1 | tee $output_dir/train-laptop.log
