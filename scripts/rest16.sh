output_dir=model_weight/roberta_rest16
mkdir -p $output_dir 

CUDA_VISIBLE_DEVICES=0 python -m main_total -ds_name 16semeval_rest -bs 32 -learning_rate 0.00003 -dropout_rate 0.1 -n_epoch 20 -model KGNN -dim_w 768 -dim_k 200 -kge distmult -gcn 0 -is_test 0 -is_bert 2 -save_dir $output_dir 2>&1 | tee $output_dir/train-roberta-rest.log

output_dir=model_weight/bert_rest16
mkdir -p $output_dir 

CUDA_VISIBLE_DEVICES=0 python -m main_total -ds_name 16semeval_rest -bs 32 -learning_rate 0.00003 -dropout_rate 0.3 -n_epoch 20 -model KGNN -dim_w 768 -dim_k 200 -kge distmult -gcn 0 -is_test 0 -is_bert 1 -save_dir $output_dir 2>&1 | tee $output_dir/train-bert-rest.log


output_dir=model_weight/glove_rest16
mkdir -p $output_dir

CUDA_VISIBLE_DEVICES=0 python -m main_total -ds_name 16semeval_rest -bs 32 -learning_rate 0.001 -dropout_rate 0.3 -n_epoch 20 -model KGNN -dim_w 300 -dim_k 200 -kge distmult -gcn 0 -is_test 0 -is_bert 0 -save_dir $output_dir 2>&1 | tee $output_dir/train-rest.log