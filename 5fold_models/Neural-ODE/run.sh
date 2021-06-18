
python process_data.py
for fold in 1 2 3 4 5; do
    for model in 1 2 3 4 5; do
        python data_split.py --data data.csv --fold $fold --model $model

        CUDA_VISIBLE_DEVICES="" python run_train.py --fold $fold --model $model --save fold_$fold --lr 0.00005 --tol 1e-4 --epochs 30 --l2 0.1
        CUDA_VISIBLE_DEVICES="" python run_predict.py --fold $fold --model $model --save fold_$fold --tol 1e-4
    done
done    
