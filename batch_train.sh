start=0
end=`cat config.json | jq '.data_loader.args.num_folds'`
end=$((end-1))

for i in $(eval echo {$start..$end})
do
   python train_Kfold_CV.py --fold_id=$i --device 1 --np_data_dir path/to/numpy/files
done
