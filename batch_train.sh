gpu=$1
data=$2

start=0
end=`cat config.json | jq '.data_loader.args.num_folds'`
end=$((end-1))

for i in $(eval echo {$start..$end})
do
   python train_Kfold_CV.py --fold_id=$i --device $gpu --np_data_dir $data
done
