start=0
end=19

for i in $(eval echo {$start..$end})
do
   python train_Kfold_CV.py --fold_id=$i --device 1 --np_data_dir path/to/numpy/files
done
