# Flag3D
Implementation of TSA based on OpenMMLab's MMEngine
## Training
```
python train.py configs/baseline-g1xb32-lr1e-4/baseline-g1xb32-lr1e-4.py --auto-scale-lr     # launcher: none GPUS[1] x batch_size[21]
bash tools/slurm_train.sh configs/baseline-g1xb32-lr1e-4/baseline-g1xb32-lr1e-4.py           # launcher: slurm 
bash tools/dist_train.sh configs/baseline-g1xb32-lr1e-4/baseline-g1xb32-lr1e-4.py 1          # launcher: pytorch
```
