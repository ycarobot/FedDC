cd ../../../src || exit

# CIFAR-100 hybrid shift experiments

dataset='cifar100'
num_clients=300
partition='step_2_51' # 2 major class
data_holdout=0.2
client_holdout=0.2
corruption="ood" # train and test use different set of distortions

partition_seed=0

python ./cifar_prepare.py \
  --dataset ${dataset} \
  --num_clients ${num_clients} \
  --partition ${partition} \
  --data_holdout ${data_holdout} \
  --client_holdout ${client_holdout} \
  --corruption ${corruption} \
  --partition_seed ${partition_seed}
