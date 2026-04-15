cd ../../../src || exit

gpu=2

dataset='cifar100'
num_clients=300
partition='step_2_51'
data_holdout=0.2
client_holdout=0.2
corruption="ood"

model='resnet18'
algorithm='fedawi'
gm_rounds=200
part_rate=1.0

lm_lr=0.01
lm_epochs=1
batch_size=20

partition_seed=0

fluctuate_type=5
pre_train_init=0.3


for seed in {0..0}; do
  {
    save_model_path="../weights/cifar100/hybrid/pretrain_${algorithm}_${model}_pseed_${partition_seed}_seed_${seed}.pkl"
    history_path="../history/cifar100/hybrid/pretrain_${algorithm}_${model}_pseed_${partition_seed}_seed_${seed}.pkl"

    CUDA_VISIBLE_DEVICES=${gpu} python main.py \
      --dataset ${dataset} \
      --num_clients ${num_clients} \
      --partition ${partition} \
      --data_holdout ${data_holdout} \
      --client_holdout ${client_holdout} \
      --partition_seed ${partition_seed} \
      --corruption ${corruption} \
      --model ${model} \
      --algorithm ${algorithm} \
      --gm_rounds ${gm_rounds} \
      --part_rate ${part_rate} \
      --lm_lr ${lm_lr} \
      --lm_epochs ${lm_epochs} \
      --batch_size ${batch_size} \
      --seed ${seed} \
      --cuda \
      --history_path ${history_path} \
      --save_model_path ${save_model_path} \
      --fluctuate_type ${fluctuate_type} \
      --pre_train_init ${pre_train_init}\
    } 
  }
done
