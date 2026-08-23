repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$repo_root" || exit

gpu=2

dataset='cifar100'
num_clients=300
partition='step_2_51'
data_holdout=0.2
client_holdout=0.2
corruption="ood"

model='resnet18'
algorithm='feddc'

batch_size=20

partition_seed=0

tests=('batch' 'online')  # feddc-batch, feddc-online

for seed in {0..0}; do
  for i in {0,1}; do
    {
      echo ${tests[i]}
      load_model_path="artifacts/cifar100/${model}/checkpoints/pretrain_fedawi_${model}_pseed_${partition_seed}_seed_${seed}.pkl"
      load_adapt_path="artifacts/cifar100/${model}/history/feddc_${model}_pseed_${partition_seed}_seed_${seed}.pkl"
      load_adapt_idx=0
      load_adapt_round=-1

      history_path="artifacts/cifar100/${model}/history/feddc_test_${tests[i]}_pseed_${partition_seed}_seed_${seed}.pkl"

      CUDA_VISIBLE_DEVICES=${gpu} python main.py \
        --dataset ${dataset} \
        --num_clients ${num_clients} \
        --partition ${partition} \
        --data_holdout ${data_holdout} \
        --client_holdout ${client_holdout} \
        --partition_seed ${partition_seed} \
        --corruption ${corruption}\
        --model ${model} \
        --algorithm ${algorithm} \
        --test ${tests[i]} \
        --load_adapt_path ${load_adapt_path} \
        --load_adapt_idx ${load_adapt_idx} \
        --load_adapt_round ${load_adapt_round} \
        --batch_size ${batch_size} \
        --seed ${seed} \
        --cuda \
        --history_path ${history_path} \
        --load_model_path ${load_model_path}

    }
  done
done
