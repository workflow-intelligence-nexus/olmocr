#!/usr/bin/env bash

set -ex

# check if jq is installed
if ! command -v jq &> /dev/null
then
    echo "jq could not be found. Please install it."
    exit
fi


EXTRA_ARGS="-c olmocr/train/config/qwen25vl-7b.yaml --num_proc 64 --save.path \"s3://ai2-oe-data/jakep/experiments/qwen25vl-pdf/v1/models/\${BEAKER_USER_ID}\""

run_name=$(basename "$0" .sh)

# --cluster 'ai2/jupiter*' \
# --cluster 'ai2/pluto*' \
# --cluster 'ai2/allennlp-cirrascale' \
# --priority high \

CLUSTER='jupiter'

gantry run \
    --description "${run_name}"\
    --task-name "${run_name}"\
    --allow-dirty \
    --host-networking \
    --workspace ai2/oe-data-model-based-cleanup \
    --beaker-image 'jakep/jakep-pdf-finetunev1.2' \
    --venv 'base' \
    --pip gantry-requirements.txt \
    --priority high \
    --gpus 8 \
    --preemptible \
    --cluster "ai2/${CLUSTER}*" \
    --budget ai2/oe-data \
    --weka "oe-data-default:/data" \
    --env LOG_FILTER_TYPE=local_rank0_only \
    --env OMP_NUM_THREADS=8 \
    --env BEAKER_USER_ID=$(beaker account whoami --format json | jq '.[0].name' -cr) \
    --env-secret AWS_ACCESS_KEY_ID=S2_AWS_ACCESS_KEY_ID \
    --env-secret AWS_SECRET_ACCESS_KEY=S2_AWS_SECRET_ACCESS_KEY \
    --env-secret WANDB_API_KEY=JAKE_WANDB_API_KEY \
    --shared-memory 10GiB \
    --yes \
    -- /bin/bash -c "source scripts/beaker/${CLUSTER}-ib.sh && python -m olmocr.train.loaddataset ${EXTRA_ARGS} && accelerate launch --use_fsdp --num_processes \${BEAKER_ASSIGNED_GPU_COUNT} --fsdp_offload_params false --fsdp_sharding_strategy FULL_SHARD --fsdp_auto_wrap_policy TRANSFORMER_BASED_WRAP --mixed_precision bf16 -m olmocr.train.train ${EXTRA_ARGS}"