BASE_INSTANCE_DIR="data"
BASE_PRIOR_DIR="created_prior"
OUTPUT_DIR_PREFIX="weights/style_"

GRADIENT_ACCUMULATION_STEPS=1
MAX_NUM=28

INSTANCE_DIR="${BASE_INSTANCE_DIR}/$(printf "%02d" $TASK_ID)/images"
PRIOR_DIR="${BASE_PRIOR_DIR}/$(printf "%02d" $TASK_ID)"
PROMPTS_DIR="${BASE_INSTANCE_DIR}/$(printf "%02d" $TASK_ID)"
OUTPUT_DIR="${OUTPUT_DIR_PREFIX}$(printf "%02d" $TASK_ID)/${WEIGHT_ID}"

COMMAND="CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES python train.py
    --prompt_type=$PROMPT_TYPE \
    --prior_weight=$PRIOR_WEIGHT \
    --text_encoder_rank=$TEXT_ENCODER_RANK \
    --instance_data_dir=$INSTANCE_DIR \
    --prior_data_dir=$PRIOR_DIR \
    --prompts_data_dir=$PROMPTS_DIR \
    --rank=$LORA_RANK \
    --train_batch_size=$BATCH_SIZE \
    --learning_rate=${BATCH_SIZE}e-4 \
    --num_train_epochs=$NUM_EPOCHS \
    --output_dir=$OUTPUT_DIR"

eval $COMMAND
