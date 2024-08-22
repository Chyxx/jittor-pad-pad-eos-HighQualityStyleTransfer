import os
import subprocess
import jsonlines

with jsonlines.open("configs/train_config.jsonl", 'r') as reader:
    tasks = [task for task in reader]

for task in tasks:
    for config in task["config"]:
        subprocess.run(
            ["bash", "train.sh"],
            env={
                **os.environ, 
                "CUDA_VISIBLE_DEVICES": "0", 
                "PROMPT_TYPE": f"{config['prompt_type']}", 
                "PRIOR_WEIGHT": f"{config['prior_weight']}", 
                "LORA_RANK": f"{config['rank']}", 
                "BATCH_SIZE": f"{config['batch_size']}", 
                "NUM_EPOCHS": f"{config['num_epochs']}", 
                "TEXT_ENCODER_RANK": f"{config['text_encoder_rank']}",
                "TASK_ID": f"{task['style']}",
                "WEIGHT_ID": f"{config['weight_id']}"
            }
        )