import os
import subprocess
import jsonlines

with jsonlines.open("configs/infer_config.jsonl", 'r') as reader:
    tasks = [task for task in reader]

for task in tasks:
    for config in task["config"]:
        subprocess.run(
            ["python", "run_one.py" , f"--id={task['style']}", f"--weight_id={config['weight_id']}"],
            env={**os.environ, "CUDA_VISIBLE_DEVICES": "0"}
        )