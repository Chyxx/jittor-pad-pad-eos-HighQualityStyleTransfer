import jittor as jt
jt.misc.set_global_seed(0, different_seed_for_mpi=False)
import jsonlines
import json, os, tqdm
from argparse import ArgumentParser
from utils import Img2ImgPipeline
from utils import get_avg_image

os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

parser = ArgumentParser()
parser.add_argument("--id", type=int, required=True)
parser.add_argument("--weight_id", type=int, required=True)
args = parser.parse_args()

dataset_root = "data"

with jsonlines.open("configs/infer_config.jsonl", 'r') as reader:
    tasks = [video for video in reader]

config = tasks[args.id]["config"][args.weight_id]


with jt.no_grad():
    taskid = "{:0>2d}".format(args.id)
    # load json
    with open(f"data/{taskid}/prompt.json", "r") as file:
        prompts = json.load(file)

    avg = get_avg_image(f"data/{taskid}/images")
    images = [avg for i in range(25)]

    pipe = Img2ImgPipeline.from_pretrained("stabilityai/stable-diffusion-2-1").to(f"cuda")
    pipe.load_lora_weights(f"weights/style_{taskid}/{args.weight_id}")
    if config["prompt_type"] == "both":
        style_prompt = [f"A {prompt} in elfkj" for prompt in prompts.values()]
    elif config["prompt_type"] == "only_object":
        style_prompt = [f"A {prompt}" for prompt in prompts.values()]
    else:
        raise NotImplementedError
    
    origin_prompt = [prompt for prompt in prompts.values()]


    images = pipe(
        style_prompt=style_prompt, 
        origin_prompt=origin_prompt, 
        image=images, 
        origin_scale=config["origin_scale"], 
        strength=config["strength"], 
        num_inference_steps=40, 
        negative_prompt=[config["negative_prompt"] for i in range(25)] if config["negative_prompt"] else None
    ).images

    os.makedirs(f"result/{taskid}", exist_ok=True)

    for id, prompt in prompts.items():
        if prompt in config["prompts"]:
            images[int(id)].save(f"result/{taskid}/{prompt}.png")