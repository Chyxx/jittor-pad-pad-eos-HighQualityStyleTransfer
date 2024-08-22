import json, os
import jittor as jt

from JDiffusion.pipelines import StableDiffusionPipeline
from JDiffusion import UniPCMultistepScheduler

os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
dataset_root = "data"

def get_image_filenames(directory):
    img_extensions = {'.png'}
    all_files = os.listdir(directory)
    image_files = [os.path.splitext(file)[0].capitalize().replace("_", "") for file in all_files if os.path.splitext(file)[1].lower() in img_extensions]
    return image_files


with jt.no_grad():
    for tempid in range(28):
        taskid = "{:0>2d}".format(tempid)
        pipe = StableDiffusionPipeline.from_pretrained("stabilityai/stable-diffusion-2-1").to("cuda")
        scheduler = UniPCMultistepScheduler().from_config(pipe.scheduler.config)
        pipe.scheduler = scheduler

        train_prompts = get_image_filenames(f"{dataset_root}/{taskid}/images")
        print(train_prompts)

        for id, prompt in enumerate(train_prompts):
            print(prompt)
            os.makedirs(f"./created_prior/{taskid}/train/{prompt}", exist_ok=True)
            for i in range(5):
                image = pipe(f"A {prompt} in the center", num_inference_steps=25, width=512, height=512).images[0]
                image.save(f"./created_prior/{taskid}/train/{prompt}/{i}.png")
        
        # load json
        with open(f"{dataset_root}/{taskid}/prompt.json", "r") as file:
            test_prompts = json.load(file)
        print(tempid)
        for id, prompt in test_prompts.items():
            print(prompt)
            os.makedirs(f"./created_prior/{taskid}/test/{prompt}", exist_ok=True)
            for i in range(5):  # 为了多样性，每个prompt生成5张图片
                image = pipe(f"A {prompt} in the center", num_inference_steps=25, width=512, height=512).images[0]
                image.save(f"./created_prior/{taskid}/test/{prompt}/{i}.png")

        
