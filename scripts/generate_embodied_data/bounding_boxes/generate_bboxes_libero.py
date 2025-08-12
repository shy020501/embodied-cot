import argparse
import json
import os
import time
import warnings

import tensorflow_datasets as tfds
import torch
from PIL import Image
from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor
from utils import NumpyFloatValuesEncoder, post_process_caption

warnings.filterwarnings("ignore")

parser = argparse.ArgumentParser()

parser.add_argument("--id", default=5, type=int)
parser.add_argument("--gpu", default=0, type=int)
parser.add_argument("--splits", default=24, type=int)
parser.add_argument("--result-path", default="./bboxes")
parser.add_argument("--data-path", default="/home/work/AGI_NIH/data/embodied_features_and_demos_libero/libero_90/1.0.0")  # TFDS path


args = parser.parse_args()

bbox_json_path = os.path.join(args.result_path, f"results_{args.id}_bboxes.json")

print("📦 Loading data...")
split_percents = 100 // args.splits
start = args.id * split_percents
end = (args.id + 1) * split_percents

builder = tfds.builder_from_directory(args.data_path)
ds = builder.as_dataset(split=f"train[{start}%:{end}%]")
print("✅ Dataset loaded.")

print("📝 Loading Prismatic descriptions...")
results_json_path = "./descriptions/full_descriptions.json"
with open(results_json_path, "r") as f:
    results_json = json.load(f)
print("✅ Captions loaded.")

print(f"🚀 Loading Grounding DINO to device cuda:{args.gpu}...")
model_id = "IDEA-Research/grounding-dino-base"
device = f"cuda:{args.gpu}"
processor = AutoProcessor.from_pretrained(model_id, size={"shortest_edge": 256, "longest_edge": 256})
model = AutoModelForZeroShotObjectDetection.from_pretrained(model_id).to(device)
print("✅ gDINO ready.")

BOX_THRESHOLD = 0.3
TEXT_THRESHOLD = 0.2
bbox_results_json = {}

for ep_idx, episode in enumerate(ds):
    demo_id = episode["episode_metadata"]["demo_id"].numpy()
    file_path = episode["episode_metadata"]["file_path"].numpy().decode()
    print(f"📂 ID {args.id} processing demo_id={demo_id}, file={file_path}")

    if file_path not in bbox_results_json:
        bbox_results_json[file_path] = {}

    episode_json = results_json[file_path][str(demo_id)]
    description = episode_json["caption"]

    start_t = time.time()
    bboxes_list = []

    for step_idx, step in enumerate(episode["steps"]):
        if step_idx == 0:
            lang_instruction = step["language_instruction"].numpy().decode()

        image = Image.fromarray(step["observation"]["image"].numpy())  # LIBERO image key
        inputs = processor(
            images=image,
            text=post_process_caption(description, lang_instruction),
            return_tensors="pt",
        ).to(device)

        with torch.no_grad():
            outputs = model(**inputs)

        results = processor.post_process_grounded_object_detection(
            outputs,
            inputs.input_ids,
            box_threshold=BOX_THRESHOLD,
            text_threshold=TEXT_THRESHOLD,
            target_sizes=[image.size[::-1]],
        )[0]

        logits, phrases, boxes = (
            results["scores"].cpu().numpy(),
            results["labels"],
            results["boxes"].cpu().numpy(),
        )

        bboxes = []
        for lg, p, b in zip(logits, phrases, boxes):
            b = list(b.astype(int))
            lg = round(lg, 5)
            bboxes.append((lg, p, b))
            break  # 하나만 사용

        bboxes_list.append(bboxes)

    end_t = time.time()

    bbox_results_json[file_path][str(demo_id)] = {
        "demo_id": int(demo_id),
        "file_path": file_path,
        "bboxes": bboxes_list,
    }

    with open(bbox_json_path, "w") as f:
        json.dump(bbox_results_json, f, cls=NumpyFloatValuesEncoder)

    print(f"✅ Finished demo_id={demo_id} in {round(end_t - start_t, 2)}s")

