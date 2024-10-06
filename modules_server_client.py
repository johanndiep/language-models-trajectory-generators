import numpy as np
import matplotlib.pyplot as plt
import sys
import torch
import config
from openai import OpenAI
from PIL import Image
from torchvision import transforms
from torchvision.utils import draw_bounding_boxes, draw_segmentation_masks

sys.path.append("./XMem/")

from XMem.inference.inference_core import InferenceCore
from XMem.inference.interact.interactive_utils import image_to_torch, index_numpy_to_one_hot_torch, torch_prob_to_numpy_mask, overlay_davis

import requests
import pickle
import io

def get_langsam_output(image_path, model, segmentation_texts, segmentation_count):
    # Serialize the model
    model_bytes = pickle.dumps(model)
    # printing debug info
    print("entered get_langsam_output_client")
    # API endpoint
    server_url = "http://195.242.23.14:8000/process_image/"

    # Open the image file and send it to the server along with the serialized model
    with open(image_path, "rb") as image_file:
        files = {
            "file": image_file,
            "model_file": io.BytesIO(model_bytes),  # Send the model as a file-like object. this will take some time and might even hang up your computer. Do not worry.
        }
        data = {
            "segmentation_texts": segmentation_texts,
            "segmentation_count": segmentation_count,
        }
        response = requests.post(server_url, files=files, data=data)

    # Parse the server's response
    response_data = response.json()
    masks = torch.tensor(response_data["masks"])
    boxes = [torch.tensor(box) for box in response_data["boxes"]]
    phrases = response_data["phrases"]

    return masks, boxes, phrases

# # Example usage with a placeholder model
# # Replace with your actual model instance
# model = torch.hub.load('pytorch/vision', 'fasterrcnn_resnet50_fpn', pretrained=True)
# model.eval()

# # Use the updated function
# masks, boxes, phrases = get_langsam_output_client("your_image_path.jpg", model, ["example text"], 1)


def get_chatgpt_output(model, new_prompt, messages, role, file=sys.stdout):

    print(role + ":", file=file)
    print(new_prompt, file=file)
    messages.append({"role":role, "content":new_prompt})

    client = OpenAI()

    completion = client.chat.completions.create(
        model=model,
        temperature=0,
        messages=messages,
        stream=True
    )

    print("assistant:", file=file)

    new_output = ""

    for chunk in completion:
        chunk_content = chunk.choices[0].delta.content
        finish_reason = chunk.choices[0].finish_reason
        if chunk_content is not None:
            print(chunk_content, end="", file=file)
            new_output += chunk_content
        else:
            print("finish_reason:", finish_reason, file=file)

    messages.append({"role":"assistant", "content":new_output})

    return messages



def get_xmem_output(model, device, trajectory_length):

    mask = np.array(Image.open(config.xmem_input_path).convert("L"))
    mask = np.unique(mask, return_inverse=True)[1].reshape(mask.shape)
    num_objects = len(np.unique(mask)) - 1

    torch.cuda.empty_cache()

    processor = InferenceCore(model, config.xmem_config)
    processor.set_all_labels(range(1, num_objects + 1))

    masks = []

    with torch.cuda.amp.autocast(enabled=True):

        for i in range(0, trajectory_length + 1, config.xmem_output_every):

            frame = np.array(Image.open(config.rgb_image_trajectory_path.format(step=i)).convert("RGB"))

            frame_torch, _ = image_to_torch(frame, device)
            if i == 0:
                mask_torch = index_numpy_to_one_hot_torch(mask, num_objects + 1).to(device)
                prediction = processor.step(frame_torch, mask_torch[1:])
            else:
                prediction = processor.step(frame_torch)

            prediction = torch_prob_to_numpy_mask(prediction)
            masks.append(prediction)

            if i % config.xmem_visualise_every == 0:
                visualisation = overlay_davis(frame, prediction)
                output = Image.fromarray(visualisation)
                output.save(config.xmem_output_path.format(step=i))

    return masks
