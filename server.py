from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse
from typing import List
import torch
import pickle
from PIL import Image
import io
import matplotlib.pyplot as plt
from torchvision.utils import draw_bounding_boxes, draw_segmentation_masks
import torchvision.transforms as transforms

app = FastAPI()

class Config:
    langsam_image_path = "output_segmentation_{object}.png"

config = Config()

def get_langsam_output(image, model, segmentation_texts, segmentation_count):
    segmentation_texts = " . ".join(segmentation_texts)
    masks, boxes, phrases, logits = model.predict(image, segmentation_texts)

    _, ax = plt.subplots(1, 1 + len(masks), figsize=(5 + (5 * len(masks)), 5))
    [a.axis("off") for a in ax.flatten()]
    ax[0].imshow(image)

    for i, (mask, box, phrase) in enumerate(zip(masks, boxes, phrases)):
        to_tensor = transforms.PILToTensor()
        image_tensor = to_tensor(image)
        box = box.unsqueeze(dim=0)
        image_tensor = draw_bounding_boxes(image_tensor, box, colors=["red"], width=3)
        image_tensor = draw_segmentation_masks(image_tensor, mask, alpha=0.5, colors=["cyan"])
        to_pil_image = transforms.ToPILImage()
        image_pil = to_pil_image(image_tensor)

        ax[1 + i].imshow(image_pil)
        ax[1 + i].text(box[0][0], box[0][1] - 15, phrase, color="red", bbox={"facecolor":"white", "edgecolor":"red", "boxstyle":"square"})

    plt.savefig(config.langsam_image_path.format(object=segmentation_count))
    plt.show()

    masks = masks.float()

    return masks, boxes, phrases

@app.post("/process_image/")
async def process_image(file: UploadFile = File(...), model_file: UploadFile = File(...), segmentation_texts: List[str] = [], segmentation_count: int = 0):
    # Load the uploaded model file
    model_bytes = await model_file.read()
    model = pickle.loads(model_bytes)

    # Read the uploaded image file
    image_bytes = await file.read()
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")

    # Run the processing function
    masks, boxes, phrases = get_langsam_output(image, model, segmentation_texts, segmentation_count)

    # Convert tensors to list format for JSON serialization
    masks = masks.tolist()
    boxes = [b.tolist() for b in boxes]
    phrases = list(phrases)

    return JSONResponse({"masks": masks, "boxes": boxes, "phrases": phrases})


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
