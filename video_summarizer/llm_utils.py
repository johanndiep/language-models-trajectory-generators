import base64
import os
from dotenv import load_dotenv
from mistralai import Mistral


class ImageAnalyzer:
    def __init__(self, frames_directory: str, model: str, api_key: str):
        self.frames_directory = frames_directory
        self.model = model
        self.client = Mistral(api_key=api_key)

    def encode_image(self, image_path: str) -> str:
        try:
            with open(image_path, "rb") as image_file:
                return base64.b64encode(image_file.read()).decode("utf-8")

        except Exception as e:
            print(f"Error encoding image {image_path}: {e}")
            return None

    def collect_images(self):
        image_messages = []
        for filename in os.listdir(self.frames_directory):
            if filename.endswith(".jpg"):
                image_path = os.path.join(self.frames_directory, filename)
                base64_image = self.encode_image(image_path)
                if base64_image:
                    image_messages.append(
                        {
                            "type": "image_url",
                            "image_url": f"data:image/jpeg;base64,{base64_image}",
                        }
                    )
        return image_messages

    def collect_first_image(self):
        jpg_files = sorted(
            [f for f in os.listdir(self.frames_directory) if f.endswith(".jpg")]
        )

        if jpg_files:
            image_path = os.path.join(self.frames_directory, jpg_files[0])
            base64_image = self.encode_image(image_path)
            if base64_image:
                return [
                    {
                        "type": "image_url",
                        "image_url": f"data:image/jpeg;base64,{base64_image}",
                    }
                ]
        return None

    def caption_first_frame(self):
        image_messages = self.collect_first_image()

        content = {
            "type": "text",
            "text": "Describe what you can see in this image in one sentence.",
        }

        messages = [{"role": "user", "content": [content] + image_messages}]

        chat_response = self.client.chat.complete(model=self.model, messages=messages)
        return chat_response.choices[0].message.content

    def caption_difference(self, previous_difference, sequential_image_messages):

        content = {
            "type": "text",
            "text": f"This happened previously:\nprevious_difference\nDescribe the task happening between these two frames in one sentence.",
        }

        messages = [
            {
                "role": "user",
                "content": [content] + sequential_image_messages,
            }
        ]

        chat_response = self.client.chat.complete(model=self.model, messages=messages)
        return chat_response.choices[0].message.content


if __name__ == "__main__":
    load_dotenv()
    api_key = os.environ["MISTRAL_API_KEY"]

    frames_directory = "frames"
    model = "pixtral-12b-2409"

    caption_list = []

    analyzer = ImageAnalyzer(frames_directory, model, api_key)
    grounding = analyzer.caption_first_frame()
    caption_list.append(grounding)

    frame_list = analyzer.collect_images()

    for i in range(len(frame_list) - 1):
        if i == 0:
            difference = grounding

        consecutive_frames = frame_list[i : i + 2]

        if len(consecutive_frames) == 2:
            difference = analyzer.caption_difference(difference, consecutive_frames)
            caption_list.append(difference)
