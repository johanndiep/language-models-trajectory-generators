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

    def analyze_images(self):
        image_messages = self.collect_images()

        content = {
            "type": "text",
            "text": "In the following sequential video frames, you see a human doing a task. Describe the task the human is doing as detailed as possible in one sentence.",
        }

        messages = [{"role": "user", "content": [content] + image_messages}]

        chat_response = self.client.chat.complete(model=self.model, messages=messages)
        return chat_response.choices[0].message.content


if __name__ == "__main__":
    load_dotenv()
    api_key = os.environ["MISTRAL_API_KEY"]

    frames_directory = "frames"
    model = "pixtral-12b-2409"

    analyzer = ImageAnalyzer(frames_directory, model, api_key)
    response = analyzer.analyze_images()

    print(response)
