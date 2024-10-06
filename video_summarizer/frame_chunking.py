import cv2
import os
from PIL import Image
import torch
from semantic_router.encoders import VitEncoder
from semantic_chunkers import ConsecutiveChunker
import matplotlib.pyplot as plt
from PIL import Image
import torch
from semantic_router.encoders import VitEncoder
from semantic_chunkers import ConsecutiveChunker
import matplotlib.pyplot as plt

class FrameChunker:
    """
    A class to handle video processing tasks such as extracting key frames
    at specified intervals.
    """

    def __init__(self, video_path):
        """
        Initialize the VideoProcessor with a video file.

        Args:
            video_path (str): Path to the video file.
        """
        self.video_path = video_path
        self.cap = cv2.VideoCapture(self.video_path)
        self.fps = self.cap.get(cv2.CAP_PROP_FPS)

        self.frames_dir = os.path.join(os.getcwd(), "frames")
        if not os.path.exists(self.frames_dir):
            os.makedirs(self.frames_dir)

    def extract_key_frames(self, interval=1):
        """
        Extract key frames from the video at the given time interval. The time interval is multiplied
        with the framerate to define the keyframes.

        Args:
            interval (int, optional): Interval in seconds between frames to extract. Defaults to 1.

        Returns:
            list: A list of file paths for the extracted key frames.
        """
        frames = []
        frame_interval = int(self.fps * interval)
        frame_count = 0

        while self.cap.isOpened():
            ret, frame = self.cap.read()

            if not ret:
                break

            if frame_count % frame_interval == 0:
                frame_file = os.path.join(self.frames_dir, f"frame_{frame_count}.jpg")
                cv2.imwrite(frame_file, frame)
                frames.append(frame_file)

            frame_count += 1

        self.cap.release()

        return frames

    def semantic_segmentation_VitEncoder(self):
        """
        Perform semantic segmentation on the video frames to identify objects and their
        locations in the video.
        """
        vidcap = cv2.VideoCapture(self.video_path)

        frames = []
        success, image = vidcap.read()
        while success:
            frames.append(image)
            success, image = vidcap.read()
        length = len(frames)


        image_frames = list(map(Image.fromarray, frames))
        
        device = (
            "mps"
            if torch.backends.mps.is_available()
            else "cuda" if torch.cuda.is_available() else "cpu"
        )


        encoder = VitEncoder(device=device)

        chunker = ConsecutiveChunker(encoder=encoder, score_threshold=0.9)

        chunks = chunker(docs=[image_frames])

        
        # Ensure the frames directory exists
        frames_dir = "frames"
        os.makedirs(frames_dir, exist_ok=True)

        for i, chunk in enumerate(chunks[0]):
            # Save the first split
            first_split_path = os.path.join(frames_dir, f"chunk_{i}_1.png")
            chunk.splits[0].save(first_split_path)

            # Calculate the middle index
            num_docs = len(chunk.splits)
            mid = num_docs // 2

            # Save the middle split
            mid_split_path = os.path.join(frames_dir, f"chunk_{i}_2.png")
            chunk.splits[mid].save(mid_split_path)

            # Save the last split
            last_split_path = os.path.join(frames_dir, f"chunk_{i}_3.png")
            chunk.splits[num_docs - 1].save(last_split_path)
        return chunks
    
    def consecutive_chunker(self):
        """
        Perform semantic segmentation on the video frames to identify objects and their
        locations in the video.
        """
        vidcap = cv2.VideoCapture(self.video_path)

        frames = []
        success, image = vidcap.read()
        while success:
            frames.append(image)
            success, image = vidcap.read()
        length = len(frames)


        image_frames = list(map(Image.fromarray, frames))

        # Initialize the encoder
        device = (
            "mps"
            if torch.backends.mps.is_available()
            else "cuda" if torch.cuda.is_available() else "cpu"
        )


        encoder = VitEncoder(device=device)

        chunker = ConsecutiveChunker(encoder=encoder, score_threshold=0.912)

        chunks = chunker(docs=[image_frames])



        # saving chunks as frames in frames folder
        # Ensure the frames directory exists
        frames_dir = "frames"
        os.makedirs(frames_dir, exist_ok=True)
        
        for i, chunk in enumerate(chunks[0]):
            # Save the first split
            first_split_path = os.path.join(frames_dir, f"chunk_{i}_1.png")
            chunk.splits[0].save(first_split_path)
        
            # Calculate the middle index
            num_docs = len(chunk.splits)
            mid = num_docs // 2
        
            # Save the middle split
            mid_split_path = os.path.join(frames_dir, f"chunk_{i}_2.png")
            chunk.splits[mid].save(mid_split_path)
        
            # Save the last split
            last_split_path = os.path.join(frames_dir, f"chunk_{i}_3.png")
            chunk.splits[num_docs - 1].save(last_split_path)

        return chunks

    def semantic_segmentation_VitEncoder(self):
        """
        Perform semantic segmentation on the video frames to identify objects and their
        locations in the video.
        """
        vidcap = cv2.VideoCapture(self.video_path)

        frames = []
        success, image = vidcap.read()
        while success:
            frames.append(image)
            success, image = vidcap.read()
        length = len(frames)


        image_frames = list(map(Image.fromarray, frames))
        
        device = (
            "mps"
            if torch.backends.mps.is_available()
            else "cuda" if torch.cuda.is_available() else "cpu"
        )


        encoder = VitEncoder(device=device)

        chunker = ConsecutiveChunker(encoder=encoder, score_threshold=0.9)

        chunks = chunker(docs=[image_frames])

        
        # Ensure the frames directory exists
        frames_dir = "frames"
        os.makedirs(frames_dir, exist_ok=True)

        for i, chunk in enumerate(chunks[0]):
            # Save the first split
            first_split_path = os.path.join(frames_dir, f"chunk_{i}_1.png")
            chunk.splits[0].save(first_split_path)

            # Calculate the middle index
            num_docs = len(chunk.splits)
            mid = num_docs // 2

            # Save the middle split
            mid_split_path = os.path.join(frames_dir, f"chunk_{i}_2.png")
            chunk.splits[mid].save(mid_split_path)

            # Save the last split
            last_split_path = os.path.join(frames_dir, f"chunk_{i}_3.png")
            chunk.splits[num_docs - 1].save(last_split_path)
        return chunks
    
    def consecutive_chunker(self):
        """
        Perform semantic segmentation on the video frames to identify objects and their
        locations in the video.
        """
        vidcap = cv2.VideoCapture(self.video_path)

        frames = []
        success, image = vidcap.read()
        while success:
            frames.append(image)
            success, image = vidcap.read()
        length = len(frames)


        image_frames = list(map(Image.fromarray, frames))

        # Initialize the encoder
        device = (
            "mps"
            if torch.backends.mps.is_available()
            else "cuda" if torch.cuda.is_available() else "cpu"
        )


        encoder = VitEncoder(device=device)

        chunker = ConsecutiveChunker(encoder=encoder, score_threshold=0.912)

        chunks = chunker(docs=[image_frames])



        # saving chunks as frames in frames folder
        # Ensure the frames directory exists
        frames_dir = "frames"
        os.makedirs(frames_dir, exist_ok=True)
        
        for i, chunk in enumerate(chunks[0]):
            # Save the first split
            first_split_path = os.path.join(frames_dir, f"chunk_{i}_1.png")
            chunk.splits[0].save(first_split_path)
        
            # Calculate the middle index
            num_docs = len(chunk.splits)
            mid = num_docs // 2
        
            # Save the middle split
            mid_split_path = os.path.join(frames_dir, f"chunk_{i}_2.png")
            chunk.splits[mid].save(mid_split_path)
        
            # Save the last split
            last_split_path = os.path.join(frames_dir, f"chunk_{i}_3.png")
            chunk.splits[num_docs - 1].save(last_split_path)

        return chunks


if __name__ == "__main__":
    video_path = "video.mp4"
    video_path = "video.mp4"

    video_processor = FrameChunker(video_path)

    # key_frames = video_processor.extract_key_frames(interval=1)
    # key_frames = video_processor.semantic_segmentation_VitEncoder()
    key_frames = video_processor.consecutive_chunker()