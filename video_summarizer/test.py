import cv2
import os


class VideoProcessor:
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


if __name__ == "__main__":
    video_path = "video.mp4"

    video_processor = VideoProcessor(video_path)

    key_frames = video_processor.extract_key_frames(interval=1)
