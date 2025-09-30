import numpy as np
import cv2
import matplotlib.pyplot as plt
from IPython.display import HTML
from base64 import b64encode 

"""Display Whole Video"""
def display_video(video_path: str, width: int = 500) -> HTML:
    """
    Display the whole video inside Jupyter Notebook.

    Parameters
    ----------
    video_path : str
        Path to the video file (e.g., "videos/sample.mp4").
    width : int, optional, default=500
        Display width of the video player in pixels.

    Returns
    -------
    HTML
        An HTML video player object for Jupyter Notebook.
    """
    with open(video_path, mode='rb') as f:
        video_url = f.read()
    data_url = "data:video/mp4;base64," + b64encode(video_url).decode()
    return HTML(f"""<video width={width} controls>
                       <source src="{data_url}" type="video/mp4">
                    </video>""")

"""Display Specific Frame in Video"""
def display_img_from_video(video_path: str, frame_number: int) -> None:
    """
    Display a specific frame from the given video.

    Parameters
    ----------
    video_path : str
        Path to the video file (e.g., "videos/sample.mp4").
    frame_number : int
        The index of the frame to display (0-based index).

    Returns
    -------
    None
        Displays the selected frame using matplotlib.
    """
  
    capture_image = cv2.VideoCapture(video_path)
    if not capture_image.isOpened():
        raise FileNotFoundError(f"Could not open video file: {video_path}")

    total_frames = int(capture_image.get(cv2.CAP_PROP_FRAME_COUNT))
    if frame_number < 1 or frame_number > total_frames:
        capture_image.release()
        raise ValueError(
            f"Frame number {frame_number} is out of range. "
            f"Valid range is 1 to {total_frames}."
        )

    capture_image.set(cv2.CAP_PROP_POS_FRAMES, frame_number - 1)
    ret, frame = capture_image.read()
    capture_image.release()

    if not ret or frame is None:
        raise RuntimeError(f"Failed to read frame {frame_number} from {video_path}")

    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    plt.figure(figsize=(12, 6))
    plt.title(f"Frame Number: {frame_number} / {total_frames}", size=15, fontweight='bold')
    plt.imshow(frame)
    plt.axis('off')
    plt.show()