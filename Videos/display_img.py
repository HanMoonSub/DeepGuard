import os
import argparse
import cv2

# -------------------------------
# Display specific frame
# -------------------------------
def display_img(video_path: str, frame_number: int) -> None:
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise FileNotFoundError(f"Could not open video file: {video_path}")

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if frame_number < 1 or frame_number > total_frames:
        cap.release()
        raise ValueError(f"Frame number {frame_number} is out of range (1-{total_frames})")

    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number - 1)
    ret, frame = cap.read()
    cap.release()
    if not ret or frame is None:
        raise RuntimeError(f"Failed to read frame {frame_number} from {video_path}")

    cv2.imshow(f"Frame {frame_number}/{total_frames}", frame)
    print("Press any key to close window...")
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# -------------------------------
# Main
# -------------------------------
def main():
    parser = argparse.ArgumentParser(description="Play a video or display a specific frame")
    parser.add_argument("--video_path", type=str, required=True, help="Path to video file")
    parser.add_argument("--frame", type=int, default=1, help="Frame number in Video")

    args = parser.parse_args()
    print(args)

    if not os.path.exists(args.video_path):
        raise FileNotFoundError(f"Video file not found: {args.video_path}")

    display_img(args.video_path, args.frame)

if __name__ == "__main__":
    main()
