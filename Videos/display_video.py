import os
import argparse
import cv2

# -------------------------------
# Display whole video
# -------------------------------

## cv2.CAP_PROP_FPS: Video Frame per Second
## cv2.CAP_PROP_FRAME_COUNT: Video Total Frames
## cv2.CAP_PROP_FRAME_HEIGHT: Frame Height
## cv2.CAP_PROP_FRAME_WIDTH: Frame Width

def display_video(video_path: str, resize_width: int = None) -> None:
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise FileNotFoundError(f"Could not open video file: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    delay = int(1000 / fps) if fps > 0 else 30
    
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            
    print(f"🎬 Playing video: {video_path}")
    print(f"📸 FPS: {fps:.2f}, Total_Frames: {total_frames}, Delay: {delay}ms")
    print(f"📸 Frame Width: {w}, Frame Height: {h}")
    print("Press 'q' to stop playback.")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("✅ Video playback finished.")
            break

        if resize_width is not None:
            new_h = int(resize_width * h / w)
            frame = cv2.resize(frame, (resize_width, new_h))

        cv2.imshow("Video Playback", frame)
        
        if cv2.waitKey(delay) & 0xFF == ord('q'):
            print("🛑 Video stopped by user.")
            break

    cap.release()
    cv2.destroyAllWindows()

# -------------------------------
# Main
# -------------------------------
def main():
    parser = argparse.ArgumentParser(description="Play a video or display a specific frame")
    parser.add_argument("--video_path", type=str, required=True, help="Path to video file")
    parser.add_argument("--width", type=int, default=None, help="Resize width (optional, used only in 'video' mode')")
    args = parser.parse_args()
    
    print(args)

    if not os.path.exists(args.video_path):
        raise FileNotFoundError(f"Video file not found: {args.video_path}")

    display_video(args.video_path, args.width)

if __name__ == "__main__":
    main()
