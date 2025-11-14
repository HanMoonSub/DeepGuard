import argparse
import cv2
import numpy as np
import os
import glob
import random
import time
import sys

try:
    from mtcnn.mtcnn import MTCNN
except ImportError:
    print("="*50)
    print("!!! 오류: 'mtcnn' 라이브러리를 찾을 수 없습니다.")
    print("!!! 터미널에 [ pip install -r requirements.txt ]를 실행해 주세요.")
    print("="*50)
    sys.exit(1)



# --- 함수 : 믹스업 (MixUp) ---
def mixup_augmentation(image_a, image_b, alpha=1.0):
    lambda_val = np.random.beta(alpha, alpha) 
    print(f"  > MixUp 적용 (A: {lambda_val*100:.1f}%, B: {(1-lambda_val)*100:.1f}%)")
    mixed_image = (lambda_val * image_a.astype(float) +
                   (1 - lambda_val) * image_b.astype(float))
    return mixed_image.astype(np.uint8)

# --- 함수 : MTCNN 얼굴 추출 ---
def extract_face_from_video(face_detector, video_path,
                               frame_num=30, target_size=(160, 160)):
    cap = cv2.VideoCapture(video_path)
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
    ret, frame = cap.read()
    cap.release()
    if not ret:
        print(f"  [실패] 프레임 읽기 실패: {os.path.basename(video_path)}")
        return None
    frame_rgb_mtcnn = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    faces = face_detector.detect_faces(frame_rgb_mtcnn)
    if len(faces) == 0:
        print(f"  [실패] MTCNN 얼굴 찾기 실패: {os.path.basename(video_path)}")
        return None
    (x, y, w, h) = faces[0]['box']
    x, y = abs(x), abs(y)
    face_crop = frame[y:y+h, x:x+w]
    if face_crop.size == 0:
        print(f"  [실패] 얼굴 영역 자르기 실패 (크기 0): {os.path.basename(video_path)}")
        return None
    print(f"  [성공] 얼굴 추출 완료: {os.path.basename(video_path)}")
    return cv2.resize(face_crop, target_size)


# --- main 함수 ---
def main(args):
    video_dir = args.root_dir 

    if args.detector_type == "MTCNN":
        print("딥러닝 얼굴 탐지기(MTCNN) 로드 중...")
        face_detector = MTCNN()
        print("MTCNN 로드 완료.")
    else:
        print(f"오류: {args.detector_type} 탐지기는 이 스크립트에서 지원하지 않습니다. MTCNN을 로드합니다.")
        face_detector = MTCNN()

    if face_detector is None:
        print("오류: MTCNN 분류기를 로드할 수 없습니다.")
        return

    print(f"\n지정된 비디오 경로: {video_dir}")
    try:
        raw_file_list = os.listdir(video_dir)
        print(f"os.listdir로 찾은 파일/폴더 (상위 10개): {raw_file_list[:10]}")
    except FileNotFoundError:
        print(f"!!! 오류: '{video_dir}' 경로를 찾을 수 없습니다!")
        print("!!! --root_dir 인자가 올바른지 확인하세요.")
        return
    except Exception as e:
        print(f"os.listdir 오류: {e}")
        return

    video_files = glob.glob(os.path.join(video_dir, '*.mp4'))
    print(f"경로에서 찾은 .mp4 파일 개수: {len(video_files)} 개")

    if len(video_files) < 2:
        print(f"!!! 오류: MixUp 시연을 위해 '{video_dir}' 폴더에 2개 이상의 비디오 파일이 필요합니다.")
        return

    print("MixUp 시연을 위해 2개의 랜덤 비디오에서 '실제 얼굴'을 추출")
    try:
        video_path_A, video_path_B = random.sample(video_files, 2)
        print(f"  > 랜덤 비디오 A: {os.path.basename(video_path_A)}")
        print(f"  > 랜덤 비디오 B: {os.path.basename(video_path_B)}")
    except ValueError:
        video_path_A, video_path_B = video_files[0], video_files[1]

    face_A = extract_face_from_video(
        face_detector, video_path_A, frame_num=30
    )
    face_B = extract_face_from_video(
        face_detector, video_path_B, frame_num=60
    )

    if face_A is not None and face_B is not None:
        
        print("\n--- 결과 이미지를 새 창으로 띄웁니다 ---")
        print("--- 창을 클릭하고 키보드의 아무 키나 누르면 다음으로 넘어갑니다 ---")

        print("\n--- 1. 원본 얼굴 A ---")
        cv2.imshow("1. Original Face A", face_A)
        cv2.waitKey(0)

        print("\n--- 2. 원본 얼굴 B ---")
        cv2.imshow("2. Original Face B", face_B)
        cv2.waitKey(0)

        if random.random() < args.mixup_prob:
            print("\n--- 3. [Final] (얼굴 A)와 (얼굴 B)를 [MixUp]한 결과 ---")
            mixed_face = mixup_augmentation(face_A, face_B, alpha=args.mixup_alpha)
            cv2.imshow("3. MixUp Result", mixed_face)
            cv2.waitKey(0)
        else:
            print(f"\n--- 3. MixUp이 확률(P={args.mixup_prob})에 의해 건너뛰었습니다. ---")

        cv2.destroyAllWindows()
        
    else:
        print("\n얼굴 추출에 실패하여 증강 시연을 중단합니다.")

    print("\n--- 모든 작업 완료 ---")


# --- 스크립트 실행 지점 ---
if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(
        description="Process original videos with face detector"
    )
    parser.add_argument("--root_dir", 
                        help='root directory', 
                        default='DFDC_Folder/test_videos/') # ★ 로컬 기본 경로 (수정 필요)
    parser.add_argument("--detector_type", 
                        help='type of the detector', 
                        default="MTCNN",
                        choices=["FacenetDetector", "MTCNN"])
    parser.add_argument("--batch_size", 
                        default=32, 
                        type=int, 
                        help="Face Detector Batch Size")
    parser.add_argument("--apply_clahe", 
                        default=False, 
                        type=bool, 
                        help="Apply clahe for find calibrated bbox coordinates")
    parser.add_argument("--mixup_prob", 
                        default=1.0,
                        type=float, 
                        help="Probability of applying MixUp")
    parser.add_argument("--mixup_alpha", 
                        default=1.0,
                        type=float, 
                        help="Alpha value for Beta distribution in MixUp")
    
    args = parser.parse_args() 
    
    print("\n--- argparse 설정 로드 완료 ---")
    print(f"  > Root Dir: {args.root_dir}")
    print(f"  > Detector: {args.detector_type}")
    print(f"  > Batch Size: {args.batch_size} ")
    print(f"  > MixUp Probability (P): {args.mixup_prob}")
    print(f"  > MixUp Alpha: {args.mixup_alpha}")
    
    main(args)