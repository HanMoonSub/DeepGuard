import cv2
import numpy as np
import os
import glob
import random
from mtcnn.mtcnn import MTCNN

# --- 2-1. 핵심 함수 (1): 다이나믹 페이스 컷아웃 ---
def dynamic_face_cutout(image, min_points=3, max_points=7, min_area_ratio=0.1, max_area_ratio=0.5):
    """
    이미지에 불규칙한 "다각형" 형태의 검은색 마스크(컷아웃)를 적용합니다.
    """
  
    h, w, _ = image.shape
    image_area = h * w
    
    for _ in range(10): 
        num_points = np.random.randint(min_points, max_points + 1)
        points = np.random.randint(0, [w, h], size=(num_points, 2))
        
        try:
            hull = cv2.convexHull(points)
        except Exception as e:
            continue 

        hull_area = cv2.contourArea(hull)
        area_ratio = hull_area / image_area
        
        if min_area_ratio < area_ratio < max_area_ratio:
            mask = np.ones(image.shape[:2], dtype=np.uint8) * 255
            cv2.fillPoly(mask, [hull], 0)
            mask_3d = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
            augmented_image = cv2.bitwise_and(image, mask_3d)
            return augmented_image 

    return image

# --- 2-2. 핵심 함수 (2): 믹스업 (MixUp) ---
def mixup_augmentation(image_a, image_b, alpha=1.0):
    """
    두 이미지에 MixUp을 적용합니다. (실제 훈련 시에는 레이블도 섞어야 함)
    """
    lambda_val = np.random.beta(alpha, alpha)
    mixed_image = (lambda_val * image_a.astype(float) + 
                   (1 - lambda_val) * image_b.astype(float))
    return mixed_image.astype(np.uint8)

# --- 3-1. 핵심 함수 (3): MTCNN 얼굴 추출기 ---
def extract_face_from_video(face_detector, video_path, frame_num=30, target_size=(160, 160)):
    """
    비디오 파일 경로를 받아, 특정 프레임에서 MTCNN으로 얼굴을 찾아
    160x160 크기로 잘라낸 이미지를 반환합니다.
    """
    cap = cv2.VideoCapture(video_path)
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
    ret, frame = cap.read()
    cap.release()
    
    if not ret:
        print(f"  [실패] 프레임 읽기 실패: {os.path.basename(video_path)}")
        return None
    
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    faces = face_detector.detect_faces(frame_rgb)
    
    if len(faces) == 0:
        print(f"  [실패] 얼굴 찾기 실패: {os.path.basename(video_path)}")
        return None
    
    (x, y, w, h) = faces[0]['box']
    x, y = abs(x), abs(y) 
    face_crop = frame[y:y+h, x:x+w]
    
    if face_crop.size == 0:
        print(f"  [실패] 얼굴 영역 자르기 실패 (크기 0): {os.path.basename(video_path)}")
        return None

    print(f"  [성공] 얼굴 추출 완료: {os.path.basename(video_path)}")
    return cv2.resize(face_crop, target_size)

# --- 3-2. 메인 실행부: 증강 파이프라인 시연 ---
def main():
    print("\n--- 비디오 처리 시작 ---")

    video_dir = 'DFDC_Folder/test_videos/' 
    
    print("딥러닝 얼굴 탐지기(MTCNN) 로드 중...")
    face_detector = MTCNN()
    print("MTCNN 로드 완료.")

    if face_detector is None:
        print("오류: MTCNN 분류기를 로드할 수 없습니다.")
        return 

    print(f"\n지정된 비디오 경로: {video_dir}")

    try:
        raw_file_list = os.listdir(video_dir)
        print(f"os.listdir로 찾은 파일/폴더 (상위 10개): {raw_file_list[:10]}")
    except Exception as e:
        print(f"os.listdir 오류: {e}")
        print("!!! 경로를 확인하세요. 'video_dir' 변수가 올바른가요?")
        return 

    video_files = glob.glob(os.path.join(video_dir, '*.mp4'))
    print(f"경로에서 찾은 .mp4 파일 개수: {len(video_files)} 개")

    if len(video_files) < 2:
        print(f"!!! 오류: MixUp 시연을 위해 '{video_dir}' 폴더에 2개 이상의 비디오 파일이 필요합니다.")
    else:
        print("MixUp 시연을 위해 2개의 랜덤 비디오에서 얼굴을 추출합니다...")
        
        try:
            video_path_A, video_path_B = random.sample(video_files, 2)
            print(f"  > 랜덤 비디오 A: {os.path.basename(video_path_A)}")
            print(f"  > 랜덤 비디오 B: {os.path.basename(video_path_B)}")
        except ValueError:
            print("!!! 오류: 비디오 리스트 샘플링 실패. 처음 2개 파일을 사용합니다.")
            video_path_A, video_path_B = video_files[0], video_files[1]
        
        face_A = extract_face_from_video(face_detector, video_path_A, frame_num=30)
        face_B = extract_face_from_video(face_detector, video_path_B, frame_num=60) 

        if face_A is not None and face_B is not None:
            
            aug_A = dynamic_face_cutout(face_A.copy())
            aug_B = dynamic_face_cutout(face_B.copy())

            mixed_face = mixup_augmentation(aug_A, aug_B, alpha=1.0)
            
            print("\n--- 결과 이미지를 새 창으로 띄웁니다 ---")
            print("--- 창을 클릭하고 키보드의 아무 키나 누르면 다음으로 넘어갑니다 ---")

            cv2.imshow("1. Original Face A", face_A)
            cv2.waitKey(0) 
            cv2.imshow("2. Original Face B", face_B)
            cv2.waitKey(0)
            cv2.imshow("3. Cutout Face A", aug_A)
            cv2.waitKey(0)
            cv2.imshow("4. Cutout Face B", aug_B)
            cv2.waitKey(0)
            cv2.imshow("5. Final MixUp Result", mixed_face)
            cv2.waitKey(0)
            
            cv2.destroyAllWindows()
            
        else:
            print("\n얼굴 추출에 실패하여 증강 시연을 중단합니다.")

    print("\n--- 모든 작업 완료 ---")

if __name__ == "__main__":
    main()