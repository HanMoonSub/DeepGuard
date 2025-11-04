import os, json
from glob import glob
from pathlib import Path
from colorama import Fore, Style


def get_original_video_paths(root_dir, cropped=False, basename=False):
    """
    주어진 루트 디렉터리에서 REAL(진짜) 비디오 경로를 반환합니다.
    
    Args:
        root_dir (str): 데이터셋의 루트 디렉터리
        cropped (bool): crop된 영상만 포함할지 여부
        basename (bool): 파일명만 반환할지 여부 (True면 video.mp4 형태)
    
    Returns:
        list: REAL 영상의 경로 리스트 (basename 옵션에 따라 전체 경로 또는 파일명만)
    """

    originals = set()   # 전체 경로 저장용 (예: root_dir/video_folder/video.mp4)
    originals_v = set() # 파일명만 저장용 (예: video.mp4)

    # 각 하위 폴더의 metadata.json 파일 탐색
    for json_path in glob(os.path.join(root_dir, "*/metadata.json")):
        """
        metadata.json 예시:

        {
            "video_01.mp4": {
                "label": "REAL"
            },
            "video_02.mp4": {
                "label": "FAKE",
                "original": "video_01.mp4"
            }
        }
        """

        dir = Path(json_path).parent  # metadata.json이 위치한 폴더 경로
        with open(json_path, "r") as f:
            metadata = json.load(f)

        # metadata 내 모든 (video, 정보) 쌍 순회
        for k, v in metadata.items():
            # crop 모드일 경우, crops 디렉토리 내 영상 폴더 존재 여부 확인
            if cropped:
                crop_dir = os.path.join(root_dir, 'crops', k[:-4])
                if not os.path.exists(crop_dir):
                    continue  # crop된 영상이 없으면 건너뜀

            # REAL 비디오의 경우 original 필드는 존재하지 않음
            original = v.get("original", None)
            if v["label"] == "REAL":
                original = k
                originals_v.add(original)
                originals.add(os.path.join(dir, original))

    # set → list 변환
    originals = list(originals)   # 전체 경로
    originals_v = list(originals_v)  # 파일명만

    print(f"{Fore.BLUE}{Style.BRIGHT}### Total REAL Videos in Dataset are {len(originals)}")

    # basename=True면 파일명만 반환
    return originals_v if basename else originals



def get_original_with_fakes(root_dir, cropped=False):
    """
    주어진 루트 디렉터리에서 (REAL 비디오, FAKE 비디오) 쌍을 반환합니다.

    Args:
        root_dir (str): 데이터셋 루트 디렉터리
        cropped (bool): crop된 영상만 포함할지 여부

    Returns:
        list: (real_video_name, fake_video_name) 형태의 튜플 리스트
    """
    pairs = []

    # 각 metadata.json 탐색
    for json_path in glob(os.path.join(root_dir, "*/metadata.json")):
        with open(json_path, "r") as f:
            metadata = json.load(f)

        for k, v in metadata.items():
            # crop된 영상 존재 여부 확인
            if cropped:
                crop_dir = os.path.join(root_dir, "crops", k[:-4])
                if not os.path.exists(crop_dir):
                    continue

            # FAKE 비디오일 경우 원본(real) 이름 추출
            original = v.get("original", None)
            if v["label"] == "FAKE":
                # 확장자(.mp4) 제외 후 튜플로 추가
                pairs.append((original[:-4], k[:-4]))

    # 예: [('real_video_01', 'fake_video_03'), ...]
    return pairs



def get_video_paths(root_dir, cropped=False):
    """
    모든 비디오의 (비디오 경로, 박스 정보 경로)을 반환합니다.

    Args:
        root_dir (str): 데이터셋 루트 디렉터리
        cropped (bool): crop된 영상만 포함할지 여부

    Returns:
        list: (video_path, bboxes_path) 형태의 튜플 리스트
    """
    paths = []

    # metadata.json 파일 탐색
    for json_path in glob(os.path.join(root_dir, "*/metadata.json")):
        dir = Path(json_path).parent  # 각 비디오 폴더 경로

        with open(json_path, "r") as f:
            metadata = json.load(f)

        for k, v in metadata.items():
            # crop 모드 시, crop 폴더 존재 확인
            if cropped:
                crop_dir = os.path.join(root_dir, "crops", k[:-4])
                if not os.path.exists(crop_dir):
                    continue

            # FAKE 영상이면 원본 이름 사용, REAL이면 자기 자신
            original = v.get("original", None)
            if not original:
                original = k  # 예: video_01.mp4

            # bounding box 정보 파일 경로
            bboxes_path = os.path.join(root_dir, "boxes", original[:-4] + ".json")

            # bbox 파일이 없으면 건너뜀
            if not os.path.exists(bboxes_path):
                continue

            # (비디오 경로, bbox 경로) 추가
            paths.append((os.path.join(dir, k), bboxes_path))

    # 예: [('root/.../video_01.mp4', 'root/boxes/video_01.json'), ...]
    return paths
