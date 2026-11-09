# 홈 스크롤 히어로 — AI 이미지·영상 제작 가이드

홈 화면 맨 위(`client/src/components/ScrollHero.js`)는 스크롤 위치에 맞춰 사진과 영상 프레임을 바꿔 그리는 구조다.
아래는 소스를 만든 방법이고, 실제로 들어간 파일은 3단계에 정리했다.
## 스크롤 흐름 (영상도 이 순서를 따라야 함)

| 스크롤 | 단계 | 화면 | 왼쪽 문구 |
| --- | --- | --- | --- |
| 0 ~ 20% | 인트로 | 어두운 공간, 오른쪽에 인물 정면 얼굴. 스캔 라인이 훑음 | 이 얼굴, 진짜일까요? |
| 20 ~ 54% | 01 Low-level | 카메라가 눈가 피부까지 파고듦. 픽셀/압축 블록이 보이고 일부가 붉게 빛남 | 픽셀 하나까지 들여다봅니다 |
| 54 ~ 74% | 02 High-level | 다시 빠져나와 얼굴 전체. 랜드마크 메시, 전역 격자 | 얼굴 전체의 앞뒤를 맞춰봅니다 |
| 74 ~ 88% | 03 XAI | 얼굴 위에 히트맵(빨강·노랑)이 떠오름 | 어디가, 왜 가짜인지 보여드립니다 |
| 88 ~ 100% | 판정 | 붉은 HUD, 정지 | FAKE 99.3% + 분석 버튼 |

데스크톱은 **왼쪽 40%에 글자**가 올라가므로, 인물은 항상 **화면 오른쪽 1/3**에 두어야 한다.

## 1단계 — 키프레임 이미지 4장 만들기 (Higgsfield 이미지 생성)

- 비율 **16:9**, 4장 모두 같은 인물·같은 조명으로 만든다. 첫 장을 만든 뒤 나머지는 그 이미지를 레퍼런스로 넣는다.
- 인물은 **실존 인물이 아닌 AI 생성 얼굴**을 사용한다. 딥페이크 탐지 서비스가 실존 인물 얼굴을 쓰면 오해를 살 수 있다.

**K1 — 인트로**
```
Cinematic 16:9 shot, pitch-black futuristic lab, a photorealistic AI-generated young adult face looking straight at camera, positioned in the right third of the frame, left side empty dark space, cool blue rim light, thin holographic scanning frame around the face, subtle blue grid on the floor, shallow depth of field, moody, high detail skin
```

**K2 — Low-level (눈가 초접사)**
```
Extreme macro close-up of the same person's left eye and cheek skin, the image breaks into visible square pixel blocks and JPEG compression artifacts, a few blocks along the skin blending boundary glow faint red, blue holographic grid overlay, dark cinematic lighting, 16:9
```

**K3 — High-level (얼굴 전체 구조)**
```
Same face and lighting as reference, framed in the right third, floating blue wireframe connecting eyes, nose tip and mouth corners, large faint holographic 7x7 grid over the whole face, analytical HUD feeling, dark background, 16:9
```

**K4 — XAI + 판정**
```
Same face and framing as reference, translucent heatmap blobs in red, orange and yellow glowing over the eyes, nose and mouth, red HUD corner brackets around the face, tense red accent light, dark background, 16:9
```

## 2단계 — 전환 영상 3개 만들기 (시작 프레임 → 끝 프레임)

시작/끝 프레임을 모두 지원하는 영상 모델을 쓴다. 연결된 Higgsfield 카탈로그 기준 후보:

- **Seedance 2.0 Mini**: 가장 저렴함. 720p까지 지원해서 먼저 시험해 보기 좋음
- **Seedance 2.0** 또는 **Cinema Studio Video 3.0**: 품질 우선, 1080p 지원

공통 설정: **16:9, 5초, 소리 끔(generate_audio off)**, 카메라 흔들림 최소.

| 클립 | 시작 → 끝 | 프롬프트 |
| --- | --- | --- |
| clip1 | K1 → K2 | `Smooth continuous dolly push-in from the face into extreme macro of the left eye, no cuts, steady camera, pixels gradually become visible` |
| clip2 | K2 → K3 | `Smooth continuous pull-back from the eye macro to the full face, blue wireframe and grid fade in, no cuts, steady camera` |
| clip3 | K3 → K4 | `Camera holds almost still with a slow gentle push, heatmap blobs fade in over eyes, nose and mouth, HUD turns red, no cuts` |

- 스크롤로 앞뒤를 오가므로 **컷 전환이 없는 한 번의 카메라 이동**이 핵심이다.
- 결과가 덜컥거리면 같은 프롬프트로 2~3번 다시 뽑아 가장 부드러운 것을 고른다.

## 3단계 — 실제로 들어간 파일 (2026-11-09)

크레딧 문제로 최종 소스는 이렇게 만들었다.

- 사진 K1~K4: ChatGPT 이미지 생성 (위 프롬프트, 같은 대화창에서 이어서 수정)
- 줌인 영상 1개: BytePlus Lumina · Seedance 2.0 (16:9, 480p, 4초), 시작 이미지 K1. Lumina에서 2배 업스케일(1728×992)한 파일을 사용
- K1·K3·K4는 구도가 같아서 영상 없이 겹쳐서 전환하고, 눈 접사로 들어갔다 나오는 구간만 영상(정방향 → 역재생)을 쓴다.

| 파일 | 내용 |
| --- | --- |
| `public/hero/k1.webp` ~ `k4.webp` | 인트로 / 눈 접사 / 랜드마크 / 히트맵 (1672×941) |
| `public/hero/clip/f_001~095.webp` | 줌인 영상 프레임 (1322×744). 오른쪽 아래 Lumina 워터마크와 왼쪽 위 AI 표시를 피해 864×496 기준 x 102~763, y 40~412 영역을 2배 해상도로 자름 |

원본(`public/hero/source/`)은 git에 올리지 않는다. 영상을 바꾸면 프레임을 다시 뽑고,
`ScrollHero.js`의 `CLIP_*` 상수(프레임 수, 첫 프레임 얼굴 프레임 위치, 마지막 프레임 눈동자 위치)를 새 영상에 맞게 고친다.
