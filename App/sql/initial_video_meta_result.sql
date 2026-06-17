drop table if exists video_meta_result;

create table deepfake_db.video_meta_result (
    id integer auto_increment primary key,-- 행 자체의 고유 ID (대리키).
    video_id integer not null unique,
    fps float not null, -- 초당 프레임수
    total_frames integer not null,-- 영상 전체 프레임 수
    num_sampled integer not null,-- [1단계] 분석을 위해 샘플링한 프레임 수
    num_extracted integer not null,-- [2단계] 샘플 중 얼굴 추출에 성공한 프레임 수
    num_detected integer not null,-- [3단계] 추출된 얼굴 중 딥페이크 score 산출에 성공한 프레임 수

    index video_id_idx(video_id)-- video_id로 메타 조회 시 사용
);