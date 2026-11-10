create table video_frame_result (
    id integer auto_increment primary key, -- 행 자체의 고유 ID (대리키)
    video_id integer not null, 
    frame_index integer not null, -- 영상 내 프레임 순번.
    frame_time float not null, -- 영상 내 타임스탬프(초)
    score float not null,-- 해당 프레임의 딥페이크 의심 점수 (0.0~1.0)
    artifact_idx tinyint null, -- DeepClip(v2) 해당 프레임 top-1 artifact 인덱스 (PAIR_ORDER), v1은 null
    artifact_prob float null, -- 해당 프레임 top-1 artifact 확률, v1은 null
    
    index video_id_idx(video_id)
);