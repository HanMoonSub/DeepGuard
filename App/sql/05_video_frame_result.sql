create table video_frame_result (
    id integer auto_increment primary key, -- 행 자체의 고유 ID (대리키)
    video_id integer not null, 
    frame_index integer not null, -- 영상 내 프레임 순번.
    frame_time float not null, -- 영상 내 타임스탬프(초)
    score float not null,-- 해당 프레임의 딥페이크 의심 점수 (0.0~1.0)
    
    index video_id_idx(video_id)
);