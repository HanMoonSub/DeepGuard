create table video_artifact_result (
    id integer auto_increment primary key, -- 행 자체의 고유 ID (대리키)
    video_id integer not null unique, -- video_result.id (FAKE 판정 비디오만)
    model_name varchar(20) not null, -- 사용한 DeepClip 모델 (deepclip_b0 / deepclip_b5)
    artifact_probs json not null, -- DeepClip 20개 텍스트 확률의 프레임 평균 (PAIR_ORDER 순서, idx = 5 * region + type)
    created_at timestamp default current_timestamp,

    index video_id_idx(video_id) -- video_id로 조회 시 사용
);
