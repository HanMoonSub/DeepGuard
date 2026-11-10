create table image_artifact_result (
    id integer auto_increment primary key, -- 행 자체의 고유 ID (대리키)
    image_id integer not null unique, -- image_result.id (FAKE 판정 이미지만)
    model_name varchar(20) not null, -- 사용한 DeepClip 모델 (deepclip_b0 / deepclip_b5)
    artifact_probs json not null, -- DeepClip 20개 텍스트 확률 (PAIR_ORDER 순서, idx = 5 * region + type)
    created_at timestamp default current_timestamp,

    index image_id_idx(image_id) -- image_id로 조회 시 사용
);
