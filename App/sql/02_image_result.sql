create table image_result (
    id integer auto_increment primary key,
    user_id integer null, -- null 허용
    image_loc varchar(300) not null unique,
    status varchar(20) not null default "PENDING", -- status 추가
    label varchar(10) not null,
    score float not null,
    artifact_idx tinyint null, -- DeepClip(v2) top-1 artifact 인덱스 (PAIR_ORDER: 5 * region + type), v1은 null
    artifact_prob float null, -- top-1 artifact 확률, v1은 null
    version_type varchar(10) not null,
    model_type varchar(10) not null,
    domain_type varchar(20) not null,
    result_msg varchar(200) not null,
    created_at timestamp default current_timestamp,
    
    index user_id_idx(user_id)
);