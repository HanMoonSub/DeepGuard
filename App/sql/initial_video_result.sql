drop table if exists video_result;

create table deepfake_db.video_result (
    id integer auto_increment primary key,
    user_id integer null, -- null 허용
    video_loc varchar(300) not null unique,
    status varchar(20) not null default "PENDING",
    label varchar(10) not null,
    score float not null,
    face_conf float not null,
    face_ratio float not null,
    face_brightness float not null,
    version_type varchar(10) not null,
    model_type varchar(10) not null,
    domain_type varchar(20) not null,
    result_msg varchar(200) not null,
    -- 상세 분석 메타데이터
    fps float null,
    total_frames integer null,
    num_sampled integer null,
    num_extracted integer null,
    num_detected integer null,
    created_at timestamp default current_timestamp,
    
    index user_id_idx(user_id)
);