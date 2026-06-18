drop table if exists image_result;

create table deepfake_db.image_result (
    id integer auto_increment primary key,
    user_id integer null, -- null 허용
    image_loc varchar(300) not null unique,
    status varchar(20) not null default "PENDING", -- status 추가
    label varchar(10) not null,
    score float not null,
    face_conf float not null,
    face_ratio float not null,
    face_brightness float not null,
    version_type varchar(10) not null,
    model_type varchar(10) not null,
    domain_type varchar(20) not null,
    result_msg varchar(200) not null,
    created_at timestamp default current_timestamp,
    
    index user_id_idx(user_id)
);