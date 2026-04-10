drop table if exists video_result;

create table deepfake_db.video_result (
    id integer auto_increment primary key,
    user_id integer not null,
    video_loc varchar(300) not null unique,
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