drop table if exists video_frame_result;

create table deepfake_db.video_frame_result (
    id integer auto_increment primary key
    video_id integer not null,
    frame_index integer not null,
    frame_time float not null, -- 영상 내 타임스탬프(초)
    score float not null,
    face_conf float not null,
    face_ratio float not null,
    face_brightness float not null,
    
    index video_id_idx(video_id)
);