drop table if exists video_meta_result;

create table deepfake_db.video_meta_result (
    id integer auto_increment primary key,
    video_id integer not null unique,
    fps float not null,
    total_frames integer not null,
    num_sampled integer not null,
    num_extracted integer not null,
    num_detected integer not null,

    index video_id_idx(video_id)
);