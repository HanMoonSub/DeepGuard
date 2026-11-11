create table video_meta_result (
    id integer auto_increment primary key,-- unique ID of the row itself (surrogate key).
    video_id integer not null unique,
    fps float not null, -- frames per second
    total_frames integer not null,-- total number of frames in the video
    num_sampled integer not null,-- [step 1] number of frames sampled for analysis
    num_extracted integer not null,-- [step 2] number of sampled frames where a face was extracted
    num_detected integer not null,-- [step 3] number of extracted faces where a deepfake score was computed

    index video_id_idx(video_id)-- used when looking up meta data by video_id
);