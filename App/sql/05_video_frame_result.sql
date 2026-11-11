create table video_frame_result (
    id integer auto_increment primary key, -- unique ID of the row itself (surrogate key)
    video_id integer not null, 
    frame_index integer not null, -- frame number within the video.
    frame_time float not null, -- timestamp within the video (seconds)
    score float not null,-- deepfake suspicion score of this frame (0.0~1.0)
    
    index video_id_idx(video_id)
);