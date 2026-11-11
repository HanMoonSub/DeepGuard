create table video_artifact_result (
    id integer auto_increment primary key, -- unique ID of the row itself (surrogate key)
    video_id integer not null unique, -- video_result.id (FAKE videos only)
    model_name varchar(20) not null, -- DeepClip model used (deepclip_b0 / deepclip_b5)
    artifact_probs json not null, -- frame mean of the DeepClip 20 text probabilities (PAIR_ORDER order, idx = 5 * region + type)
    created_at timestamp default current_timestamp,

    index video_id_idx(video_id) -- used when looking up by video_id
);
