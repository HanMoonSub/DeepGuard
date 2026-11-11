create table image_artifact_result (
    id integer auto_increment primary key, -- unique ID of the row itself (surrogate key)
    image_id integer not null unique, -- image_result.id (FAKE images only)
    model_name varchar(20) not null, -- DeepClip model used (deepclip_b0 / deepclip_b5)
    artifact_probs json not null, -- DeepClip 20 text probabilities (PAIR_ORDER order, idx = 5 * region + type)
    created_at timestamp default current_timestamp,

    index image_id_idx(image_id) -- used when looking up by image_id
);
