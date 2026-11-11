-- Create the table that stores deepfake analysis results
create table video_result (
    id integer auto_increment primary key, -- unique identifier assigned by the system
    user_id integer null,                  -- ID of the user who uploaded the video (nullable to allow guests)
    video_loc varchar(300) not null unique, -- path/URL where the video file is stored (prevents duplicate entries)
    
    -- Status and result info
    status varchar(20) not null default "PENDING", -- analysis progress status (pending, analyzing, done, failed, etc.)
    label varchar(10) not null,                    -- final verdict (e.g. 'Fake', 'Real')
    score float not null, 
    
    -- Analysis environment info
    version_type varchar(10) not null,
    model_type varchar(10) not null,
    domain_type varchar(20) not null,
    
    -- Other records
    result_msg varchar(200) not null,  -- detailed message about the analysis result or the error
    created_at timestamp default current_timestamp, -- time the record was created (analysis requested)
    
    -- Performance optimization
    index user_id_idx(user_id) -- index for quickly looking up a user's analysis history
);