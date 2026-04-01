drop table if exists user_image;

create table deepfake_db.user_image(
	id integer auto_increment primary key,
    user_id integer null,
    image_loc varchar(300) not null unique,
    created_at timestamp default current_timestamp
);

create index user_id_idx on user_image(user_id);