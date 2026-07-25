create table user
(
    id integer auto_increment primary key,
    name varchar(100) not null,
    email varchar(100) not null unique,
    hashed_password varchar(200) not null
);