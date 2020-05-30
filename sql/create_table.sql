CREATE TABLE weight (
    id serial primary key,
    name varchar(255) not null,
    time timestamp not null,
    weight float not null
)