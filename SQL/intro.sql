show databases;

create database main;

use main;

create table student(name varchar(20), cOllege varchar(20));

insert into student values ('Jack', 'Villanova'), ('Priya', 'bvit');

select * from student;

SELECT first_name AS First, actor.last_name AS Last
FROM actor 
WHERE first_name = 'Penelope'
LIMIT 10;

ALTER USER 'root'@'localhost' IDENTIFIED WITH mysql_native_password BY 'password';
flush privileges;