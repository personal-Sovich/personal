show databases;

create database main;

use main;

create table student(name varchar(20), cllege varchar(20));

insert into student values ('Jack', 'Villanova'), ('Priya', 'bvit');

select * from student;

SELECT *
FROM actor 
WHERE first_name = 'Penelope';

ALTER USER 'root'@'localhost' IDENTIFIED WITH mysql_native_password BY 'password';
flush privileges;