create database SVS;
show databases;
create database SVS2;
use SVS2;
show tables;
create table Customer(
Fname varchar (100),
     Lname varchar (100),
     Customer_id varchar(100),
     Bdate date,
     Address varchar (200),
     Sex char (1),
     primary key (Customer_id)
);
drop table signatures;
create table Signatures(
c_id varchar (100),
sign_id varchar(100),
     Sign1 blob,
     Sign2 blob,
     Sign3 blob,
     Sign4 blob,
     primary key (sign_id)
);
select* from signatures;
create table Employee(
Fname varchar (100),
     Lname varchar (100),
     emp_id varchar(100),
     is_admin char(1),
     Address varchar (200),
     Sex char (1),
     designation varchar(100),
     primary key (emp_id)
);
create table upload_sign(
sig_id varchar(100),
cust_id varchar(100),
emp_id varchar(100)
);
select* from upload_sign;
select* from employee;
select* from customer;
select* from signatures;
alter table signatures add foreign key(c_id) references customer(Customer_id);
alter table upload_sign add foreign key(sig_id) references Signatures(sign_id);
alter table upload_sign add foreign key(cust_id) references customer(Customer_id);
alter table upload_sign add foreign key(emp_id) references employee(emp_id);