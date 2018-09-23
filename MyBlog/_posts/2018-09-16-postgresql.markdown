---
layout: post
title:  "Postgresql"
date:   2018-08-24 13:15:16 +0200
categories: programming
---
This post is an introduction to Postgresql.

### Installation (Mac)

```
> brew install postgresql
```

Then, if upgrading from an older database version:

```
> brew postgresql-upgrade-database
```

To launch at login:

```
>  brew services start postgresql
```

Or to start in console:

```
> pg_ctl -D /usr/local/var/postgres start
```

You may want to create a database:

```
> createdb your_database_name
```

or drop it

```
> dropdb your_database_name
```

or connect to it using `psql`, the command line client:

```
> psql your_database_name
```

Then the list of basic commands:

- `\c` - shows current database and username
- `\l` - lists all databases
- `\d` - describes current database
- `\c database_name` - connects to `database_name`
- `help <command>` - inline help

### Creating a table with an automatically generated ID:

```
psql> CREATE TABLE test_ids(rowid serial, myname character(50));
```

will create one relation and one sequence:

```
alexandrugris=# \d
                   List of relations
 Schema |        Name        |   Type   |     Owner     
--------+--------------------+----------+---------------
 public | test_ids           | table    | alexandrugris
 public | test_ids_rowid_seq | sequence | alexandrugris
(2 rows)
```

Inserting data into the table will automatically generate ID:

```
psql> INSERT INTO test_ids(name) VALUES('Alexandru Gris`);
```

The sequence values will be unique and reserved for each connection, thus one can do on each connection

```
psql> SELECT currval(`test_ids_rowid_seq`);
```

to get the latest inserted ID.

or

```
psql> SELECT nextval(`test_ids_rowid_seq`)
```

to reserve a unique ID only for this connection.

### Character encoding

Postgres automatically encodes characters as UTF-8, thus there is no need for distinction between `varchar` and `nvarchar`.

### Adding a new timestamp column

```
psql>ALTER TABLE test_ids ADD COLUMN borndate timestamp CHECK (borndate > '1/1/1940')
```

with the result

```
alexandrugris=# \d test_ids
                                         Table "public.test_ids"
  Column  |            Type             | Collation | Nullable |                 Default                 
----------+-----------------------------+-----------+----------+-----------------------------------------
 rowid    | integer                     |           | not null | nextval('test_ids_rowid_seq'::regclass)
 myname   | character(50)               |           |          | 
 borndate | timestamp without time zone |           |          | 
Check constraints:
    "test_ids_borndate_check" CHECK (borndate > '1940-01-01 00:00:00'::timestamp without time zone)
```

The `check` keyword is beautiful addition to adding constraints to a column.

For date types there are special keywords like `'yesterday'`, `'today'`, `'tomorrow'`, `'now'`, `'Infinity'`  to make inserts simpler. Note: `'Infinity'` can be used, for instance, to mark an unscheduled event in the future.

```
psql> insert into test_ids(myname, borndate) values ('New Born', 'yesterday');
```

To add timezone support, just use `timestampz` instead of `timestamp` when creating the table.

### Table inheritance

```
psql> CREATE TABLE awesome_ids (description varchar(50), turned_on bit) INHERITS (test_ids);
```

then

```
psql> INSERT INTO awesome_ids(myname, description) VALUES ('Inherited', 'This is inherited');
```

then

```
psql> SELECT * FROM awesome_ids;
psql> SELECT * FROM row_ids;
```

And we get the following two results:

```
alexandrugris=# select * from awesome_ids;
 rowid |                       myname                       | borndate |    description    | turned_on 
-------+----------------------------------------------------+----------+-------------------+-----------
    15 | Inherited                                          |          | This is inherited | 
(1 row)

alexandrugris=# select * from test_ids;
 rowid |                       myname                       |          borndate          
-------+----------------------------------------------------+----------------------------
     7 | Alexandru Gris                                     | 
     8 | Alexandru Gris                                     | 
     9 | Alexandru Gris                                     | 
    10 | Alexandru Gris                                     | 
    12 | Alexandru Gris                                     | 
    13 | New Born                                           | 2018-09-22 00:00:00
    14 | New Born 2                                         | 2018-09-23 12:30:13.424582
    15 | Inherited                                          | 
(8 rows)
```

The inserted row appears in the select from the original table. If we want to see only the `test_ids` without the `awesome_ids`, we should add the `only` keyword to the `select` query:

```
psql>SELECT * FROM ONLY test_ids;
```