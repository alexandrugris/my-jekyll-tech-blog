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

```sql
psql> CREATE TABLE test_ids(rowid serial, myname character(50));
```

will create one relation and one sequence:

```sql
alexandrugris=# \d
                   List of relations
 Schema |        Name        |   Type   |     Owner     
--------+--------------------+----------+---------------
 public | test_ids           | table    | alexandrugris
 public | test_ids_rowid_seq | sequence | alexandrugris
(2 rows)
```

Inserting data into the table will automatically generate ID:

```sql
psql> INSERT INTO test_ids(name) VALUES('Alexandru Gris`);
```

The sequence values will be unique and reserved for each connection, thus one can do on each connection

```sql
psql> SELECT currval(`test_ids_rowid_seq`);
```

to get the latest inserted ID.

or

```sql
psql> SELECT nextval(`test_ids_rowid_seq`)
```

to reserve a unique ID only for this connection.

### Character encoding

Postgres automatically encodes characters as UTF-8, thus there is no need for distinction between `varchar` and `nvarchar`.

### Adding a new timestamp column

```sql
psql>ALTER TABLE test_ids ADD COLUMN borndate timestamp CHECK (borndate > '1/1/1940')
```

with the result

```sql
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

```sql
psql> insert into test_ids(myname, borndate) values ('New Born', 'yesterday');
```

To add timezone support, just use `timestampz` instead of `timestamp` when creating the table.

### Table inheritance

```sql
psql> CREATE TABLE awesome_ids (description varchar(50), turned_on bit) INHERITS (test_ids);
```

then

```sql
psql> INSERT INTO awesome_ids(myname, description) VALUES ('Inherited', 'This is inherited');
```

then

```sql
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

```sql
psql>SELECT * FROM ONLY test_ids;
```

### Performance tuning

pgAdmin is a great tool to play around and visualize query execution plans:

![pgAdmin]({{site.url}}/assets/postgres_1.png)

Postgres has the ability to set per-query parameters to optimize execution.

Inheritance is the foundation for vertical partitioning in Postgres and it brings the extra benefit of adding a semantic twist to it. One shouldn't try to add inheritance to an existing table, but rather create a new set of tables and move the data over to them.

### Full Text Search

GIN (generalized inverted index) creation:

```sql
psql>ALTER TEST test_ids ADD COLUMN my_text TEXT;
psql>CREATE INDEX ftidx_myname ON test_ids USING GIN (to_tsvector('english'::regconfig, my_text::text));
```

Full read [here](https://www.postgresql.org/docs/10/static/datatype-textsearch.html)

How does the `to_tsvector()` work?

```sql
psql>select to_tsvector('english', 'this is the best thing I have seen in my life');
```

outputs

```
'best':4 'life':11 'seen':8 'thing':5
```

where the number is the word in the sentence I entered as parameter. And a small variation:

```
psql>select to_tsvector('english', 'this is the best thing I have seen in my life, oh my life, again my life';
```

outputs

```
'best':4 'life':11,14,17 'oh':12 'seen':8 'thing':5
```

The next step, of course, is to query the `ts_vector`. For this, postgres offers the `ts_query` function together with [a set of operators to match the vector and the query](https://www.postgresql.org/docs/10/static/functions-textsearch.html).

```sql
psql>select to_tsvector('english', 'this is the best thing I have seen in my life, oh my life again my life') @@ to_tsquery('english', 'life');
```

returns true.

Ranking the results:

```sql
psql>select ts_rank(v, q) from to_tsvector('english', 'this is the best thing I have seen in my life, life, life, super life') v, to_tsquery('english', 'life & best') q;
```

or, using an already existing table:

```sql
insert into test_ids(my_text) values ('Hello, Hello');
insert into test_ids(my_text) values ('Hello World');
insert into test_ids(my_text) values ('Hello Alexandru');
insert into test_ids(my_text) values ('Hello Alexandru Gris');
insert into test_ids(my_text) values ('Hello Alexandru Hello Gris Hello World');

select t.my_text,  ts_rank(v, q) as rank 
    from test_ids t, to_tsvector('english', t.my_text) v, to_tsquery('english', 'hello') q 
    where v @@ q 
    order by rank desc;
```

If we want to make things even faster and skip the generation of `tsvector` at query runtime, we can do the following:

```sql

--- add a column with type tsvector
alter table test_ids add column words_vector tsvector;

--- index it
create index idxtxt on test_ids using gin(words_vector);

--- create a trigger to update this column with tsvectors
create trigger update_words_vector before insert or update on test_ids
  for each row execute procedure tsvector_update_trigger('words_vector', 'pg_catalog.english', 'my_text');

--- force self update the table
update test_ids set my_text = my_text;

--- run the query on this particular column
select t.my_text, ts_rank(t.words_vector, q)
    as rank from test_ids t, to_tsquery('english', 'hello') q
    where t.words_vector @@ q
    order by rank desc;
```