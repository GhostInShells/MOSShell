---
name: 'sqlite_dogfood'
description: 'SQLite dogfood — provides the sqlite channel with a seeded demo db.'
singleton: true
exec:
  command: python
  args: main.py
---

Provides a `sqlite` channel (`ghoshell_moss.channels.sqlite_channel`) for dogfooding.

Seeded demo database: `<node>/runtime/dogfood.db` — tables `users` and `events`.

CTML examples:

    <sqlite:open db_path="/path/to/runtime/dogfood.db" name="mem"/>
    <sqlite:tables name="mem"/>
    <sqlite:schema name="mem" table="users"/>
    <sqlite:query name="mem">SELECT * FROM users</sqlite:query>
    <sqlite:sample name="mem" table="events" limit="5"/>
    <sqlite:close name="mem"/>
