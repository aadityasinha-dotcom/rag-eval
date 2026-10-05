# Schemas

A database contains one or more named schemas, which in turn contain tables. Schemas also contain other kinds of named objects, including data types, functions, and operators. Within one schema, two objects cannot have the same name.

## Creating a schema

To create a schema, use the `CREATE SCHEMA` command. Give the schema a name of your choice. See the [reference](https://example.invalid/create-schema) for the full syntax.

## The public schema

Every new database contains a schema named public. Unless you specify otherwise, tables and other objects are created in it.
