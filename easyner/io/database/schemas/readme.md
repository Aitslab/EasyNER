#

## DuckDB
- Default is nullable

## Best Practices for Database Schemas - DuckDB
### constraints
DuckDB allows defining constraints such as UNIQUE, PRIMARY KEY, and FOREIGN KEY. These constraints can be beneficial for ensuring data integrity but they have a negative effect on load performance as they necessitate building indexes and performing checks. Moreover, they very rarely improve the performance of queries as DuckDB does not rely on these indexes for join and aggregation operators (see indexing for more details).

´´´Do not define constraints unless your goal is to ensure data integrity.´´´

### Primary keys and constraints
- Only use primary keys, foreign keys, or unique constraints, if these are necessary for enforcing constraints on your data.
- Do not define explicit indexes unless you have highly selective queries and enough memory available.
- If you define an ART index, do so after bulk loading the data to the table. Adding an index prior to loading, either explicitly or via primary/foreign keys, is detrimental to load performance.
