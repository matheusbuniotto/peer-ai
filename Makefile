.PHONY: demo sql

demo:
	uv run peer demo

sql:
	uv run duckdb -c ".read analysis/eval_queries.sql"
