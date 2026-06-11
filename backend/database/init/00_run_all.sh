#!/bin/bash
# /docker-entrypoint-initdb.d/00_run_all.sh
#
# Postgres runs every file in /docker-entrypoint-initdb.d/ exactly ONCE —
# only when the data directory is first initialised (fresh volume).
# On subsequent restarts, this script is NOT run (the data dir already exists).
#
# The Dashboard API's db.py also runs all migrations on every startup, so
# this script is a belt-and-suspenders measure for a clean first boot.
#
# The $POSTGRES_DB variable is set by the postgres container from the
# POSTGRES_DB environment variable in docker-compose.yml.

set -e
SQL_DIR="/docker-entrypoint-initdb.d/sql"

echo "=== Running AMAS schema migrations ==="
for f in $(ls "$SQL_DIR"/*.sql | sort); do
    echo "  Applying: $(basename $f)"
    psql -v ON_ERROR_STOP=1 --username "$POSTGRES_USER" --dbname "$POSTGRES_DB" -f "$f"
done
echo "=== All migrations applied ==="