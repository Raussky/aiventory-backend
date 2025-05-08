#!/bin/bash
# /docker-entrypoint.sh
set -e

echo "Activating conda environment..."
source /opt/conda/etc/profile.d/conda.sh
conda activate inventory

# Wait for PostgreSQL to be ready
echo "Waiting for PostgreSQL..."
until PGPASSWORD=$POSTGRES_PASSWORD psql -h "$POSTGRES_SERVER" -U "$POSTGRES_USER" -d "$POSTGRES_DB" -c '\q'; do
  echo "PostgreSQL is unavailable - sleeping"
  sleep 1
done
echo "PostgreSQL is up - continuing"

# Wait for Redis if needed
if [ ! -z "$REDIS_HOST" ]; then
  echo "Waiting for Redis..."
  until redis-cli -h $REDIS_HOST -p $REDIS_PORT --raw ping 2>/dev/null; do
    echo "Redis is unavailable - sleeping"
    sleep 1
  done
  echo "Redis is up - continuing"
fi

# Run database migrations
echo "Running database migrations..."
alembic upgrade head

# Start the application
echo "Starting application..."
exec "$@"