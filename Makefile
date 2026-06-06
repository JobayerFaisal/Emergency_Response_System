# ============================================================
# Makefile — Emergency Response System
# Usage:  make <target>
# ============================================================

.PHONY: up down build logs seed mock test clean help

## Start all services (real agents + GEE)
up:
	docker-compose up --build

## Start in detached mode
up-d:
	docker-compose up --build -d

## Start with mock publisher (no GEE/Twitter needed — fast for dev)
mock:
	docker-compose --profile mock up --build

## Stop and remove containers
down:
	docker-compose down

## Stop and remove containers + volumes (wipes database)
down-v:
	docker-compose down -v

## Build images without starting
build:
	docker-compose build

## Rebuild a single agent without full rebuild
## Usage:  make rebuild SERVICE=agent3
rebuild:
	docker-compose build $(SERVICE) && docker-compose up -d $(SERVICE)

## Tail logs for all services
logs:
	docker-compose logs -f

## Tail logs for a single service
## Usage:  make log SERVICE=agent1
log:
	docker-compose logs -f $(SERVICE)

## Seed the database (run after services are up)
seed:
	docker-compose exec agent3 python -m src.agents.agent_3_resource.seed_data

## Run smoke tests against all running agents
test:
	@echo "Testing Agent 1..." && \
	curl -sf http://localhost:8001/health | python -m json.tool && \
	echo "Testing Agent 2..." && \
	curl -sf http://localhost:8002/health | python -m json.tool && \
	echo "Testing Agent 3..." && \
	curl -sf http://localhost:8003/health | python -m json.tool && \
	echo "Testing Agent 4..." && \
	curl -sf http://localhost:8004/health | python -m json.tool && \
	echo "Testing Dashboard API..." && \
	curl -sf http://localhost:8005/health | python -m json.tool && \
	echo "All agents healthy!"

## Trigger a manual flood alert through Agent 2 (for pipeline testing)
trigger:
	curl -X POST http://localhost:8002/trigger/flood-alert \
	  -H "Content-Type: application/json" \
	  -d '{"zone_id":"sylhet-sadar","zone_name":"Sylhet Sadar","risk_score":0.85,"severity_level":"critical","confidence":0.9,"risk_factors":{}}'

## Remove all stopped containers and dangling images
clean:
	docker-compose down
	docker image prune -f

## Show disk usage by image
sizes:
	docker images | grep -E "agent|disaster|dashboard|frontend"

## Open API docs in browser (requires xdg-open or open)
docs:
	@which xdg-open && xdg-open http://localhost:8005/docs || open http://localhost:8005/docs

help:
	@echo ""
	@echo "Emergency Response System — make targets"
	@echo ""
	@echo "  make up          Start all services with real agents"
	@echo "  make mock        Start with mock publisher (no GEE/Twitter)"
	@echo "  make down        Stop all services"
	@echo "  make down-v      Stop + wipe database volumes"
	@echo "  make logs        Follow all logs"
	@echo "  make test        Smoke-test all agent /health endpoints"
	@echo "  make trigger     Send a test flood alert through the pipeline"
	@echo "  make seed        Seed Agent 3 resource inventory"
	@echo "  make sizes       Show Docker image sizes"
	@echo "  make clean       Remove stopped containers + dangling images"
	@echo ""
