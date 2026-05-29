.PHONY: dev build up down logs clean test install-desktop

# Development with hot-reload
dev:
	docker compose -f docker-compose.dev.yml up --build

# Production build and run
build:
	docker compose build

up:
	docker compose up -d

down:
	docker compose down

# View logs
logs:
	docker compose logs -f

logs-backend:
	docker compose logs -f backend

logs-frontend:
	docker compose logs -f frontend

logs-converter:
	docker compose logs -f converter

# Run tests
test:
	docker compose exec backend pytest tests/ -v

test-backend:
	cd backend && python -m pytest tests/ -v

# Clean up
clean:
	docker compose down -v --rmi local
	rm -rf frontend/node_modules frontend/dist
	find backend -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true

# Desktop launcher
install-desktop:
	./scripts/install-desktop.sh

# Initial setup
setup:
	cp -n .env.example .env || true
	@echo "Edit .env with your settings, then run 'make dev'"
