# Voucher OCR Telegram Bot

A Telegram bot that accepts voucher/receipt images, extracts structured data via OCR, and stores it in a searchable database — all running in Docker.

## Tech Stack

| Layer | Technology |
|---|---|
| API | [FastAPI](https://fastapi.tiangolo.com/) |
| Telegram Bot | [aiogram 3](https://docs.aiogram.dev/) |
| OCR Engine | [PaddleOCR](https://github.com/PaddlePaddle/PaddleOCR) |
| Task Queue | [Celery](https://docs.celeryq.dev/) |
| Message Broker | [Redis 7](https://redis.io/) |
| Database | [PostgreSQL 16](https://www.postgresql.org/) |
| Containerisation | [Docker](https://www.docker.com/) / Docker Compose |

## Setup

### 1. Clone the repository

```bash
git clone <repo-url>
cd voucher-ocr
```

### 2. Configure environment

```bash
cp .env.example .env
```

Open `.env` and fill in:

| Variable | Description |
|---|---|
| `POSTGRES_PASSWORD` | Strong password for the database |
| `TELEGRAM_BOT_TOKEN` | Token from [@BotFather](https://t.me/BotFather) |
| `ADMIN_TELEGRAM_IDS` | Comma-separated Telegram user IDs for admins |

### 3. Start all services

```bash
docker compose up -d
# or via Makefile
make up
```

### 4. Run database migrations

```bash
make migrate
```

## Bot Commands

| Command | Description |
|---|---|
| `/start` | Greeting and usage instructions |
| `/help` | Display all available commands |
| `/search <query>` | Search stored vouchers by keyword |
| `/stats` | Show your personal voucher statistics |
| `/list` | List your most recent vouchers |

Send any photo or document to the bot to trigger OCR processing.

## Makefile Targets

```bash
make up          # Start all containers in detached mode
make down        # Stop and remove containers
make build       # Rebuild images
make logs        # Stream logs from all services
make migrate     # Run Alembic database migrations
make test        # Run the test suite inside the api container
make shell-api   # Open a bash shell in the api container
make shell-db    # Open a psql shell in the postgres container
```

## Storage Structure

Data is persisted in named Docker volumes:

```
voucher_postgres_data   – PostgreSQL data files
voucher_redis_data      – Redis AOF persistence
voucher_originals       – Raw uploaded images  (/data/originals)
voucher_crops           – Cropped OCR regions  (/data/crops)
voucher_models          – PaddleOCR model files (/data/models)
```

## Architecture

```
Telegram  ──▶  bot (aiogram3)
                │
                ▼
              Redis  ◀──  worker (Celery + PaddleOCR)
                │
                ▼
            PostgreSQL
                │
                ▼
             api (FastAPI)
```

The **bot** receives images and queues OCR jobs via Redis. The **worker** runs PaddleOCR, writes results to PostgreSQL, and notifies the user through the bot. The **api** exposes a REST interface for search and administration.

## Development

To rebuild after code changes:

```bash
make build
make up
```

To view logs for a specific service:

```bash
docker compose logs -f worker
```
