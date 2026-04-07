# Diabetes Risk Support App

Medical-style diabetes risk assessment application with:

- FastAPI backend
- React + TypeScript + Vite frontend
- Ant Design with custom product styling
- Framer Motion transitions
- Google OAuth sign-in
- PostgreSQL-ready persistence
- PDF lab-report parsing
- saved prediction history
- Docker + docker-compose
- optional Cloudflare Tunnel

## Product Shape

This repo now behaves like a lightweight medical SaaS product instead of a generic survey:

- auth-first experience
- `My Health` dashboard
- wizard-based clinical intake
- optional lab-values step
- editable parsed PDF values
- animated prediction result
- saved timestamped history

## Main Endpoints

- `GET /health`
- `GET /auth/google`
- `GET /auth/google/login`
- `GET /auth/google/callback`
- `POST /auth/logout`
- `GET /feature-schema`
- `GET /profile`
- `GET /history`
- `POST /predict`
- `POST /predict-batch`
- `POST /parse-pdf`

## Backend Notes

The trained model still comes from the notebook-derived Pima pipeline, but the backend now accepts richer clinical inputs and derives the model feature vector from them. This is intentionally framed as a risk support workflow only, not diagnosis.

## Environment

Copy the template first:

```powershell
Copy-Item .env.example .env
```

Important auth settings:

- `GOOGLE_CLIENT_ID`
- `GOOGLE_CLIENT_SECRET`
- `BACKEND_BASE_URL`
- `FRONTEND_URL`
- `JWT_SECRET_KEY`
- `SESSION_SECRET_KEY`

For local Google OAuth, add a Google OAuth redirect URI matching:

```text
http://127.0.0.1:8005/auth/google/callback
```

## Local Run

### Backend

```powershell
cd C:\Users\Fanta\Downloads\DIP_APP\backend
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
uvicorn app.main:app --reload --host 127.0.0.1 --port 8005
```

### Frontend

```powershell
cd C:\Users\Fanta\Downloads\DIP_APP\frontend
npm.cmd install
npm.cmd run dev
```

Open:

- frontend: [http://127.0.0.1:5173](http://127.0.0.1:5173)
- backend docs: [http://127.0.0.1:8005/docs](http://127.0.0.1:8005/docs)

## Docker Run

```powershell
Copy-Item .env.example .env
docker compose up --build
```

Docker services:

- frontend: [http://127.0.0.1:3000](http://127.0.0.1:3000)
- backend: [http://127.0.0.1:8005/docs](http://127.0.0.1:8005/docs)
- postgres: `127.0.0.1:5432`

By default the backend uses PostgreSQL in Docker and local SQLite in bare local development unless you override `DATABASE_URL`.

## Cloudflare Tunnel

Set `CLOUDFLARE_TUNNEL_TOKEN` in `.env`, then run:

```powershell
docker compose --profile tunnel up --build
```

## Repo Structure

```text
.
|-- backend
|   |-- app
|   |   |-- auth.py
|   |   |-- config.py
|   |   |-- db.py
|   |   |-- feature_schema.py
|   |   |-- main.py
|   |   |-- models.py
|   |   |-- schemas.py
|   |   |-- ml
|   |   |   |-- data.py
|   |   |   |-- inference.py
|   |   |   `-- training.py
|   |   `-- services
|   |       |-- metrics.py
|   |       `-- pdf_parser.py
|   |-- Dockerfile
|   `-- requirements.txt
|-- frontend
|   |-- src
|   |   |-- api.ts
|   |   |-- App.tsx
|   |   |-- main.tsx
|   |   |-- styles.css
|   |   `-- types.ts
|   |-- Dockerfile
|   |-- nginx.conf
|   `-- package.json
|-- docker-compose.yml
`-- .env.example
```

## Implementation Assumptions

- Google OAuth is real and redirect-based, but it requires your own Google Cloud credentials.
- Lab parsing is text-extraction based; image-only PDFs still need OCR if you want stronger support for scanned reports.
- The richer clinical intake is mapped into the existing notebook-trained model through derived features, so this should still be treated as a support estimate only.
