# Project Fixes and Logging Configuration

## Summary of Changes

I have debugged the project and identified issues with the `package.json` scripts and the logging configuration in `main.py`.

### 1. Fixed `package.json` Scripts
The original scripts were pointing to a non-existent `backend` directory. I have updated them to point to the correct files in the root directory.

- **`dev:backend`**: Now runs `python main.py` (using `venv` if available).
- **`install:backend`**: Now runs `pip install -r requirements.txt` in the root directory.
- **`backend:setup`**: Now creates a virtual environment in the root directory.

### 2. Configured Logging
I have configured the system to automatically write logs to `server.log` and `backend.log`.

- **`backend.log`**: Contains all logs from the Python backend, including:
    - Application logs (INFO level and above)
    - Uvicorn server logs (startup, requests, errors)
    - Gemini pipeline steps and results
- **`server.log`**: Contains all logs from the Next.js frontend development server.

## How to Run

### Setup (First Time)
1.  **Frontend**: `npm install`
2.  **Backend**: `npm run backend:setup` (This creates a `venv` and installs requirements)

### Development
To run both frontend and backend with logging enabled:

```bash
npm run dev:full
```

This will:
1.  Start the Python backend and redirect output to `backend.log`.
2.  Start the Next.js frontend and redirect output to `server.log`.

You can monitor the logs in real-time using `tail`:

```bash
tail -f backend.log
tail -f server.log
```

## What is in the Log Files?

### `backend.log`
This file will contain structured logs from the FastAPI application:
- **Startup**: "Starting Gemini-powered STL Generator API..."
- **Requests**: "Received file: filename.jpg..."
- **Pipeline Steps**: "Step 1: Input & Ingestion complete", "Step 2: ...", etc.
- **Errors**: Full stack traces if exceptions occur.

### `server.log`
This file will contain standard Next.js output:
- Compilation status
- API route hits
- Frontend errors and warnings

## Verified Files
- `main.py`: Updated to use `logging` module.
- `package.json`: Updated scripts for correct paths and log redirection.
