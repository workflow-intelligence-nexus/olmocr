# HTTP Service for PDF Processing

## Setup

1. Install base requirements:
```bash
pip install -r gantry-requirements.txt
```

2. Install HTTP service dependencies:
```bash
pip install -r requirements-http.txt
```

## Running the Service

Start the development server:
```bash
uvicorn http_service:app --reload
```

## API Endpoints

### Process PDF
`POST /process-pdf`

Accepts a PDF file upload and returns processed results.

Example:
```bash
curl -X POST -F "file=@document.pdf" http://localhost:8000/process-pdf
```

## Docker Deployment

1. Build the image:
```bash
docker build -t pdf-processor .
```

2. Run the container:
```bash
docker run -p 8000:8000 pdf-processor
```

## Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `WORKERS` | Number of worker processes | 1 |
| `TIMEOUT` | Request timeout (seconds) | 300 |
| `LOG_LEVEL` | Logging level | info |
