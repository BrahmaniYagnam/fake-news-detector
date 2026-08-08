# Deployment Guide

## Docker Deployment

1. Build the Docker image:
   ```bash
   docker compose build
   ```
2. Start the container:
   ```bash
   docker compose up
   ```
3. Confirm the API is available at `http://localhost:8000`
4. Open the Swagger UI at `http://localhost:8000/docs`

## Environment Variables

The application reads the following values from environment variables:

- `APP_NAME`
- `APP_VERSION`
- `MODEL_DIR`
- `LOG_LEVEL`

## Deploying to Render or Railway

- Set up a new service and connect the repository.
- Use the Dockerfile to build the container.
- Set the public port to `8000`.
- Add environment variables in the service settings.
- Deploy and verify that `/docs` loads successfully.
