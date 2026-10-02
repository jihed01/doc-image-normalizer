# Doc Image Normalizer

Microservice for document image preprocessing before OCR.

The goal of this project is to provide a reliable and reproducible image preprocessing pipeline for scanned documents, with a focus on improving downstream OCR quality.

## Current Scope

The project is currently in its initial setup phase.

The first development iterations will focus on:

* document image deskewing
* adaptive binarization
* image denoising
* OCR-oriented preprocessing
* API integration
* automated testing
* containerization and CI/CD

The image processing pipeline will be implemented incrementally as the project evolves.

## Technology Stack

* Python 3.14
* OpenCV
* NumPy
* FastAPI
* Uvicorn
* Pydantic
* Pytest
* Ruff
* Black
* Mypy
* Docker
* GitHub Actions
* uv

## Development Environment

The project uses `uv` for Python dependency and environment management.

Install the project dependencies with:

```bash
uv sync --extra dev
```

Run the test suite:

```bash
uv run pytest
```

Run Ruff:

```bash
uv run ruff check .
```

Check code formatting with Black:

```bash
uv run black --check .
```

Run Mypy:

```bash
uv run mypy src tests
```

## Docker

Build the Docker image:

```bash
docker build -f docker/Dockerfile -t doc-image-normalizer .
```

Run the container:

```bash
docker run --rm doc-image-normalizer
```

The current container command is only a temporary smoke test. The application runtime will be added in a later development iteration.

## Project Status

The project is currently in the initial scaffolding and development phase.

Implemented so far:

* Python project configuration
* dependency management with uv
* test configuration with Pytest
* code quality checks with Ruff, Black and Mypy
* Docker configuration
* GitHub Actions CI
* initial project structure

The image processing algorithms and production API will be developed incrementally in subsequent tasks.

## Quality Checks

Before submitting changes, the following checks should pass:

```bash
uv run pytest
uv run ruff check .
uv run black --check .
uv run mypy src tests
```

GitHub Actions runs the automated test suite on pushed branches and pull requests.

## License

This project is currently a portfolio project and does not yet define a public license.
