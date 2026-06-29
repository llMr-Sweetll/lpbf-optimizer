FROM python:3.12-slim

# Install system dependencies useful for scientific Python and h5py.
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libhdf5-dev \
    && rm -rf /var/lib/apt/lists/*

# Create a non-root user to run the application.
RUN useradd -m -u 1000 lpbf

WORKDIR /app

# Copy the project source and install in editable mode.
COPY --chown=lpbf:lpbf . /app
RUN pip install --no-cache-dir -e ".[dev]"

USER lpbf

CMD ["bash"]
