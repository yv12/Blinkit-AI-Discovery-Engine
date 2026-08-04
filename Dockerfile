FROM python:3.10-slim

# Install git and git-lfs to pull the large data files
RUN apt-get update && apt-get install -y git git-lfs && apt-get clean

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the source code (including .git folder so we can run git lfs pull)
COPY . .

# Run git lfs pull to fetch the actual large files from GitHub
RUN git lfs install && git lfs pull

CMD uvicorn api:app --host 0.0.0.0 --port $PORT
