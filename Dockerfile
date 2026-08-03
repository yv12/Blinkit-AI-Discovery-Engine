# Use a lightweight Python base image
FROM python:3.10-slim

# Hugging Face Spaces run as a non-root user (uid 1000) by default.
# It's good practice to set this up explicitly.
RUN useradd -m -u 1000 user
USER user
ENV PATH="/home/user/.local/bin:$PATH"

# Set the working directory
WORKDIR /app

# Copy the requirements file first to leverage Docker cache
COPY --chown=user requirements.txt .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application code
# Assuming the data folder with generated artifacts is committed/copied here
COPY --chown=user . .

# Hugging Face Spaces expose port 7860 by default
EXPOSE 7860

# Command to run the application
CMD ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "7860"]
