# Use a base image that includes Python and necessary build tools
FROM python:3.9-slim

# Set working directory
WORKDIR /app

# Copy requirements.txt to the container
COPY requirements.txt .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application code
COPY . .

# Expose the port number on which the app will run
EXPOSE 8000

# Command to run the Flask application
CMD ["python", "app.py"]
