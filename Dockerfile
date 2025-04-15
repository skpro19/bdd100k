# Use an official Python runtime as a parent image
FROM python:3.9-slim

# Set the working directory in the container
WORKDIR /app

# Copy the requirements file into the container at /app
COPY requirements.txt .

# Install any needed packages specified in requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Copy the Python scripts into the container at /app
COPY data_analysis.py .
COPY vis.py .

# Define the entry point for the container.
# This assumes the 'data' directory and 'assets' directory (for output) 
# will be mounted as volumes when the container is run.
ENTRYPOINT ["python", "data_analysis.py"] 