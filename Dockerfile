# Use an official Python runtime as a parent image
FROM python:3.9-slim

# Set the working directory in the container
WORKDIR /app

# Copy the current directory contents into the container at /app
COPY . /app

# Install any needed packages specified in requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Make ports 8501-8510 available
EXPOSE 8501-8510

# Set environment variable to run streamlit in container
ENV PYTHONPATH=/app
ENV PYTHONUNBUFFERED=1

# Create a startup script
COPY <<EOF /app/start.sh
#!/bin/bash
port=8501
max_port=8510

while [ $port -le $max_port ]; do
    if ! lsof -i :$port > /dev/null 2>&1; then
        echo "Using port: $port"
        exec streamlit run app.py --server.port=$port --server.address=0.0.0.0
        break
    fi
    port=$((port + 1))
done

if [ $port -gt $max_port ]; then
    echo "No available ports found in range 8501-8510"
    exit 1
fi
EOF

RUN chmod +x /app/start.sh

# Run the startup script
CMD ["/app/start.sh"]
