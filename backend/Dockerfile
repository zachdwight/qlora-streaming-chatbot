FROM nvidia/cuda:12.1.1-runtime-ubuntu22.04

# Install Python and dependencies
RUN apt-get update && apt-get install -y python3 python3-pip git && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 5000
CMD ["python3", "backend.py"]
