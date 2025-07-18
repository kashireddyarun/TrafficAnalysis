"""
Docker deployment configuration for Traffic Analysis System
Production-ready containerized deployment
"""

# Dockerfile content
DOCKERFILE_CONTENT = '''
FROM python:3.9-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \\
    libgl1-mesa-glx \\
    libglib2.0-0 \\
    libsm6 \\
    libxext6 \\
    libxrender-dev \\
    libgomp1 \\
    libgthread-2.0-0 \\
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Create necessary directories
RUN mkdir -p data/sample_videos data/models output logs

# Set environment variables
ENV PYTHONPATH=/app
ENV STREAMLIT_SERVER_PORT=8501
ENV STREAMLIT_SERVER_ADDRESS=0.0.0.0

# Expose ports
EXPOSE 8501 8000

# Health check
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \\
    CMD curl -f http://localhost:8501/_stcore/health || exit 1

# Default command
CMD ["streamlit", "run", "src/app/streamlit_app.py", "--server.port=8501", "--server.address=0.0.0.0"]
'''

# Docker Compose configuration
DOCKER_COMPOSE_CONTENT = '''
version: '3.8'

services:
  traffic-analyzer:
    build: .
    ports:
      - "8501:8501"  # Streamlit dashboard
      - "8000:8000"  # API endpoint
    volumes:
      - ./data:/app/data
      - ./output:/app/output
      - ./logs:/app/logs
      - ./config:/app/config
    environment:
      - PYTHONPATH=/app
      - CUDA_VISIBLE_DEVICES=0  # Use GPU if available
    restart: unless-stopped
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8501/_stcore/health"]
      interval: 30s
      timeout: 10s
      retries: 3
      start_period: 40s

  redis:
    image: redis:alpine
    ports:
      - "6379:6379"
    restart: unless-stopped
    volumes:
      - redis_data:/data

  nginx:
    image: nginx:alpine
    ports:
      - "80:80"
      - "443:443"
    volumes:
      - ./nginx.conf:/etc/nginx/nginx.conf:ro
      - ./ssl:/etc/nginx/ssl:ro
    depends_on:
      - traffic-analyzer
    restart: unless-stopped

volumes:
  redis_data:
'''

# Nginx configuration
NGINX_CONFIG = '''
events {
    worker_connections 1024;
}

http {
    upstream traffic_app {
        server traffic-analyzer:8501;
    }

    server {
        listen 80;
        server_name localhost;

        # Redirect HTTP to HTTPS
        return 301 https://$server_name$request_uri;
    }

    server {
        listen 443 ssl;
        server_name localhost;

        # SSL configuration
        ssl_certificate /etc/nginx/ssl/cert.pem;
        ssl_certificate_key /etc/nginx/ssl/key.pem;

        # Streamlit specific configuration
        location / {
            proxy_pass http://traffic_app;
            proxy_set_header Host $host;
            proxy_set_header X-Real-IP $remote_addr;
            proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
            proxy_set_header X-Forwarded-Proto $scheme;

            # WebSocket support for Streamlit
            proxy_http_version 1.1;
            proxy_set_header Upgrade $http_upgrade;
            proxy_set_header Connection "upgrade";
            proxy_read_timeout 86400;
        }

        # Static files
        location /static/ {
            alias /app/static/;
            expires 30d;
            add_header Cache-Control "public, immutable";
        }
    }
}
'''

# Kubernetes deployment
K8S_DEPLOYMENT = '''
apiVersion: apps/v1
kind: Deployment
metadata:
  name: traffic-analyzer
  labels:
    app: traffic-analyzer
spec:
  replicas: 3
  selector:
    matchLabels:
      app: traffic-analyzer
  template:
    metadata:
      labels:
        app: traffic-analyzer
    spec:
      containers:
      - name: traffic-analyzer
        image: traffic-analyzer:latest
        ports:
        - containerPort: 8501
        env:
        - name: PYTHONPATH
          value: "/app"
        resources:
          requests:
            memory: "2Gi"
            cpu: "1000m"
          limits:
            memory: "4Gi"
            cpu: "2000m"
        volumeMounts:
        - name: data-volume
          mountPath: /app/data
        - name: config-volume
          mountPath: /app/config
        livenessProbe:
          httpGet:
            path: /_stcore/health
            port: 8501
          initialDelaySeconds: 30
          periodSeconds: 10
        readinessProbe:
          httpGet:
            path: /_stcore/health
            port: 8501
          initialDelaySeconds: 5
          periodSeconds: 5
      volumes:
      - name: data-volume
        persistentVolumeClaim:
          claimName: traffic-data-pvc
      - name: config-volume
        configMap:
          name: traffic-config

---
apiVersion: v1
kind: Service
metadata:
  name: traffic-analyzer-service
spec:
  selector:
    app: traffic-analyzer
  ports:
  - protocol: TCP
    port: 80
    targetPort: 8501
  type: LoadBalancer

---
apiVersion: v1
kind: PersistentVolumeClaim
metadata:
  name: traffic-data-pvc
spec:
  accessModes:
    - ReadWriteOnce
  resources:
    requests:
      storage: 10Gi
'''

def create_deployment_files():
    """Create all deployment configuration files"""
    
    files = {
        'Dockerfile': DOCKERFILE_CONTENT,
        'docker-compose.yml': DOCKER_COMPOSE_CONTENT,
        'nginx.conf': NGINX_CONFIG,
        'k8s-deployment.yaml': K8S_DEPLOYMENT
    }
    
    return files

if __name__ == "__main__":
    # Create deployment files
    files = create_deployment_files()
    
    for filename, content in files.items():
        with open(filename, 'w') as f:
            f.write(content.strip())
        print(f"Created {filename}")
    
    print("\\nDeployment files created successfully!")
    print("\\nQuick deployment commands:")
    print("Docker: docker-compose up -d")
    print("Kubernetes: kubectl apply -f k8s-deployment.yaml")
