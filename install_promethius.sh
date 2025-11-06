# DOWNLOAD AND INSTALL
cd /tmp
wget https://github.com/prometheus/prometheus/releases/download/v2.54.1/prometheus-2.54.1.linux-amd64.tar.gz
tar xvfz prometheus-2.54.1.linux-amd64.tar.gz
sudo cp prometheus-2.54.1.linux-amd64/prometheus /usr/local/bin/
sudo cp prometheus-2.54.1.linux-amd64/promtool /usr/local/bin/

# CREATE DIRECTORIES
sudo mkdir -p /etc/prometheus /var/lib/prometheus

# CREATE CONFIG FILE
sudo tee /etc/prometheus/prometheus.yml << 'EOF'
global:
  scrape_interval: 15s  # How often to scrape targets
  evaluation_interval: 15s

# Scrape targets
scrape_configs:
  # Prometheus monitoring itself
  - job_name: 'prometheus'
    static_configs:
      - targets: ['localhost:9090']

  # node_exporter for system metrics
  - job_name: 'node_exporter'
    static_configs:
      - targets: ['localhost:9100']

  # NVIDIA DCGM Exporter for GPU metrics
  - job_name: 'nvidia_dcgm_exporter'
    static_configs:
      - targets: ['localhost:9400']

  # Your chess bot (uncomment when you add metrics)
  # - job_name: 'chess_bot'
  #   static_configs:
  #     - targets: ['localhost:8000']
EOF

# CREATE SYSTEM SERVICE
sudo tee /etc/systemd/system/prometheus.service << EOF
[Unit]
Description=Prometheus
After=network.target

[Service]
Type=simple
User=nobody
ExecStart=/usr/local/bin/prometheus \\
  --config.file=/etc/prometheus/prometheus.yml \\
  --storage.tsdb.path=/var/lib/prometheus \\
  --web.console.templates=/etc/prometheus/consoles \\
  --web.console.libraries=/etc/prometheus/console_libraries \\
  --storage.tsdb.retention.time=15d
Restart=on-failure

[Install]
WantedBy=multi-user.target
EOF

# Fix permissions
sudo chown -R nobody:nogroup /var/lib/prometheus

# Start it
sudo systemctl daemon-reload
sudo systemctl start prometheus
sudo systemctl enable prometheus

