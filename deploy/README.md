# Deployment

Docker and Kubernetes (Helm) assets for running DiGiTerra as a web app in a container.

## Docker

Build the image:

```bash
docker build -f deploy/docker/Dockerfile -t digiterra:local .
```

Run the container:

```bash
docker run --rm -p 5000:5000 digiterra:local
```

The app answers at `http://localhost:5000`. The image sets `URL_PREFIX=digiterra`, so the app also answers at `http://localhost:5000/digiterra/`. Both the root and the prefixed routes serve the page, so health checks and reverse proxies can use either.

The Dockerfile `CMD` binds gunicorn to port 5000 with one worker.

## Kubernetes with Helm

The chart in `deploy/helm/digiterra` needs `kubectl` configured for the cluster and `helm` installed.

No public image is published. Build the image with the Dockerfile above, push it to a registry the cluster can pull from, and pass that repository at install time. `image.repository` in `values.yaml` is empty, and with the value empty the pod image name is `:latest`, which no registry serves.

```bash
helm install digiterra deploy/helm/digiterra \
  --set image.repository=registry.example.com/digiterra \
  --set image.tag=latest
```

### Persistence

To keep uploads and generated visualizations across restarts:

```bash
helm upgrade --install digiterra deploy/helm/digiterra \
  --set image.repository=registry.example.com/digiterra \
  --set image.tag=latest \
  --set persistence.enabled=true \
  --set persistence.size=5Gi
```

Data is stored under `/data/DiGiTerra` in the container. The cluster must provision a `ReadWriteOnce` PVC, and the mounted volume must be writable by the `app` user.

### Ingress

To expose the app through an Ingress:

```bash
helm upgrade --install digiterra deploy/helm/digiterra \
  --set image.repository=registry.example.com/digiterra \
  --set image.tag=latest \
  --set ingress.enabled=true \
  --set ingress.className=nginx \
  --set "ingress.hosts[0].host=digiterra.example.com" \
  --set "ingress.hosts[0].paths[0].path=/" \
  --set "ingress.hosts[0].paths[0].pathType=Prefix"
```

`ingress.className` and `ingress.hosts` match the cluster's controller (nginx, traefik, and so on). TLS is set through `ingress.tls` and `ingress.annotations`, for example for cert-manager.

### Chart values

`deploy/helm/digiterra/values.yaml` defines:

- `replicaCount`: number of pods (default 1). Session state is in process memory, so more than one replica splits sessions across pods.
- `image.repository`, `image.tag`: image to run. The repository has no default.
- `service.port`: Service port and `containerPort` (default 5000). The Dockerfile `CMD` binds gunicorn to 5000, so a different value needs a matching change to the Dockerfile.
- `persistence.enabled`, `persistence.size`, `persistence.storageClassName`: PVC settings.
- `ingress.enabled`, `ingress.className`, `ingress.hosts`, `ingress.tls`: Ingress settings.
- `resources`: CPU and memory requests and limits.
- `env`: `DIGITERRA_HOST`, `DIGITERRA_PORT`, and the upload limit variables.

## Production settings

- Workers: one gunicorn worker, because session state is in process memory.
- HTTPS: a reverse proxy or an Ingress with TLS in front of the app.
- Secrets: none in the repository. Credentials, if any, come from environment variables or a secret manager.
- Logging: `LOG_LEVEL=INFO`, or `DEBUG` for troubleshooting. Logs go to stderr and, when the directory is writable, to `LOG_DIR` from `python_scripts/config.py`.
- Paths: `DIGITERRA_BASE_DIR`, `DIGITERRA_APP_SUPPORT_DIR`, `DIGITERRA_OUTPUT_DIR`, and `URL_PREFIX`. When `URL_PREFIX` is set, both `/` and `/<prefix>/` are served.
