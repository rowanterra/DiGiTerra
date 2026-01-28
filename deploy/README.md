# Deployment Assets

Docker and Kubernetes (Helm) assets for running DiGiTerra as a containerized web app.

## Docker

Build the image:

```bash
docker build -f deploy/docker/Dockerfile -t digiterra:local .
```

Run the container:

```bash
docker run --rm -p 5000:5000 digiterra:local
```

The app is available at `http://localhost:5000`.

## Kubernetes (Helm)

The Helm chart in `deploy/helm/digiterra` deploys DiGiTerra to a Kubernetes cluster. You need `kubectl` configured and `helm` installed.

Install the chart:

```bash
helm install digiterra deploy/helm/digiterra \
  --set image.repository=ghcr.io/netl/digiterra \
  --set image.tag=latest
```

Use your own image repo/tag if you build and push elsewhere.

### Persistence

To store uploads and generated visualizations across restarts, enable persistence:

```bash
helm upgrade --install digiterra deploy/helm/digiterra \
  --set image.repository=ghcr.io/netl/digiterra \
  --set image.tag=latest \
  --set persistence.enabled=true \
  --set persistence.size=5Gi
```

Data is stored under `/data/DiGiTerra` in the container. Ensure the cluster can provision a `ReadWriteOnce` PVC and that the mounted volume is writable by the app user (e.g. via `storageClassName` or default provisioner).

### Ingress

To expose the app via an Ingress (e.g. behind a load balancer or reverse proxy):

```bash
helm upgrade --install digiterra deploy/helm/digiterra \
  --set image.repository=ghcr.io/netl/digiterra \
  --set image.tag=latest \
  --set ingress.enabled=true \
  --set ingress.className=nginx \
  --set "ingress.hosts[0].host=digiterra.example.com" \
  --set "ingress.hosts[0].paths[0].path=/" \
  --set "ingress.hosts[0].paths[0].pathType=Prefix"
```

Adjust `ingress.className` and `ingress.hosts` to match your cluster (nginx, traefik, etc.). TLS can be configured via `ingress.tls` and `ingress.annotations` (e.g. cert-manager).

### Chart options

See `deploy/helm/digiterra/values.yaml`. Key fields:

- `replicaCount`: number of pods (default 1).
- `image.repository`, `image.tag`: image to run.
- `service.port`: container port (5000).
- `persistence.enabled`, `persistence.size`, `persistence.storageClassName`: PVC settings.
- `ingress.enabled`, `ingress.className`, `ingress.hosts`, `ingress.tls`: Ingress configuration.
- `resources`: CPU/memory requests and limits for the deployment.
