# Deploying PSX-Sentient on AWS (EC2 + S3 + CloudWatch)

Everything runs on one EC2 instance with Docker Compose: API (:8000), dashboard (:8501),
MLflow tracking + registry (:5000, artifacts in **S3**), Prometheus (:9090), Grafana (:3000).
Container logs go to **CloudWatch Logs** (group `psx-sentient`).

## 1. S3 bucket (MLflow artifact store)
S3 → Create bucket → name e.g. `psx-sentient-mlflow-<yourname>` → region **us-east-1** → defaults → Create.

## 2. IAM role for the instance
IAM → Roles → Create role → Trusted entity **AWS service / EC2** → attach policies:
- `AmazonS3FullAccess` (or a policy limited to your bucket)
- `CloudWatchLogsFullAccess`

Name it `psx-sentient-ec2-role`.

## 3. Launch the instance
EC2 → Launch instance:
- Name `psx-sentient`, AMI **Ubuntu Server 24.04 LTS**
- Instance type **m7i-flex.large** (2 vCPU, 8 GB; free-plan eligible, paid from credits). `c7i-flex.large` (4 GB) also works if you skip FinBERT (`INSTALL_NLP=false`).
- Key pair: create one (download the `.pem`)
- Network → security group inbound rules: SSH 22 (My IP); Custom TCP 8000, 8501, 3000, 5000, 9090 (Anywhere, for the demo)
- Storage **30 GiB gp3**
- Advanced details → IAM instance profile **psx-sentient-ec2-role**; **Metadata response hop limit = 2** (lets containers use the role);
  User data:
  ```bash
  #!/bin/bash
  curl -fsSL https://get.docker.com | sh
  usermod -aG docker ubuntu
  apt-get install -y git
  ```
- (Recommended) Elastic IPs → Allocate → Associate with the instance, so the URL never changes.

## 4. First start (SSH in: `ssh -i key.pem ubuntu@<public-ip>`)
```bash
git clone https://github.com/JunaidMu/PSX-Sentient.git && cd PSX-Sentient
cat > .env <<'EOF'
DAGSHUB_USER=<dagshub user>
DAGSHUB_TOKEN=<dagshub token>
AWS_REGION=us-east-1
MLFLOW_ARTIFACTS_DESTINATION=s3://psx-sentient-mlflow-<yourname>/mlflow
INSTALL_NLP=true
ENABLE_NLP=1
PUBLIC_GRAFANA_URL=http://<public-ip>:3000/d/psx-sentient
PUBLIC_MLFLOW_URL=http://<public-ip>:5000
EOF
docker compose build
docker compose up -d mlflow
docker compose run --rm trainer sh scripts/bootstrap.sh          # data from DagsHub, train, register champion
docker compose -f docker-compose.yml -f docker-compose.aws.yml up -d
```
Open `http://<public-ip>:8501` (dashboard), `:8000/docs` (API), `:3000` (Grafana), `:5000` (MLflow).

## 5. Closed loop (demo)
```bash
docker compose run --rm trainer python -m src.monitoring.replay --api http://api:8000   # live traffic Jul–Nov 2026
docker compose run --rm trainer python -m src.monitoring.loop --skip-fetch              # drift -> retrain -> promote
docker compose run --rm trainer python -m src.monitoring.retrain --rollback             # roll back if needed
curl -X POST localhost:8000/model/reload
```

## 6. Schedules (server clock is UTC)
`crontab -e` on the instance and add:
```
0 4 * * 1-5 cd ~/PSX-Sentient && docker compose run --rm trainer sh scripts/premarket.sh >> logs/premarket.log 2>&1
0 2 * * 6   cd ~/PSX-Sentient && docker compose run --rm trainer sh scripts/weekly_retrain.sh >> logs/weekly_retrain.log 2>&1
```
- **Daily 09:00 PKT:** latest prices/macro, the five PSX daily files and overnight news (FinBERT), the pre-market
  snapshot, DVC versioning (+ pointers to the git `data` branch), a drift check (retrains only for urgent drift),
  and an S3 backup of the MLflow registry.
- **Saturday 07:00 PKT:** weekly retraining check — retrains on drift vs. the champion's own training data or after
  ~4 weeks of new data (never more than once per trading week), promotes only if better, backs up the registry.
- Optional `GITHUB_DATA_TOKEN` in `.env` (fine-grained token, Contents read/write on this repo) enables the `data` branch.
- Disaster recovery on a fresh server: `docker compose run --rm trainer python -m src.monitoring.backup --restore`
  restores the registry; model files are already in S3 and data comes back with `dvc pull`.

## 7. CI/CD secrets (GitHub → Settings → Secrets and variables → Actions)
`DAGSHUB_USER`, `DAGSHUB_TOKEN` (pipeline + gate), `EC2_HOST` (public IP), `EC2_USER` (`ubuntu`),
`EC2_SSH_KEY` (contents of the `.pem`). Pushes to `main` deploy only after tests, validation and the gate pass.

Live deployment (this project): API http://3.104.86.72:8000/docs · dashboard http://3.104.86.72:8501 ·
Grafana http://3.104.86.72:3000 · MLflow http://3.104.86.72:5000 · Prometheus http://3.104.86.72:9090 (region ap-southeast-2).

## Cost
m7i-flex.large ≈ US$0.10/hour (~$2.3/day) from the free-plan credits. **Stop the instance after grading.**
