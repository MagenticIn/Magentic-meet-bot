# Organization Deployment (Today)

This deploy guide gets your meeting transcription + translation + AI summary running on a public HTTPS URL for your organization.

## 1) Provision a VM

- Ubuntu 22.04 or 24.04
- Minimum: 4 vCPU / 8 GB RAM / 80 GB disk
- Open ports: `80`, `443`
- Point DNS A record for your domain (for example `meetnotes.yourcompany.com`) to the VM public IP

## 2) Install Docker

```bash
sudo apt-get update
sudo apt-get install -y ca-certificates curl gnupg
sudo install -m 0755 -d /etc/apt/keyrings
curl -fsSL https://download.docker.com/linux/ubuntu/gpg | sudo gpg --dearmor -o /etc/apt/keyrings/docker.gpg
echo \
  "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.gpg] https://download.docker.com/linux/ubuntu \
  $(. /etc/os-release && echo "$VERSION_CODENAME") stable" | \
  sudo tee /etc/apt/sources.list.d/docker.list > /dev/null
sudo apt-get update
sudo apt-get install -y docker-ce docker-ce-cli containerd.io docker-buildx-plugin docker-compose-plugin
```

## 3) Prepare app config

```bash
git clone <your-repo-url> Magentic-meet-bot
cd Magentic-meet-bot
cp deploy/.env.prod.example .env.prod
```

Edit `.env.prod` with real values:
- `APP_DOMAIN`
- `OPENAI_API_KEY` (speech-to-text with `TRANSCRIPTION_BACKEND=openai` and summarisation)
- `HF_TOKEN` only if you use `TRANSCRIPTION_BACKEND=whisper` (local faster-whisper + diarization)
- `GOOGLE_EMAIL`, `GOOGLE_PASSWORD`
- strong DB password values

Generate Caddy dashboard password hash:

```bash
docker run --rm caddy:2.8-alpine caddy hash-password --plaintext "your-strong-password"
```

Put that hash into `DASHBOARD_PASS_HASH`.

## 4) Start production stack

```bash
docker compose --env-file .env.prod -f docker-compose.prod.yml up -d --build
```

## 5) Verify

```bash
docker compose --env-file .env.prod -f docker-compose.prod.yml ps
curl -I https://$APP_DOMAIN/health
```

Dashboard:
- `https://$APP_DOMAIN/`

API docs:
- `https://$APP_DOMAIN/docs`

## 6) Trigger a real meeting

```bash
curl -X POST "https://$APP_DOMAIN/api/v1/meetings/trigger" \
  -u "$DASHBOARD_USER:your-strong-password" \
  -H "Content-Type: application/json" \
  -d '{"meeting_url":"https://meet.google.com/xxx-xxxx-xxx","title":"Org rollout test"}'
```

## Notes for Hindi / Hinglish meetings

- Set `WHISPER_LANGUAGE=hi` in `.env.prod` for Hindi-heavy conversations.
- This improves Hindi capture and avoids English-biased ASR.

## Operational commands

```bash
# logs
docker compose --env-file .env.prod -f docker-compose.prod.yml logs -f api bot pipeline

# restart services
docker compose --env-file .env.prod -f docker-compose.prod.yml restart api bot pipeline

# update after code changes
git pull
docker compose --env-file .env.prod -f docker-compose.prod.yml up -d --build
```

