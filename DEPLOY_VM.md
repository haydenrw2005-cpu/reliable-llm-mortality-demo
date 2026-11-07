# Deploy live R + LLM chatbot on a cloud VM

Streamlit Community Cloud cannot run the full CoMoMo/StMoMo + HMD job.
Use a small always-on VM (Ubuntu 22.04+, 4+ GB RAM recommended; more is better for long R runs).

## 1. Prepare the VM

```bash
sudo apt-get update
sudo apt-get install -y docker.io docker-compose-v2 git
sudo usermod -aG docker $USER
# log out/in so docker works without sudo
```

## 2. Clone this project

```bash
git clone https://github.com/haydenrw2005-cpu/reliable-llm-mortality-demo.git
cd reliable-llm-mortality-demo
```

If you develop from the local `reliable llm build` folder, copy/sync that folder to the VM instead.

## 3. Secrets

```bash
cp .env.example .env
nano .env
```

Set at least:

```env
GROQ_API_KEY=...
HMD_USERNAME=...
HMD_PASSWORD=...
PIPELINE_MODE=full
```

## 4. Build and run

```bash
docker compose up -d --build
```

Open: `http://YOUR_VM_PUBLIC_IP:8501`

Open the cloud firewall / security group for TCP **8501** (or put nginx + TLS in front).

## 5. Workflow on the site

1. Enter a plain-English mortality request.
2. LLM extracts + validates config (clarifies if incomplete).
3. Full R CoMoMo pipeline runs (can take a long time — watch stage + log).
4. Deterministic + LLM sanity-check on CSVs.
5. Plain-English answer (only polished as “success” if sanity check passes).

## 6. Logs / restart

```bash
docker compose logs -f app
docker compose restart app
```

## Notes

- Keep **demo** mode for smoke tests without HMD/R.
- Do not commit `.env`.
- First R package / CoMoMo load may be slow; outputs land in `output/`.
