#!/usr/bin/env bash
set -euo pipefail

# Phone-friendly monorepo bootstrap for Yoshi-Trading-System.
# - Clones or updates the repo
# - Sets up a Python venv + installs deps
# - Creates systemd units that use the venv python
# - Starts: trading-core, kalshi-scanner, yoshi-bridge, clawdbot
#
# Usage:
#   curl -sSL https://raw.githubusercontent.com/DGator86/Yoshi-Trading-System/cursor/information-flow-optimization-8bb3/scripts/vps/bootstrap.sh | bash
#
# Notes:
# - You will be prompted for secrets if missing.
# - Kalshi private key is stored in ~/.kalshi/private_key.pem and referenced via KALSHI_PRIVATE_KEY_FILE.

exec < /dev/tty 2>/dev/null || true

BRANCH="${BRANCH:-cursor/information-flow-optimization-8bb3}"
APP_DIR="${APP_DIR:-/root/Yoshi-Trading-System}"
REPO_URL="${REPO_URL:-https://github.com/DGator86/Yoshi-Trading-System.git}"

GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
CYAN='\033[0;36m'
NC='\033[0m'

say() { echo -e "${CYAN}$*${NC}"; }
ok()  { echo -e "  ${GREEN}✓${NC} $*"; }
warn(){ echo -e "  ${YELLOW}⚠${NC} $*"; }
die() { echo -e "  ${RED}✗${NC} $*"; exit 1; }

need_root() {
  if [ "${EUID:-$(id -u)}" -ne 0 ]; then
    die "Run as root (sudo -i) on the VPS."
  fi
}

install_packages() {
  say "Installing system packages..."
  apt-get update -y -qq >/dev/null 2>&1 || true
  apt-get install -y -qq git curl ca-certificates python3 python3-pip python3-venv build-essential >/dev/null 2>&1 || true
  ok "System packages installed"
}

clone_or_update() {
  say "Cloning/updating repo..."
  if [ -d "$APP_DIR/.git" ]; then
    cd "$APP_DIR"
    git fetch origin "$BRANCH" >/dev/null 2>&1 || git fetch origin >/dev/null 2>&1
  else
    mkdir -p "$(dirname "$APP_DIR")"
    git clone "$REPO_URL" "$APP_DIR"
    cd "$APP_DIR"
  fi
  # Robust branch checkout (works even if branch not created locally yet)
  if git show-ref --verify --quiet "refs/remotes/origin/$BRANCH"; then
    git checkout -B "$BRANCH" "origin/$BRANCH"
  else
    git checkout "$BRANCH" 2>/dev/null || true
    git fetch origin "$BRANCH" >/dev/null 2>&1 || true
    git checkout -B "$BRANCH" "origin/$BRANCH"
  fi
  git pull origin "$BRANCH"
  ok "Repo at $APP_DIR on $BRANCH"
}

setup_venv() {
  say "Setting up venv + Python deps..."
  cd "$APP_DIR"
  python3 -m venv .venv
  # shellcheck disable=SC1091
  source .venv/bin/activate
  python3 -m pip install -U pip >/dev/null
  python3 -m pip install -e ./yoshi-bot >/dev/null
  python3 -m pip install -r ./clawdbot/requirements-ml.txt >/dev/null || true
  python3 -m pip install python-dotenv aiohttp requests cryptography uvicorn >/dev/null
  ok "Python deps installed into $APP_DIR/.venv"
}

ensure_dirs() {
  mkdir -p "$APP_DIR/data/signals" "$APP_DIR/logs"
  ok "Created runtime dirs (data/signals, logs)"
}

upsert_env() {
  local env_path="$APP_DIR/.env"
  if [ ! -f "$env_path" ]; then
    touch "$env_path"
    chmod 600 "$env_path"
  fi

  local key="$1"
  local value="$2"

  if grep -q "^${key}=" "$env_path" 2>/dev/null; then
    # Replace line
    sed -i "s|^${key}=.*|${key}=${value}|" "$env_path"
  else
    echo "" >> "$env_path"
    echo "${key}=${value}" >> "$env_path"
  fi
}

prompt_if_missing() {
  local key="$1"
  local prompt="$2"
  local env_path="$APP_DIR/.env"

  local current=""
  current="$(grep "^${key}=" "$env_path" 2>/dev/null | head -1 | cut -d'=' -f2- || true)"
  if [ -n "${current// /}" ]; then
    ok "$key already set"
    return 0
  fi

  echo -e "${YELLOW}${prompt}${NC}"
  read -r val </dev/tty || true
  if [ -z "${val// /}" ]; then
    warn "Skipping $key (empty)"
    return 0
  fi
  upsert_env "$key" "$val"
  ok "Set $key"
}

setup_kalshi_key_file() {
  say "Configuring Kalshi private key file..."
  local key_file="/root/.kalshi/private_key.pem"
  mkdir -p /root/.kalshi
  chmod 700 /root/.kalshi

  if [ -f "$key_file" ] && grep -q "BEGIN" "$key_file" 2>/dev/null; then
    ok "Found existing $key_file"
  else
    warn "No $key_file found."
    echo -e "${YELLOW}Paste your Kalshi RSA private key now (include BEGIN/END lines). Finish with Ctrl-D.${NC}"
    cat > "$key_file" </dev/tty
    chmod 600 "$key_file"
    ok "Wrote $key_file"
  fi

  # Normalize PEM formatting (handles \\n / single-line cases) using repo util if available.
  if [ -f "$APP_DIR/clawdbot/scripts/lib/pem_utils.py" ]; then
    "$APP_DIR/.venv/bin/python" -c "import sys; sys.path.insert(0,'$APP_DIR/clawdbot'); from scripts.lib.pem_utils import fix_pem_file; fix_pem_file('$key_file')" >/dev/null 2>&1 || true
  fi

  upsert_env "KALSHI_PRIVATE_KEY_FILE" "$key_file"
  ok "Set KALSHI_PRIVATE_KEY_FILE=$key_file"
}

install_units() {
  say "Installing systemd units (venv python)..."
  local py="$APP_DIR/.venv/bin/python"

  cat > /etc/systemd/system/trading-core.service <<EOF
[Unit]
Description=Yoshi Trading Core API
After=network.target

[Service]
Type=simple
User=root
WorkingDirectory=$APP_DIR
EnvironmentFile=$APP_DIR/.env
Environment=PYTHONPATH=$APP_DIR
ExecStart=$py yoshi-bot/scripts/start_trading_core.py --host 0.0.0.0 --port 8000
Restart=always
RestartSec=5
StandardOutput=journal
StandardError=journal

[Install]
WantedBy=multi-user.target
EOF

  cat > /etc/systemd/system/kalshi-scanner.service <<EOF
[Unit]
Description=Yoshi Kalshi Scanner
After=trading-core.service

[Service]
Type=simple
User=root
WorkingDirectory=$APP_DIR
EnvironmentFile=$APP_DIR/.env
Environment=PYTHONPATH=$APP_DIR
ExecStart=$py yoshi-bot/scripts/kalshi_scanner.py --symbol BTCUSDT --loop --bridge --live
Restart=always
RestartSec=10
StandardOutput=journal
StandardError=journal

[Install]
WantedBy=multi-user.target
EOF

  cat > /etc/systemd/system/yoshi-bridge.service <<EOF
[Unit]
Description=Yoshi Bridge (Signal queue -> Trading Core)
After=kalshi-scanner.service

[Service]
Type=simple
User=root
WorkingDirectory=$APP_DIR
EnvironmentFile=$APP_DIR/.env
Environment=PYTHONPATH=$APP_DIR
ExecStart=$py clawdbot/scripts/yoshi-bridge.py --signal-path data/signals/scanner_signals.jsonl --poll-interval 2 --min-edge 2.0
Restart=always
RestartSec=10
StandardOutput=journal
StandardError=journal

[Install]
WantedBy=multi-user.target
EOF

  cat > /etc/systemd/system/clawdbot.service <<EOF
[Unit]
Description=ClawdBot (Telegram Interface)
After=trading-core.service

[Service]
Type=simple
User=root
WorkingDirectory=$APP_DIR
EnvironmentFile=$APP_DIR/.env
Environment=PYTHONPATH=$APP_DIR
ExecStart=$py clawdbot/scripts/telegram-bot.py
Restart=always
RestartSec=5
StandardOutput=journal
StandardError=journal

[Install]
WantedBy=multi-user.target
EOF

  systemctl daemon-reload
  systemctl enable trading-core kalshi-scanner yoshi-bridge clawdbot >/dev/null 2>&1 || true
  ok "Installed + enabled services"
}

start_services() {
  say "Starting services..."
  systemctl restart trading-core
  sleep 1
  systemctl restart kalshi-scanner
  systemctl restart yoshi-bridge
  systemctl restart clawdbot

  ok "Started services"
  echo ""
  systemctl --no-pager --full status trading-core kalshi-scanner yoshi-bridge clawdbot | sed -n '1,60p' || true
}

post_steps() {
  cat <<'EOF'

Next steps (Telegram):
  1) Message your bot, then send: /chatid
  2) That will save TELEGRAM_CHAT_ID into .env automatically.
  3) Restart: sudo systemctl restart clawdbot trading-core kalshi-scanner yoshi-bridge

Useful logs:
  sudo journalctl -u clawdbot -f
  sudo journalctl -u yoshi-bridge -f
  sudo journalctl -u kalshi-scanner -f
  sudo journalctl -u trading-core -f

Signal queue:
  tail -f data/signals/scanner_signals.jsonl
EOF
}

main() {
  need_root
  install_packages
  clone_or_update
  setup_venv
  ensure_dirs

  say "Configuring .env..."
  upsert_env "TRADING_CORE_URL" "http://127.0.0.1:8000"
  prompt_if_missing "TELEGRAM_BOT_TOKEN" "Enter TELEGRAM_BOT_TOKEN:"
  prompt_if_missing "OPENAI_API_KEY" "Enter OPENAI_API_KEY (or OpenRouter/Ollama-compatible key used by your bot):"
  prompt_if_missing "KALSHI_KEY_ID" "Enter KALSHI_KEY_ID:"
  setup_kalshi_key_file

  install_units
  start_services
  post_steps
}

main "$@"

