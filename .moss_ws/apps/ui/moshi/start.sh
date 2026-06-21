bash "$(dirname "$0")/scripts/dev_restart.sh"
uv run reflex run &
sleep 2
exec uv run python main.py