
# INstall UV
curl -LsSf https://astral.sh/uv/install.sh | sh
# Install target python version
uv python install 3.12

# Create VENV
uv venv --python 3.12 .venv
source .venv/bin/activate
python --version

# Install requirements
uv pip install -r requirements.txt