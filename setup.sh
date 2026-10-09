#!/usr/bin/env bash
set -e

# Colors for terminal output
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${BLUE}=== Setting up RAG-FOR-RAGS ===${NC}"

# 1. Check Python version
PYTHON_CMD="python3"
if ! command -v $PYTHON_CMD &> /dev/null; then
    echo "python3 could not be found. Please install Python 3.10+."
    exit 1
fi

# 2. Setup virtual environment if not present
if [ ! -d ".venv" ]; then
    echo -e "${YELLOW}Creating virtual environment in .venv...${NC}"
    $PYTHON_CMD -m venv .venv
else
    echo -e "${GREEN}Virtual environment already exists in .venv.${NC}"
fi

# Activate virtual environment
source .venv/bin/activate

# 3. Upgrade pip and install dependencies
echo -e "${YELLOW}Installing/updating dependencies from requirements.txt...${NC}"
pip install --upgrade pip -q
pip install -r requirements.txt -q

# 4. Create .env if it doesn't exist
if [ ! -f ".env" ]; then
    echo -e "${YELLOW}Creating default .env from .env.example...${NC}"
    cp .env.example .env
fi

# 5. Ingest sample data if database doesn't exist or is empty
if [ ! -d "chroma_db" ] || [ -z "$(ls -A chroma_db 2>/dev/null)" ]; then
    echo -e "${YELLOW}Chroma database not found or empty. Ingesting data/sample.pdf...${NC}"
    python scripts/reingest.py
else
    echo -e "${GREEN}Chroma database already populated.${NC}"
fi

echo -e "\n${GREEN}=== Setup complete! ===${NC}"
echo -e "To start using the project:"
echo -e "  1. Activate the environment:  ${BLUE}source .venv/bin/activate${NC}"
echo -e "  2. Run the interactive CLI:   ${BLUE}python run.py cli${NC}"
echo -e "  3. Or start the API server:   ${BLUE}python run.py api${NC}"
echo -e "  4. Run tests:                 ${BLUE}pytest${NC}\n"
