"""
Script to run the FastAPI server.
"""
import uvicorn
from pathlib import Path
import sys

# Add project root to path
project_root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(project_root))

if __name__ == "__main__":
    # When using reload=True, pass the app as an import string
    uvicorn.run(
        "src.api.main:app",  # Import string for reload to work
        host="0.0.0.0",
        port=8000,
        reload=True,  # Auto-reload on code changes
        log_level="info",
        reload_dirs=[str(project_root / "src")]  # Watch src directory for changes
    )
