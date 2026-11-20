"""
make_deploy_requirements.py - writes a requirements.txt with EXACT library versions.

Why: your saved model files (.joblib) were made with the library versions installed
on THIS PC. If Render installs different versions, loading the model can fail or give
wrong answers. Pinning (package==version) makes Render install the same versions.

Run from the project root, inside the activated .venv:
    python make_deploy_requirements.py
"""
import shutil
import sys
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

PACKAGES = ["fastapi", "uvicorn", "pydantic", "scikit-learn", "numpy", "pandas",
            "scipy", "joblib", "matplotlib"]   # matplotlib: src/train.py imports it when loading the model

lines, missing = [], []
for name in PACKAGES:
    try:
        lines.append(f"{name}=={version(name)}")
    except PackageNotFoundError:
        missing.append(name)

if missing:
    print("These packages are not installed in this environment:", ", ".join(missing))
    print("Is the .venv active (prompt starts with (.venv))? Then run: pip install -r requirements.txt")
    sys.exit(1)

req = Path("requirements.txt")
if req.exists() and not Path("requirements-dev.txt").exists():
    shutil.copy(req, "requirements-dev.txt")     # keep your original list as a backup
req.write_text("\n".join(lines) + "\n")
print("Wrote requirements.txt:\n" + "\n".join(lines))
print(f"\nPython on this PC: {sys.version.split()[0]}  (use this for PYTHON_VERSION on Render)")
