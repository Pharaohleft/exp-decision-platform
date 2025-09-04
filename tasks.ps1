param([string]$task="help")

function Setup {
  if (-not (Test-Path .\.venv\Scripts\python.exe)) {
    python -m venv .venv
  }
  .\.venv\Scripts\pip install --upgrade pip
  .\.venv\Scripts\pip install -r requirements.txt
}

function RunApp {
  .\.venv\Scripts\python .\app\app.py
}

function Test {
  .\.venv\Scripts\pytest
}

switch ($task) {
  "setup" { Setup }
  "run"   { RunApp }
  "test"  { Test }
  default { Write-Host "Tasks: setup | run | test" }
}
