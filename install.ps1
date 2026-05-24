<#
.SYNOPSIS
    Install Nine Men's Morris (NMM) on Windows.
.DESCRIPTION
    Creates a Python virtual environment, installs dependencies, and
    optionally installs Ollama for LLM commentary features.
.PARAMETER NoOllama
    Skip Ollama installation and model download entirely.
.PARAMETER Model
    Override the Ollama model to pull (default: read from data\settings.json,
    fallback to llama3.1:8b).
.PARAMETER Yes
    Non-interactive: assume "Yes" to all prompts (installs Ollama by default).
.EXAMPLE
    .\install.ps1
    .\install.ps1 -NoOllama
    .\install.ps1 -Model "mistral:7b"
    .\install.ps1 -Yes
.NOTES
    If PowerShell refuses to run this script with "running scripts is disabled",
    either double-click install.bat (recommended) or run from PowerShell:
        powershell -ExecutionPolicy Bypass -File .\install.ps1
#>

#Requires -Version 5.1
[CmdletBinding()]
param(
    [switch]$NoOllama,
    [string]$Model = "",
    [switch]$Yes
)

# NOTE: Set-StrictMode is deliberately NOT enabled here.
# Strict mode interacts badly with $LASTEXITCODE inspection after native
# commands and with optional JSON properties, producing confusing failures.
$ErrorActionPreference = "Continue"

$NMM_DIR  = $PSScriptRoot
if (-not $NMM_DIR) { $NMM_DIR = (Get-Location).Path }

$VENV_DIR = Join-Path $NMM_DIR ".venv"
$VENV_PY  = Join-Path $VENV_DIR "Scripts\python.exe"

function Write-Info { param($msg) Write-Host "[NMM] $msg" -ForegroundColor Green }
function Write-Warn { param($msg) Write-Host "[NMM] $msg" -ForegroundColor Yellow }
function Write-Fail { param($msg) Write-Host "[NMM] ERROR: $msg" -ForegroundColor Red; exit 1 }

Write-Host ""
Write-Host "  +======================================+" -ForegroundColor Cyan
Write-Host "  |   Nine Men's Morris -- Installer     |" -ForegroundColor Cyan
Write-Host "  +======================================+" -ForegroundColor Cyan
Write-Host ""
Write-Info "Install directory: $NMM_DIR"

# Refuse to run inside paths PowerShell can't handle reliably.
if ($NMM_DIR -match '[\[\]]') {
    Write-Fail "Install path contains '[' or ']' which PowerShell handles poorly. Move the folder to a simpler path (e.g. C:\NMM) and re-run."
}

# === 1. Read model from settings.json =======================================
$SETTINGS = Join-Path $NMM_DIR "data\settings.json"
if ($Model -eq "" -and (Test-Path -LiteralPath $SETTINGS)) {
    try {
        $raw = Get-Content -LiteralPath $SETTINGS -Raw -ErrorAction Stop
        $s = $raw | ConvertFrom-Json -ErrorAction Stop
        if ($s -and ($s.PSObject.Properties.Name -contains 'ollama_model')) {
            if ($s.ollama_model) { $Model = [string]$s.ollama_model }
        }
    } catch {
        Write-Warn "Could not parse data\settings.json -- using default model."
    }
}
if ($Model -eq "") { $Model = "llama3.1:8b" }

# === 2. Find a usable Python 3.10+ ==========================================
Write-Info "Looking for Python 3.10 or newer..."

function Get-PythonVersion {
    param([string]$Exe, [string[]]$PreArgs = @())
    try {
        $argList = @()
        $argList += $PreArgs
        $argList += "--version"
        # Capture both streams; some shims write to stderr.
        $output = & $Exe @argList 2>&1
        if ($null -eq $output) { return $null }
        $text = ($output | Out-String)
        if ($text -match "Python\s+(\d+)\.(\d+)") {
            return [pscustomobject]@{
                Major = [int]$Matches[1]
                Minor = [int]$Matches[2]
                Raw   = $text.Trim()
            }
        }
    } catch {
        return $null
    }
    return $null
}

# Build candidate list: each entry is @{ Exe = ...; Args = @(...) }
# 'py -3' is preferred on Windows because it skips the Microsoft Store stub.
$candidates = @(
    @{ Exe = "py";      Args = @("-3") },
    @{ Exe = "py";      Args = @() },
    @{ Exe = "python";  Args = @() },
    @{ Exe = "python3"; Args = @() }
)

$pythonExe  = $null
$pythonArgs = @()

foreach ($c in $candidates) {
    $cmd = Get-Command $c.Exe -ErrorAction SilentlyContinue
    if (-not $cmd) { continue }

    # Skip the Windows App Execution Alias stub for python.exe (it lives under
    # WindowsApps and opens the Microsoft Store instead of running Python).
    if ($c.Exe -eq "python" -and $cmd.Source -match "WindowsApps") {
        Write-Warn "Skipping Microsoft Store 'python' stub at $($cmd.Source)."
        continue
    }

    $v = Get-PythonVersion -Exe $c.Exe -PreArgs $c.Args
    if ($null -eq $v) { continue }

    if ($v.Major -gt 3 -or ($v.Major -eq 3 -and $v.Minor -ge 10)) {
        $pythonExe  = $c.Exe
        $pythonArgs = $c.Args
        $displayCmd = if ($c.Args.Count -gt 0) { "$($c.Exe) $($c.Args -join ' ')" } else { $c.Exe }
        Write-Info "Using Python $($v.Major).$($v.Minor) via '$displayCmd'."
        break
    } else {
        Write-Warn "'$($c.Exe)' is Python $($v.Major).$($v.Minor) -- need 3.10+; trying next."
    }
}

if (-not $pythonExe) {
    Write-Host ""
    Write-Host "  Python 3.10 or newer was not found." -ForegroundColor Red
    Write-Host "  Install it from https://www.python.org/downloads/" -ForegroundColor White
    Write-Host "  IMPORTANT: tick 'Add python.exe to PATH' during install." -ForegroundColor White
    Write-Host ""
    Write-Fail "Python 3.10+ is required."
}

# === 3. Create the virtual environment ======================================
if (Test-Path -LiteralPath $VENV_PY) {
    Write-Info "Existing .venv found -- reusing it."
} else {
    Write-Info "Creating virtual environment in .venv ..."
    & $pythonExe @pythonArgs -m venv "$VENV_DIR"
    if ($LASTEXITCODE -ne 0 -or -not (Test-Path -LiteralPath $VENV_PY)) {
        Write-Fail "Failed to create virtual environment. Make sure the Python 'venv' module is available."
    }
}

# === 4. Upgrade pip via 'python -m pip' (NEVER call pip.exe directly on Windows) ===
# Calling pip.exe to upgrade itself fails on Windows because the .exe is held
# open while it tries to overwrite itself. 'python -m pip' avoids this.
Write-Info "Upgrading pip / setuptools / wheel..."
& $VENV_PY -m pip install --disable-pip-version-check --upgrade pip setuptools wheel
if ($LASTEXITCODE -ne 0) {
    Write-Warn "pip upgrade returned a non-zero exit code; continuing anyway."
}

# === 5. Install Python requirements =========================================
$REQS = Join-Path $NMM_DIR "requirements.txt"
if (-not (Test-Path -LiteralPath $REQS)) {
    Write-Fail "requirements.txt not found at $REQS"
}

Write-Info "Installing Python requirements (this can take a few minutes)..."
& $VENV_PY -m pip install --disable-pip-version-check -r "$REQS"
if ($LASTEXITCODE -ne 0) {
    Write-Host ""
    Write-Warn "Some packages failed to install."
    Write-Warn "On Windows, chromadb sometimes needs the 'Microsoft C++ Build Tools'."
    Write-Warn "Install from https://visualstudio.microsoft.com/visual-cpp-build-tools/"
    Write-Warn "then re-run install.bat."
    Write-Fail "Python dependency install failed."
}
Write-Info "Python packages installed."

# === 6. Ollama (optional) ===================================================
$installOllama = $false

if ($NoOllama) {
    Write-Info "Skipping Ollama (-NoOllama flag set). LLM features will be disabled."
} elseif ($Yes) {
    $installOllama = $true
    Write-Info "Auto-installing Ollama (-Yes flag set)."
} else {
    Write-Host ""
    Write-Host "  Ollama enables AI commentary and LLM-powered move analysis." -ForegroundColor White
    Write-Host "  The default model ($Model) is ~5 GB and requires a download." -ForegroundColor White
    Write-Host ""
    $choice = Read-Host "[NMM] Install Ollama for LLM features? [Y/n]"
    if ($choice -eq "" -or $choice -match "^[Yy]") {
        $installOllama = $true
    } else {
        Write-Info "Skipping Ollama. Run install.ps1 again (without -NoOllama) to add it later."
    }
}

if ($installOllama) {
    $ollamaExe = Get-Command ollama -ErrorAction SilentlyContinue
    if ($ollamaExe) {
        $ollamaVer = (& ollama --version 2>&1 | Select-Object -First 1)
        Write-Info "Ollama already installed ($ollamaVer)."
    } else {
        Write-Info "Downloading Ollama installer..."
        $installer = Join-Path $env:TEMP "OllamaSetup.exe"
        try {
            # Use TLS 1.2+; older Windows defaults can refuse the GitHub redirect.
            try {
                [System.Net.ServicePointManager]::SecurityProtocol = `
                    [System.Net.ServicePointManager]::SecurityProtocol -bor `
                    [System.Net.SecurityProtocolType]::Tls12
            } catch {}

            Invoke-WebRequest -Uri "https://ollama.com/download/OllamaSetup.exe" `
                -OutFile $installer -UseBasicParsing -ErrorAction Stop
        } catch {
            Write-Warn "Could not download Ollama installer automatically: $($_.Exception.Message)"
            Write-Warn "Install it manually from https://ollama.com then re-run install.bat."
            $installOllama = $false
        }

        if ($installOllama) {
            Write-Info "Running Ollama installer (follow the prompts)..."
            try {
                Start-Process -FilePath $installer -Wait -ErrorAction Stop
            } catch {
                Write-Warn "Ollama installer did not complete: $($_.Exception.Message)"
                $installOllama = $false
            }
            Remove-Item -LiteralPath $installer -Force -ErrorAction SilentlyContinue

            # Refresh PATH so 'ollama' is visible in this session.
            $machinePath = [System.Environment]::GetEnvironmentVariable("Path", "Machine")
            $userPath    = [System.Environment]::GetEnvironmentVariable("Path", "User")
            $env:Path = "$machinePath;$userPath"

            if (-not (Get-Command ollama -ErrorAction SilentlyContinue)) {
                Write-Warn "Ollama not found in PATH after install."
                Write-Warn "Restart your terminal, then run:  ollama pull $Model"
                $installOllama = $false
            } else {
                Write-Info "Ollama installed."
            }
        }
    }
}

if ($installOllama) {
    # === 7. Start Ollama service if not running =============================
    $ollamaReady = $false
    try {
        Invoke-WebRequest -Uri "http://localhost:11434/api/tags" `
            -UseBasicParsing -TimeoutSec 3 -ErrorAction Stop | Out-Null
        $ollamaReady = $true
        Write-Info "Ollama service already running."
    } catch {}

    if (-not $ollamaReady) {
        Write-Info "Starting Ollama service..."
        try {
            Start-Process -FilePath "ollama" -ArgumentList "serve" -WindowStyle Hidden -ErrorAction Stop
        } catch {
            Write-Warn "Could not start 'ollama serve': $($_.Exception.Message)"
        }
        $waited = 0
        while (-not $ollamaReady -and $waited -lt 15) {
            Start-Sleep -Seconds 1
            $waited++
            try {
                Invoke-WebRequest -Uri "http://localhost:11434/api/tags" `
                    -UseBasicParsing -TimeoutSec 2 -ErrorAction Stop | Out-Null
                $ollamaReady = $true
            } catch {}
        }
        if ($ollamaReady) {
            Write-Info "Ollama service started."
        } else {
            Write-Warn "Ollama service did not respond in time. Start it manually: ollama serve"
        }
    }

    # === 8. Pull LLM model ===================================================
    if ($ollamaReady) {
        Write-Info "Checking for model '$Model'..."
        & ollama show $Model 2>&1 | Out-Null
        if ($LASTEXITCODE -eq 0) {
            Write-Info "Model '$Model' already present."
        } else {
            Write-Info "Pulling '$Model' -- this may take several minutes..."
            & ollama pull $Model
            if ($LASTEXITCODE -ne 0) {
                Write-Warn "Model pull failed. Run manually later: ollama pull $Model"
            } else {
                Write-Info "Model '$Model' ready."
            }
        }
    }
}

# === 9. Create data directories =============================================
foreach ($d in @("data\games", "data\session_memory", "data\chroma")) {
    $path = Join-Path $NMM_DIR $d
    if (-not (Test-Path -LiteralPath $path)) {
        New-Item -ItemType Directory -Path $path -Force | Out-Null
    }
}
Write-Info "Data directories ready."

# === Done ===================================================================
Write-Host ""
Write-Host "  Installation complete!" -ForegroundColor Green
Write-Host ""
Write-Host "  To start the game:" -ForegroundColor White
Write-Host "    .\run_nmm.bat" -ForegroundColor Cyan
Write-Host ""
