<#
setup.ps1 - One-command setup of stock_predict (AlphaEdge) on a Windows 10/11 PC.

  Double-click setup.bat, or run in PowerShell from the project folder:
      powershell -ExecutionPolicy Bypass -File setup.ps1

What it does (every step is skipped if it is already done, so it is safe to run again):
  1. creates .env from .env.example (database settings)
  2. creates the Python virtual environment (venv) and installs PyTorch + requirements.txt
     (CUDA build if an NVIDIA GPU is found, CPU build otherwise)
  3. downloads the data + registered model and the database snapshot from the GitHub releases,
     checks their SHA-256, and unpacks them into the project folder
  4. installs a portable PostgreSQL 16 in .\postgres (no admin rights, no Windows service),
     starts it on port 5433 and restores the database snapshot
  5. runs verify_setup.py, which checks everything and prints today's top-20 picks

Options:
  -CpuOnly        install the CPU build of PyTorch even if an NVIDIA GPU is present
  -AllModels      also download every trained TFT checkpoint and training log (1.3 GB, optional)
  -Port <n>       PostgreSQL port for a new install (default 5433; ignored if .env already exists)
  -PostgresZip    path to an already-downloaded postgresql-16.10-1-windows-x64-binaries.zip
  -SkipPython     do not create the venv or install packages
  -SkipData       do not download or unpack data and models
  -SkipDatabase   do not install, start or restore PostgreSQL
  -SkipVerify     do not run verify_setup.py at the end

Needs: Windows 10/11 x64, Python 3.10 (3.11/3.12 also work), internet, about 6 GB of free disk.
#>
param(
    [switch]$CpuOnly,
    [switch]$AllModels,
    [int]$Port = 5433,
    [string]$PostgresZip = "",
    [switch]$SkipPython,
    [switch]$SkipData,
    [switch]$SkipDatabase,
    [switch]$SkipVerify
)

$ErrorActionPreference = 'Continue'      # native tools write progress to stderr; exit codes are checked instead
$ProgressPreference = 'SilentlyContinue' # Invoke-WebRequest is very slow with its progress bar in PowerShell 5.1
[Net.ServicePointManager]::SecurityProtocol = [Net.SecurityProtocolType]::Tls12

$Root = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $Root
$Repo = 'UzairManjre/stock_predict'
$Downloads = Join-Path $Root 'downloads'
$VenvPy = Join-Path $Root 'venv\Scripts\python.exe'

# GitHub release assets (see the Releases page of the repository)
$DataAsset   = @{ Tag = 'data-2026-11-01';   File = 'stock_predict_data_2026-11-01.zip';   Sha = 'a9471703b96d6248b909eb692ec6d3236a929a17ed535ba11c17b595c15cb8a5'; Size = '255 MB' }
$DbAsset     = @{ Tag = 'db-2026-11-01';     File = 'stock_predict_2026-11-01.dump';       Sha = 'd91f77c86066b7f6383890fc95243d019d0c9b927833dd5ff21c0142dc1c385c'; Size = '8 MB' }
$ModelsAsset = @{ Tag = 'models-2026-11-02'; File = 'stock_predict_models_2026-11-02.zip'; Sha = '9534008d307a68758bd5c90d7b128497608f40a370cc3162566d293237c05f14'; Size = '1.3 GB' }

# Official PostgreSQL 16 binaries for Windows (EnterpriseDB), the same build the project was developed on
$PgFile = 'postgresql-16.10-1-windows-x64-binaries.zip'
$PgUrl  = "https://get.enterprisedb.com/postgresql/$PgFile"
$PgSha  = 'ebb3b6af4fa69dea9951b66855bc4d42dc04e56ccb9aa7024ce3c58bd89d6b0c'

function Step($n, $text) { Write-Host ""; Write-Host "[$n/5] $text" -ForegroundColor Cyan }
function Ok($text)   { Write-Host "  ok    $text" -ForegroundColor Green }
function Info($text) { Write-Host "        $text" }
function Warn($text) { Write-Host "  note  $text" -ForegroundColor Yellow }
function Fail($text) {
    Write-Host ""
    Write-Host "  ERROR  $text" -ForegroundColor Red
    Write-Host "  Fix the problem above and run setup again; finished steps are skipped." -ForegroundColor Red
    exit 1
}

function Test-Sha($path, $sha) {
    (Test-Path $path) -and ((Get-FileHash -Algorithm SHA256 -Path $path).Hash -eq $sha)
}

function Get-Download($url, $dest, $sha, $label) {
    if (Test-Sha $dest $sha) { Ok "$label already downloaded"; return }
    Info "downloading $label ..."
    $part = "$dest.part"
    if (Test-Path $part) { Remove-Item $part -Force }
    $curl = Get-Command curl.exe -ErrorAction SilentlyContinue
    if ($curl) {
        & $curl.Source -L --fail --retry 3 --retry-delay 5 --progress-bar -o $part $url
        if ($LASTEXITCODE -ne 0) { Fail "download failed (curl exit code $LASTEXITCODE): $url" }
    } else {
        try { Invoke-WebRequest -Uri $url -OutFile $part -UseBasicParsing } catch { Fail "download failed: $url ($($_.Exception.Message))" }
    }
    if (-not (Test-Sha $part $sha)) { Remove-Item $part -Force; Fail "$label is corrupt (SHA-256 mismatch); run setup again to retry" }
    Move-Item -Force $part $dest
    Ok "$label downloaded and verified"
}

Add-Type -AssemblyName System.IO.Compression.FileSystem
function Expand-Zip($zip, $dest, [string[]]$prefixes = @()) {
    # Unpacks only files that do not exist yet, so running setup again never overwrites data the
    # daily pipeline has updated since. $prefixes limits extraction to some folders of the archive.
    $added = 0; $kept = 0
    $archive = [IO.Compression.ZipFile]::OpenRead($zip)
    try {
        foreach ($entry in $archive.Entries) {
            $name = $entry.FullName
            if ($name.EndsWith('/')) { continue }
            $wanted = $prefixes.Count -eq 0
            foreach ($prefix in $prefixes) { if ($name.StartsWith($prefix)) { $wanted = $true; break } }
            if (-not $wanted) { continue }
            $target = Join-Path $dest ($name -replace '/', '\')
            if (Test-Path -LiteralPath $target) { $kept++; continue }
            New-Item -ItemType Directory -Force -Path (Split-Path -Parent $target) | Out-Null
            [IO.Compression.ZipFileExtensions]::ExtractToFile($entry, $target)
            $added++
        }
    } catch {
        Fail "could not unpack $zip ($($_.Exception.Message))"
    } finally {
        $archive.Dispose()
    }
    if ($added -gt 0 -and $kept -gt 0) { Info "$kept files were already present and were left as they are" }
    return $added
}

function Read-EnvFile($path) {
    $cfg = @{ DB_HOST = 'localhost'; DB_PORT = "$Port"; DB_USER = 'postgres'; DB_PASSWORD = 'postgres'; DB_NAME = 'stock_predict' }
    if (Test-Path $path) {
        foreach ($line in Get-Content $path) {
            $t = $line.Trim()
            if ($t -and -not $t.StartsWith('#') -and $t.Contains('=')) {
                $i = $t.IndexOf('=')
                $cfg[$t.Substring(0, $i).Trim()] = $t.Substring($i + 1).Trim()
            }
        }
    }
    return $cfg
}

function Find-Python {
    $py = Get-Command py.exe -ErrorAction SilentlyContinue
    if ($py) {
        foreach ($v in '3.10', '3.11', '3.12') {
            $exe = & $py.Source "-$v" -c "import sys; print(sys.executable)" 2>$null
            if ($LASTEXITCODE -eq 0 -and $exe) { return "$exe".Trim() }
        }
    }
    foreach ($name in 'python.exe', 'python3.exe') {
        $cmd = Get-Command $name -ErrorAction SilentlyContinue
        if ($cmd -and $cmd.Source -notlike '*WindowsApps*') {
            $ver = & $cmd.Source -c "import sys; print('%d.%d' % sys.version_info[:2])" 2>$null
            if ($LASTEXITCODE -eq 0 -and ('3.10', '3.11', '3.12' -contains "$ver".Trim())) { return $cmd.Source }
        }
    }
    return $null
}

Write-Host "Setting up stock_predict (AlphaEdge) in $Root" -ForegroundColor White
if (-not [Environment]::Is64BitOperatingSystem) { Fail "a 64-bit Windows is required" }
# Some Python packages contain files 175 characters deep, and Windows stops at 260 characters
# unless long-path support is turned on, so a deeply nested project folder makes pip fail.
$longPaths = (Get-ItemProperty 'HKLM:\SYSTEM\CurrentControlSet\Control\FileSystem' -Name LongPathsEnabled -ErrorAction SilentlyContinue).LongPathsEnabled -eq 1
if ($Root.Length -gt 80 -and -not $longPaths) {
    Fail ("the project folder path is $($Root.Length) characters long, and Windows long-path support is off, so some`n" +
          "         Python packages cannot install. Move the folder somewhere short, for example C:\stock_predict, and run`n" +
          "         setup again (or turn on long paths: https://pip.pypa.io/warnings/enable-long-paths).")
}
New-Item -ItemType Directory -Force -Path $Downloads | Out-Null

# ---------------------------------------------------------------------------------------------
Step 1 "Database settings (.env)"
$EnvPath = Join-Path $Root '.env'
if (Test-Path $EnvPath) {
    Ok ".env already exists; using its settings"
} else {
    $template = Join-Path $Root '.env.example'
    if (-not (Test-Path $template)) { Fail ".env.example is missing; is this a complete copy of the repository?" }
    (Get-Content $template) -replace '^DB_PORT=.*', "DB_PORT=$Port" | Set-Content -Path $EnvPath -Encoding ascii
    Ok "created .env (PostgreSQL on localhost:$Port, user postgres)"
}
$Cfg = Read-EnvFile $EnvPath

# ---------------------------------------------------------------------------------------------
Step 2 "Python environment"
if ($SkipPython) {
    Warn "skipped (-SkipPython)"
} else {
    if (-not (Test-Path $VenvPy)) {
        $python = Find-Python
        if (-not $python) {
            Fail ("Python 3.10, 3.11 or 3.12 was not found. Install Python 3.10 (tick 'Add python.exe to PATH'), then run setup again:`n" +
                  "         winget install -e --id Python.Python.3.10`n         or download it from https://www.python.org/downloads/release/python-31011/")
        }
        Info "creating the virtual environment with $python"
        & $python -m venv (Join-Path $Root 'venv')
        if ($LASTEXITCODE -ne 0 -or -not (Test-Path $VenvPy)) { Fail "could not create the virtual environment" }
    }
    $ver = (& $VenvPy -c "import sys; print('%d.%d.%d' % sys.version_info[:3])").Trim()
    Ok "virtual environment: Python $ver"
    if (-not $ver.StartsWith('3.10.')) { Warn "the project was tested on Python 3.10; $ver should work but is untested" }

    # Remember a finished install, so running setup again skips pip unless requirements.txt changed
    $reqHash = (Get-FileHash -Algorithm SHA256 -Path (Join-Path $Root 'requirements.txt')).Hash
    $installed = Join-Path $Root 'venv\.stock_predict_installed'
    if ((Test-Path $installed) -and ((Get-Content $installed -Raw).Trim() -eq $reqHash)) {
        Ok "Python packages already installed (requirements.txt unchanged)"
    } else {
        & $VenvPy -m pip install --disable-pip-version-check --quiet --upgrade pip
        $gpu = (-not $CpuOnly) -and [bool](Get-Command nvidia-smi.exe -ErrorAction SilentlyContinue)
        if ($gpu) { $index = 'https://download.pytorch.org/whl/cu121'; $build = 'CUDA 12.1 (NVIDIA GPU found)' }
        else      { $index = 'https://download.pytorch.org/whl/cpu';   $build = 'CPU' }
        Info "installing PyTorch 2.5.1, $build build, unless PyTorch 2.5.1 is already installed (CUDA build 2.4 GB, CPU build 200 MB) ..."
        & $VenvPy -m pip install --disable-pip-version-check torch==2.5.1 --index-url $index
        if ($LASTEXITCODE -ne 0) { Fail "PyTorch install failed" }
        Info "installing the other packages from requirements.txt ..."
        & $VenvPy -m pip install --disable-pip-version-check -r (Join-Path $Root 'requirements.txt')
        if ($LASTEXITCODE -ne 0) { Fail "package install failed (see the pip messages above)" }
        Set-Content -Path $installed -Value $reqHash -Encoding ascii
        Ok "Python packages installed"
    }
}

# ---------------------------------------------------------------------------------------------
Step 3 "Data and models (GitHub releases)"
$DumpPath = Join-Path $Downloads $DbAsset.File
if ($SkipData) {
    Warn "skipped (-SkipData)"
} else {
    $zips = @($DataAsset)
    if ($AllModels) { $zips += $ModelsAsset }
    foreach ($a in @($DbAsset) + $zips) {
        Get-Download "https://github.com/$Repo/releases/download/$($a.Tag)/$($a.File)" (Join-Path $Downloads $a.File) $a.Sha "$($a.File) ($($a.Size))"
    }
    foreach ($a in $zips) {
        Info "unpacking $($a.File) (only files that are not there yet) ..."
        $added = Expand-Zip (Join-Path $Downloads $a.File) $Root
        if ($added -gt 0) { Ok "$($a.File) unpacked into the project folder ($added files)" }
        else              { Ok "$($a.File) already unpacked" }
    }
}

# ---------------------------------------------------------------------------------------------
Step 4 "PostgreSQL database"
if ($SkipDatabase) {
    Warn "skipped (-SkipDatabase)"
} else {
    # Portable binaries: .\postgres (new installs) or ..\postgres (the original PC)
    $PgRoot = $null
    foreach ($c in @((Join-Path $Root 'postgres'), (Join-Path (Split-Path $Root -Parent) 'postgres'))) {
        if (Test-Path (Join-Path $c 'pgsql\bin\pg_ctl.exe')) { $PgRoot = (Resolve-Path $c).Path; break }
    }
    if ($PgRoot) {
        Ok "PostgreSQL binaries found in $PgRoot"
    } else {
        $PgRoot = Join-Path $Root 'postgres'
        if ($PostgresZip) {
            if (-not (Test-Sha $PostgresZip $PgSha)) { Fail "$PostgresZip is not the expected $PgFile (SHA-256 mismatch)" }
            $zip = $PostgresZip
        } else {
            $zip = Join-Path $Downloads $PgFile
            Get-Download $PgUrl $zip $PgSha "PostgreSQL 16.10 for Windows (322 MB)"
        }
        Info "unpacking PostgreSQL (server and tools only, about 120 MB) ..."
        New-Item -ItemType Directory -Force -Path $PgRoot | Out-Null
        $null = Expand-Zip $zip $PgRoot @('pgsql/bin/', 'pgsql/lib/', 'pgsql/share/')
        if (-not (Test-Path (Join-Path $PgRoot 'pgsql\bin\pg_ctl.exe'))) { Fail "PostgreSQL did not unpack correctly into $PgRoot" }
        Ok "PostgreSQL 16.10 installed in $PgRoot"
    }
    $PgBin = Join-Path $PgRoot 'pgsql\bin'
    $PgData = Join-Path $PgRoot 'data'
    $h = $Cfg.DB_HOST; $p = $Cfg.DB_PORT; $u = $Cfg.DB_USER; $db = $Cfg.DB_NAME
    $env:PGPASSWORD = $Cfg.DB_PASSWORD

    & (Join-Path $PgBin 'pg_isready.exe') -h $h -p $p -q
    if ($LASTEXITCODE -eq 0) {
        Ok "a PostgreSQL server is already running on ${h}:$p"
    } else {
        if (-not (Test-Path (Join-Path $PgData 'PG_VERSION'))) {
            Info "creating the database cluster in $PgData ..."
            $pw = Join-Path $env:TEMP "stock_predict_pw_$PID.txt"
            Set-Content -Path $pw -Value $Cfg.DB_PASSWORD -Encoding ascii -NoNewline
            & (Join-Path $PgBin 'initdb.exe') -D $PgData -U $u "--pwfile=$pw" -E UTF8 --locale=C --auth=scram-sha-256
            $code = $LASTEXITCODE
            Remove-Item $pw -Force
            if ($code -ne 0) { Fail "initdb failed (exit code $code)" }
            Add-Content -Path (Join-Path $PgData 'postgresql.conf') -Encoding ascii -Value "`r`n# stock_predict`r`nport = $p`r`nlisten_addresses = 'localhost'`r`n"
            Ok "database cluster created (port $p, user $u)"
        }
        Info "starting PostgreSQL ..."
        & cmd /c "`"$(Join-Path $Root 'db_start.bat')`""
        $ready = $false
        for ($i = 0; $i -lt 30 -and -not $ready; $i++) {
            & (Join-Path $PgBin 'pg_isready.exe') -h $h -p $p -q
            if ($LASTEXITCODE -eq 0) { $ready = $true } else { Start-Sleep -Seconds 2 }
        }
        if (-not $ready) { Fail "PostgreSQL did not start on port $p; see $(Join-Path $PgRoot 'postgres.log')" }
        Ok "PostgreSQL is running on ${h}:$p"
    }

    $psql = Join-Path $PgBin 'psql.exe'
    $exists = "$(& $psql -h $h -p $p -U $u -d postgres -tAc "SELECT 1 FROM pg_database WHERE datname = '$db'")".Trim()
    if ($LASTEXITCODE -ne 0) { Fail "cannot log in to PostgreSQL as $u on ${h}:$p; check DB_USER / DB_PASSWORD in .env" }
    $rows = 0
    if ($exists -eq '1') {
        $hasTable = "$(& $psql -h $h -p $p -U $u -d $db -tAc "SELECT to_regclass('public.market_data_daily') IS NOT NULL")".Trim()
        if ($hasTable -eq 't') { $rows = [int64]("$(& $psql -h $h -p $p -U $u -d $db -tAc 'SELECT count(*) FROM market_data_daily')".Trim()) }
    } else {
        & (Join-Path $PgBin 'createdb.exe') -h $h -p $p -U $u -E UTF8 -T template0 $db
        if ($LASTEXITCODE -ne 0) { Fail "could not create the database $db" }
        Ok "database $db created"
    }
    if ($rows -gt 0) {
        Ok "database $db already holds $rows price rows; not restoring over it"
    } else {
        if (-not (Test-Path $DumpPath)) { Fail "database snapshot not found ($DumpPath); run setup without -SkipData" }
        Info "restoring the database snapshot (1 Nov 2026) ..."
        & (Join-Path $PgBin 'pg_restore.exe') -h $h -p $p -U $u -d $db --no-owner --no-privileges -j 4 $DumpPath
        if ($LASTEXITCODE -ne 0) { Fail "pg_restore reported errors (exit code $LASTEXITCODE)" }
        Ok "database restored"
    }
    Remove-Item Env:\PGPASSWORD -ErrorAction SilentlyContinue
}

# ---------------------------------------------------------------------------------------------
Step 5 "Checking the setup (verify_setup.py)"
if ($SkipVerify -or -not (Test-Path $VenvPy)) {
    Warn "skipped"
    exit 0
}
& $VenvPy (Join-Path $Root 'verify_setup.py')
if ($LASTEXITCODE -ne 0) { Fail "some checks failed (see above)" }

Write-Host ""
Write-Host "Setup complete." -ForegroundColor Green
Write-Host "  Dashboard:            start_dashboard.bat   (opens http://localhost:8501)"
Write-Host "  Today's picks:        venv\Scripts\python.exe Cleaner\final_model.py --score"
Write-Host "  Daily update by hand: venv\Scripts\python.exe run_live_tracking.py"
Write-Host "  Daily at 18:30:       powershell -ExecutionPolicy Bypass -File setup_live_tracking.ps1"
Write-Host "  After a reboot, start the database with db_start.bat (start_dashboard.bat does it for you)."
