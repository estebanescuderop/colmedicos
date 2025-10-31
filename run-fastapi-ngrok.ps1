# =========================
# run-fastapi-ngrok.ps1 (fix Start-Process args)
# Levanta Uvicorn + ngrok y abre /docs con la URL pública
# =========================

param(
  [string]$ProjectRoot = "$PSScriptRoot",
  [string]$VenvPath    = ".\.venv",
  [string]$AppTarget   = "app.main:app",   # <modulo>:<variable FastAPI>
  [string]$AppDir      = "COLMEDICOS",     # Deja "" si no usas carpeta
  [string]$BindHost    = "0.0.0.0",
  [int]$Port           = 8000,
  [switch]$SkipReadyCheck
)

Write-Host "==> Proyecto: $ProjectRoot"
Set-Location $ProjectRoot

# 1) Activar venv
$activate = Join-Path $VenvPath "Scripts\Activate.ps1"
if (-not (Test-Path $activate)) {
  Write-Error "No se encontró el activador del venv en $activate"
  exit 1
}
. $activate
Write-Host "==> venv activado"

# 2) Verificar uvicorn/ngrok
if (-not (Get-Command uvicorn -ErrorAction SilentlyContinue)) {
  Write-Error "uvicorn no está en PATH. Instala con: pip install uvicorn fastapi"
  exit 1
}
if (-not (Get-Command ngrok -ErrorAction SilentlyContinue)) {
  Write-Error "ngrok no está en PATH. Instálalo y registra tu authtoken."
  exit 1
}

# 3) Iniciar Uvicorn en segundo plano (ventana minimizada)
$uvicornArgs = @("-m","uvicorn")            # base: python -m uvicorn
if ($AppDir -and $AppDir.Trim() -ne "") {
  $uvicornArgs += @("--app-dir", $AppDir)
}
$uvicornArgs += @($AppTarget, "--host", $BindHost, "--port", "$Port", "--reload")

$uvProc = Start-Process -FilePath "python" -ArgumentList $uvicornArgs `
  -WindowStyle Minimized -PassThru
Write-Host "==> Uvicorn PID: $($uvProc.Id)"

# 4) Esperar a que Uvicorn escuche el puerto (máx ~10s) SIN Test-NetConnection
if (-not $SkipReadyCheck) {
  $ready = $false
  for ($i=0; $i -lt 20; $i++) {
    try {
      $client = New-Object System.Net.Sockets.TcpClient
      $iar = $client.BeginConnect("127.0.0.1", $Port, $null, $null)
      $ok = $iar.AsyncWaitHandle.WaitOne(300)
      if ($ok -and $client.Connected) {
        $client.EndConnect($iar); $client.Close()
        $ready = $true; break
      }
      $client.Close()
    } catch {}
    Start-Sleep -Milliseconds 500
  }
  if (-not $ready) {
    Write-Warning "No se confirmó que Uvicorn escuche en :$Port. Continúo con ngrok…"
  }
}

# 5) Lanzar ngrok (ventana minimizada)
$ngArgs = @("http", "$Port")
$ngrokProc = Start-Process -FilePath "ngrok" -ArgumentList $ngArgs `
  -WindowStyle Minimized -PassThru
Write-Host "==> ngrok PID: $($ngrokProc.Id)"

# 6) Consultar la API local de ngrok para obtener la URL pública (máx ~10s)
$publicUrl = $null
for ($i=0; $i -lt 20; $i++) {
  try {
    $tunnels = Invoke-RestMethod -Uri "http://127.0.0.1:4040/api/tunnels" -TimeoutSec 2
    $https = $tunnels.tunnels | Where-Object { $_.public_url -like "https://*" } | Select-Object -First 1
    if ($https) { $publicUrl = $https.public_url; break }
  } catch {}
  Start-Sleep -Milliseconds 500
}
if (-not $publicUrl) {
  Write-Warning "No pude leer la URL pública de ngrok en 127.0.0.1:4040. Ábrela manualmente: http://127.0.0.1:4040"
} else {
  Write-Host "==> URL pública: $publicUrl"
  Start-Process "$publicUrl/docs"
}

Write-Host "`nListo. Deja esta ventana abierta para mantener el túnel. CTRL+C cierra procesos."
