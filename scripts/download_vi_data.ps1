$ErrorActionPreference = "Continue"
$outFile = Join-Path $PSScriptRoot "data\vi_sentences.txt"
$dataset = "uitnlp/vietnamese_students_feedback"
$pageSize = 100
$apiBase = "https://datasets-server.huggingface.co/rows"

# Ensure data directory
New-Item -ItemType Directory -Force -Path (Join-Path $PSScriptRoot "data") | Out-Null

$sentences = [System.Collections.Generic.List[string]]::new()
$offset = 0
$total = 99999 # will be updated from API
$maxRetries = 5

while ($offset -lt $total) {
    $url = "${apiBase}?dataset=${dataset}&config=default&split=train&offset=${offset}&length=${pageSize}"
    $success = $false
    
    for ($retry = 0; $retry -lt $maxRetries; $retry++) {
        try {
            $resp = Invoke-RestMethod -Uri $url -Method Get -TimeoutSec 30
            $total = $resp.num_rows_total
            foreach ($row in $resp.rows) {
                $s = $row.row.sentence
                if ($s -and $s.Length -ge 5) {
                    $sentences.Add($s)
                }
            }
            $success = $true
            break
        } catch {
            $wait = [math]::Pow(2, $retry + 1)
            Write-Host "  Retry $($retry+1)/$maxRetries at offset $offset (waiting ${wait}s)..."
            Start-Sleep -Seconds $wait
        }
    }
    
    if (-not $success) {
        Write-Host "FAILED at offset $offset after $maxRetries retries"
        break
    }
    
    $offset += $pageSize
    if ($offset % 1000 -eq 0) {
        Write-Host "  Progress: $offset/$total ($($sentences.Count) sentences)"
    }
    
    # Small delay between requests to avoid rate limiting
    Start-Sleep -Milliseconds 500
}

Write-Host ""
Write-Host "Downloaded $($sentences.Count) sentences"
$sentences | Out-File $outFile -Encoding utf8
$fileSize = (Get-Item $outFile).Length
Write-Host "Saved to $outFile ($([math]::Round($fileSize / 1024, 1)) KB)"
