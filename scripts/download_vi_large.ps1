# Download large Vietnamese dataset from HuggingFace
# Sources: vietnamese_students_feedback, wiki, news, etc.
# Usage: powershell -ExecutionPolicy Bypass -File scripts/download_vi_large.ps1

$OutputFile = "data/vi_sentences.txt"
$TempFile = "data/vi_temp.txt"

# Clear temp
if (Test-Path $TempFile) { Remove-Item $TempFile }

Write-Host "=== Vietnamese Dataset Downloader ===" -ForegroundColor Cyan
Write-Host ""

$totalLines = 0

# ── Source 1: vietnamese_students_feedback ──
Write-Host "[1/4] uitnlp/vietnamese_students_feedback..." -ForegroundColor Yellow
$dataset = "uitnlp/vietnamese_students_feedback"
$field = "sentence"
$batchSize = 100
for ($offset = 0; $offset -lt 20000; $offset += $batchSize) {
    try {
        $url = "https://datasets-server.huggingface.co/rows?dataset=$dataset&config=default&split=train&offset=$offset&length=$batchSize"
        $resp = Invoke-RestMethod -Uri $url -TimeoutSec 30 -ErrorAction Stop
        $rows = $resp.rows
        if ($rows.Count -eq 0) { break }
        foreach ($row in $rows) {
            $text = $row.row.$field
            if ($text -and $text.Length -gt 5) {
                $text = $text.Trim() -replace "`r`n", " " -replace "`n", " "
                Add-Content -Path $TempFile -Value $text -Encoding UTF8
                $totalLines++
            }
        }
        if ($offset % 500 -eq 0) { Write-Host "  offset=$offset lines=$totalLines" }
        Start-Sleep -Milliseconds 800
    } catch {
        if ($_.Exception.Message -match "429") {
            Write-Host "  Rate limited, waiting 10s..." -ForegroundColor Red
            Start-Sleep 10
            $offset -= $batchSize  # retry
        } else {
            Write-Host "  Done at offset=$offset ($($_.Exception.Message))"
            break
        }
    }
}
Write-Host "  Source 1: $totalLines lines" -ForegroundColor Green

# ── Source 2: Vietnamese Wikipedia (bkai) ──
Write-Host "[2/4] bkai-foundation-models/vi-wiki..." -ForegroundColor Yellow
$dataset2 = "linhtran92/viwiki_100k"
$field2 = "text"
$count2 = 0
for ($offset = 0; $offset -lt 50000; $offset += $batchSize) {
    try {
        $url = "https://datasets-server.huggingface.co/rows?dataset=$dataset2&config=default&split=train&offset=$offset&length=$batchSize"
        $resp = Invoke-RestMethod -Uri $url -TimeoutSec 30 -ErrorAction Stop
        $rows = $resp.rows
        if ($rows.Count -eq 0) { break }
        foreach ($row in $rows) {
            $text = $row.row.$field2
            if ($text -and $text.Length -gt 20) {
                # Split long articles into sentences
                $sentences = $text -split '[.!?]\s+' | Where-Object { $_.Length -gt 10 -and $_.Length -lt 500 }
                foreach ($sent in $sentences) {
                    $clean = $sent.Trim() -replace "`r`n", " " -replace "`n", " " -replace "\s+", " "
                    if ($clean.Length -gt 10) {
                        Add-Content -Path $TempFile -Value $clean -Encoding UTF8
                        $totalLines++
                        $count2++
                    }
                }
            }
        }
        if ($offset % 500 -eq 0) { Write-Host "  offset=$offset wiki_lines=$count2 total=$totalLines" }
        Start-Sleep -Milliseconds 800
    } catch {
        if ($_.Exception.Message -match "429") {
            Write-Host "  Rate limited, waiting 10s..." -ForegroundColor Red
            Start-Sleep 10
            $offset -= $batchSize
        } else {
            Write-Host "  Done at offset=$offset ($($_.Exception.Message))"
            break
        }
    }
}
Write-Host "  Source 2: $count2 lines" -ForegroundColor Green

# ── Source 3: Vietnamese news (VietAI) ──
Write-Host "[3/4] VietAI/vi_news..." -ForegroundColor Yellow
$dataset3 = "jetaudio/200k-vi-en"
$field3 = "vi"
$count3 = 0
for ($offset = 0; $offset -lt 50000; $offset += $batchSize) {
    try {
        $url = "https://datasets-server.huggingface.co/rows?dataset=$dataset3&config=default&split=train&offset=$offset&length=$batchSize"
        $resp = Invoke-RestMethod -Uri $url -TimeoutSec 30 -ErrorAction Stop
        $rows = $resp.rows
        if ($rows.Count -eq 0) { break }
        foreach ($row in $rows) {
            $text = $row.row.$field3
            if ($text -and $text.Length -gt 10) {
                $clean = $text.Trim() -replace "`r`n", " " -replace "`n", " " -replace "\s+", " "
                if ($clean.Length -gt 10 -and $clean.Length -lt 500) {
                    Add-Content -Path $TempFile -Value $clean -Encoding UTF8
                    $totalLines++
                    $count3++
                }
            }
        }
        if ($offset % 500 -eq 0) { Write-Host "  offset=$offset news_lines=$count3 total=$totalLines" }
        Start-Sleep -Milliseconds 800
    } catch {
        if ($_.Exception.Message -match "429") {
            Write-Host "  Rate limited, waiting 10s..." -ForegroundColor Red
            Start-Sleep 10
            $offset -= $batchSize
        } else {
            Write-Host "  Done at offset=$offset ($($_.Exception.Message))"
            break
        }
    }
}
Write-Host "  Source 3: $count3 lines" -ForegroundColor Green

# ── Source 4: Vietnamese sentences (tatoeba) ──
Write-Host "[4/4] wiki40b/vi..." -ForegroundColor Yellow
$dataset4 = "wikimedia/wikipedia"
$field4 = "text"
$config4 = "20231101.vi"
$count4 = 0
for ($offset = 0; $offset -lt 30000; $offset += $batchSize) {
    try {
        $url = "https://datasets-server.huggingface.co/rows?dataset=$dataset4&config=$config4&split=train&offset=$offset&length=$batchSize"
        $resp = Invoke-RestMethod -Uri $url -TimeoutSec 30 -ErrorAction Stop
        $rows = $resp.rows
        if ($rows.Count -eq 0) { break }
        foreach ($row in $rows) {
            $text = $row.row.$field4
            if ($text -and $text.Length -gt 20) {
                $sentences = $text -split '[.!?]\s+' | Where-Object { $_.Length -gt 10 -and $_.Length -lt 500 }
                foreach ($sent in $sentences[0..([Math]::Min(5, $sentences.Count - 1))]) {
                    $clean = $sent.Trim() -replace "`r`n", " " -replace "`n", " " -replace "\s+", " "
                    if ($clean.Length -gt 10) {
                        Add-Content -Path $TempFile -Value $clean -Encoding UTF8
                        $totalLines++
                        $count4++
                    }
                }
            }
        }
        if ($offset % 500 -eq 0) { Write-Host "  offset=$offset wiki_lines=$count4 total=$totalLines" }
        Start-Sleep -Milliseconds 800
    } catch {
        if ($_.Exception.Message -match "429") {
            Write-Host "  Rate limited, waiting 15s..." -ForegroundColor Red
            Start-Sleep 15
            $offset -= $batchSize
        } else {
            Write-Host "  Done at offset=$offset ($($_.Exception.Message))"
            break
        }
    }
}
Write-Host "  Source 4: $count4 lines" -ForegroundColor Green

# ── Merge with existing data ──
Write-Host ""
Write-Host "=== Merging ===" -ForegroundColor Cyan

if (Test-Path $TempFile) {
    if (Test-Path $OutputFile) {
        $existing = (Get-Content $OutputFile -Encoding UTF8).Count
        Write-Host "  Existing: $existing lines"
        Get-Content $TempFile -Encoding UTF8 | Add-Content $OutputFile -Encoding UTF8
    } else {
        Copy-Item $TempFile $OutputFile
    }
    Remove-Item $TempFile
}

$finalCount = (Get-Content $OutputFile -Encoding UTF8).Count
$fileSize = [Math]::Round((Get-Item $OutputFile).Length / 1024 / 1024, 2)

Write-Host ""
Write-Host "=== DONE ===" -ForegroundColor Green
Write-Host "  Total lines: $finalCount" -ForegroundColor Cyan
Write-Host "  File size:   $fileSize MB" -ForegroundColor Cyan
Write-Host "  Path:        $OutputFile" -ForegroundColor Cyan
Write-Host ""
Write-Host "To train:" -ForegroundColor Yellow
Write-Host "  cargo run --example train_vietnamese -p vagi-lm --release"
