<#
.SYNOPSIS
    Anonymize the current repository for double-blind review.
.DESCRIPTION
    Removes git history, identifier strings, non-ASCII punctuation, and
    local-only artifacts. Re-initializes a clean git repository.
.PARAMETER RepoRoot
    Repository root to scrub. Defaults to the current directory.
.PARAMETER SkipGitInit
    Do not re-initialize the git repository (for dry runs).
#>

[CmdletBinding()]
param(
    [string]$RepoRoot = (Get-Location).Path,
    [switch]$SkipGitInit
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

if (-not [System.IO.Path]::IsPathRooted($RepoRoot)) {
    $RepoRoot = Join-Path (Get-Location).Path $RepoRoot
}
$RepoRoot = [System.IO.Path]::GetFullPath($RepoRoot)

if (-not (Test-Path -LiteralPath $RepoRoot -PathType Container)) {
    throw "RepoRoot does not exist: $RepoRoot"
}

Push-Location $RepoRoot

Write-Host ""
Write-Host "=================================================================" -ForegroundColor Cyan
Write-Host "  scrub.ps1 - anonymize repository for double-blind review" -ForegroundColor Cyan
Write-Host "=================================================================" -ForegroundColor Cyan
Write-Host ("  repo root: {0}" -f $RepoRoot)
Write-Host ("  git init : {0}" -f (-not $SkipGitInit))
Write-Host ""

# 1. Delete git history
if (Test-Path ".git") {
    Write-Host "[1/7] Removing .git history" -ForegroundColor Cyan
    Remove-Item -Recurse -Force ".git"
} else {
    Write-Host "[1/7] No .git directory (already clean)" -ForegroundColor DarkGray
}

# 2. Delete local-only artifacts
Write-Host "[2/7] Removing local artifacts" -ForegroundColor Cyan
$toDelete = @(
    "results","experiments","artifacts","logs","mlruns","data_cache",
    ".venv","venv","env",".conda",".idea",".vscode"
)
foreach ($d in $toDelete) {
    if (Test-Path -LiteralPath $d) {
        Remove-Item -Recurse -Force -LiteralPath $d
        Write-Host ("      removed: {0}" -f $d)
    }
}

Get-ChildItem -Recurse -Directory -Filter "__pycache__" -ErrorAction SilentlyContinue |
    ForEach-Object {
        Remove-Item -Recurse -Force -LiteralPath $_.FullName
        Write-Host ("      removed: {0}" -f $_.FullName)
    }

Get-ChildItem -File -Filter "*.log" -ErrorAction SilentlyContinue |
    ForEach-Object {
        Remove-Item -Force -LiteralPath $_.FullName
        Write-Host ("      removed: {0}" -f $_.Name)
    }

if (Test-Path -LiteralPath "_copy_manifest.txt") {
    Remove-Item -Force -LiteralPath "_copy_manifest.txt"
    Write-Host "      removed: _copy_manifest.txt"
}

# 3. Normalize non-ASCII punctuation (ASCII-only replacements)
Write-Host "[3/7] Normalizing non-ASCII punctuation" -ForegroundColor Cyan

$codeExts = @("*.py","*.yaml","*.yml","*.md","*.txt","*.toml","*.cfg","*.ps1")

$fixes = [ordered]@{
    ([char]0x2014) = '-'    # em dash
    ([char]0x2013) = '-'    # en dash
    ([char]0x2011) = '-'    # non-breaking hyphen
    ([char]0x2012) = '-'    # figure dash
    ([char]0x2015) = '-'    # horizontal bar
    ([char]0x2018) = "'"    # left single quote
    ([char]0x2019) = "'"    # right single quote
    ([char]0x201A) = "'"    # single low-9
    ([char]0x201C) = '"'    # left double quote
    ([char]0x201D) = '"'    # right double quote
    ([char]0x201E) = '"'    # double low-9
    ([char]0x2026) = '...'  # horizontal ellipsis
    ([char]0x00A0) = ' '    # non-breaking space
    ([char]0x202F) = ' '    # narrow no-break space
    ([char]0x2009) = ' '    # thin space
}

$normalized = 0
foreach ($ext in $codeExts) {
    Get-ChildItem -Recurse -File -Filter $ext -ErrorAction SilentlyContinue |
        ForEach-Object {
            $path = $_.FullName
            $content  = [System.IO.File]::ReadAllText($path, [System.Text.Encoding]::UTF8)
            $original = $content
            foreach ($k in $fixes.Keys) {
                $from = [string]$k
                $to   = [string]$fixes[$k]
                $content = $content.Replace($from, $to)
            }
            if ($content -ne $original) {
                [System.IO.File]::WriteAllText(
                    $path, $content,
                    (New-Object System.Text.UTF8Encoding $false))
                Write-Host ("      normalized: {0}" -f $path)
                $normalized++
            }
        }
}
Write-Host ("      files normalized: {0}" -f $normalized)

# 4. Scan for identity leaks
Write-Host "[4/7] Scanning for potential identity leaks" -ForegroundColor Cyan

$pattern = 'author|@[A-Za-z0-9._-]+\.(com|edu|org|io|ai|net)|University|Institute|C:\\Users|/home/|/Users/'
$include = @("*.py","*.yaml","*.yml","*.md","*.txt","*.toml","*.cfg")
$hits = @(
    Get-ChildItem -Recurse -File -Include $include -ErrorAction SilentlyContinue |
        Select-String -Pattern $pattern -CaseSensitive:$false
)

if ($hits.Count -gt 0) {
    Write-Host ""
    Write-Host "      POTENTIAL LEAKS (review manually):" -ForegroundColor Yellow
    foreach ($h in $hits) {
        $rel = $h.Path.Substring($RepoRoot.Length).TrimStart('\','/')
        Write-Host ("      {0}:{1}" -f $rel, $h.LineNumber) -ForegroundColor Yellow
        Write-Host ("          {0}" -f $h.Line.Trim()) -ForegroundColor DarkYellow
    }
    Write-Host ""
    Write-Host ("      total hits: {0}" -f $hits.Count) -ForegroundColor Yellow
    Write-Host "      If any hit contains a real name or email, edit the file" -ForegroundColor Yellow
    Write-Host "      and re-run this script." -ForegroundColor Yellow
} else {
    Write-Host "      No obvious leaks found." -ForegroundColor Green
}

# 5. Strip packaging metadata
Write-Host "[5/7] Stripping metadata from setup.py / pyproject.toml" -ForegroundColor Cyan
$packagingFound = $false
foreach ($f in @("setup.py","pyproject.toml","setup.cfg")) {
    if (Test-Path -LiteralPath $f) {
        $packagingFound = $true
        $resolved = (Resolve-Path -LiteralPath $f).Path
        $text = [System.IO.File]::ReadAllText($resolved)
        $text = [regex]::Replace($text, '(?im)^\s*author\s*=.*$', '')
        $text = [regex]::Replace($text, '(?im)^\s*authors\s*=.*$', '')
        $text = [regex]::Replace($text, '(?im)^\s*maintainer\s*=.*$', '')
        $text = [regex]::Replace($text, '(?im)^\s*url\s*=.*$', '')
        $text = [regex]::Replace($text, '(?im)^\s*homepage\s*=.*$', '')
        $text = [regex]::Replace($text, '(?im)^\s*repository\s*=.*$', '')
        [System.IO.File]::WriteAllText($resolved, $text,
            (New-Object System.Text.UTF8Encoding $false))
        Write-Host ("      scrubbed: {0}" -f $f)
    }
}
if (-not $packagingFound) {
    Write-Host "      no packaging metadata found" -ForegroundColor DarkGray
}

# 6. Check baseline_config.py UTF-8 write
Write-Host "[6/7] Verifying baseline_config.py uses UTF-8 writes" -ForegroundColor Cyan
$bc = Join-Path "config" "baseline_config.py"
if (Test-Path -LiteralPath $bc) {
    $text = [System.IO.File]::ReadAllText((Resolve-Path -LiteralPath $bc).Path)
    if ($text -match "with open\(config_path,\s*'w',\s*encoding='utf-8'\)\s*as f:") {
        Write-Host "      OK: UTF-8 write already in place" -ForegroundColor Green
    } elseif ($text -match "with open\(config_path,\s*'w'\)\s*as f:") {
        Write-Host "      WARNING: baseline_config.py still writes without encoding" -ForegroundColor Yellow
        Write-Host "      Apply: with open(config_path, 'w', encoding='utf-8') as f:" -ForegroundColor Yellow
    } else {
        Write-Host "      pattern not found; inspect manually" -ForegroundColor DarkGray
    }
} else {
    Write-Host "      config\baseline_config.py not found" -ForegroundColor DarkGray
}

# 7. Re-init git
if ($SkipGitInit) {
    Write-Host "[7/7] Skipping git re-initialization" -ForegroundColor DarkGray
} else {
    Write-Host "[7/7] Re-initializing clean git repository" -ForegroundColor Cyan
    git init -b main | Out-Null
    git config user.name  "Anonymous"
    git config user.email "anonymous@example.com"
    git add -A
    git commit -m "Anonymous code release for double-blind review" | Out-Null
    Write-Host "      git repository initialized with anonymous author"
}

Write-Host ""
Write-Host "=================================================================" -ForegroundColor Green
Write-Host "  DONE" -ForegroundColor Green
Write-Host "=================================================================" -ForegroundColor Green
Write-Host ""

if (-not $SkipGitInit) {
    Write-Host "Next steps:" -ForegroundColor Cyan
    Write-Host "  git log --format='%an <%ae> %s'"
    Write-Host "  git remote add origin https://github.com/<throwaway-user>/<throwaway-repo>.git"
    Write-Host "  git push -u origin main"
    Write-Host "  Then anonymize at https://anonymous.4open.science/"
} else {
    Write-Host "Dry run complete. Review output before running without -SkipGitInit." -ForegroundColor Cyan
}

Pop-Location