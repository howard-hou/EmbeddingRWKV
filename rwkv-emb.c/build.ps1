$ErrorActionPreference = 'Stop'
Push-Location $PSScriptRoot
try {
    $env:Path = (Join-Path $PSScriptRoot '.tools\w64devkit\bin') + ';' + $env:Path
    $flags = @('-Ofast','-march=native','-mprefer-vector-width=512','-fopenmp','-Wall','-Wextra','-Wpedantic','-Werror')
    & gcc @flags -shared -DRWKV_NO_MAIN rwkv_emb.c -lm -o rwkv_emb.dll
    if ($LASTEXITCODE -ne 0) { throw 'DLL compilation failed' }
    & gcc @flags rwkv_emb.c -lm -o rwkv-emb.exe
    if ($LASTEXITCODE -ne 0) { throw 'CLI compilation failed' }
} finally { Pop-Location }
