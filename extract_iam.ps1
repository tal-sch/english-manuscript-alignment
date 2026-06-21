$ErrorActionPreference = "Stop"

$destination = "C:\Users\Tal Sch\Desktop\english_manuscript_alignment\IAM_Data"
$downloads = "C:\Users\Tal Sch\Downloads"

# Create directories
New-Item -ItemType Directory -Force -Path "$destination\forms"
New-Item -ItemType Directory -Force -Path "$destination\words"
New-Item -ItemType Directory -Force -Path "$destination\xml"
New-Item -ItemType Directory -Force -Path "$destination\ascii"

Write-Host "Extracting formsA-D..."
tar -xzf "$downloads\formsA-D.tgz" -C "$destination\forms"
Write-Host "Extracting formsE-H..."
tar -xzf "$downloads\formsE-H.tgz" -C "$destination\forms"
Write-Host "Extracting formsI-Z..."
tar -xzf "$downloads\formsI-Z.tgz" -C "$destination\forms"

Write-Host "Extracting words..."
tar -xzf "$downloads\words.tgz" -C "$destination\words"

Write-Host "Extracting xml..."
tar -xzf "$downloads\xml.tgz" -C "$destination\xml"

Write-Host "Extracting ascii..."
tar -xzf "$downloads\ascii.tgz" -C "$destination\ascii"

Write-Host "All done!"
