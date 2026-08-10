param(
  [string]$Frontier = "outputs/frontier_qwen_math_1_5b_n24_k4/raw_results.jsonl",
  [string]$Output = "outputs/capability_curriculum_smoke",
  [int]$Step = 0,
  [int]$FadeStart = 0,
  [int]$FadeEnd = 1000
)

$ErrorActionPreference = "Stop"
python frontier_to_curriculum.py --input $Frontier --output-dir $Output --group-size 4 --band-low 0.25 --band-high 0.60 --max-hint-tokens 512
Write-Host "Curriculum manifest written to $Output"
