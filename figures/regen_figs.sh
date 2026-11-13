#!/bin/bash
# Regenerate every paper figure through run_one.py (PNG + PDF, 300 dpi, one folder).
set -u
cd /glade/work/jrotondo/siaf_drk_paper_final/notebooks/revision_20261028 || exit 1

module load conda >/dev/null 2>&1
eval "$(conda shell.bash hook)" >/dev/null 2>&1
conda activate cice-scm-da >/dev/null 2>&1

export TMPDIR=/glade/derecho/scratch/$USER/tmp
mkdir -p "$TMPDIR"

echo "python: $(which python)"
echo "start:  $(date)"
echo

for f in f01 f02 f03 f04 f05 f06 f07 f08 f09 s01 s02 s03 s04 s05 s06 s07 s08 s09 s11; do
    if [ ! -f "${f}.py" ]; then echo "SKIP ${f} (no script)"; continue; fi
    printf '%-6s ... ' "$f"
    if nice -n 10 python run_one.py "$f" > "logs_${f}.txt" 2>&1; then
        echo "ok   -> $(ls -1 figures/${f}*.png 2>/dev/null | tr '\n' ' ')"
    else
        echo "FAIL (see logs_${f}.txt)"
        tail -3 "logs_${f}.txt" | sed 's/^/        /'
    fi
done

echo
echo "done: $(date)"
ls -la figures/*.png | awk '{printf "%-30s %8.2f MB\n", $9, $5/1048576}'
