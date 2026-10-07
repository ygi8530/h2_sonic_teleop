#!/usr/bin/env bash
# BONES-SEED(SONIC 훈련 SMPL 데이터셋) 다운로드 — 공개 HF: nvidia/GEAR-SONIC
# 7개 tar 파트, 합계 ~30 GB → ../bones_seed/data/smpl_filtered/*.pkl (131,455클립)
# (eval 파이프라인의 fetch_smpl_filtered.sh와 동일 레시피)
#   사용: bash setup_dataset.sh [목적지=../bones_seed]
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DEST="${1:-$HERE/../bones_seed}"
BASE="https://huggingface.co/nvidia/GEAR-SONIC/resolve/main/bones_seed_smpl"
PARTS=(aa ab ac ad ae af ag)

mkdir -p "$DEST/parts"
echo "[fetch] 7 parts (~30 GB) -> $DEST/parts"
for p in "${PARTS[@]}"; do
    f="$DEST/parts/bones_seed_smpl.tar.part_$p"
    if [ -s "$f" ]; then echo "  part_$p 있음 ($(du -h "$f" | cut -f1))"; continue; fi
    echo "  part_$p ..."
    curl -fL --retry 5 --retry-delay 5 -C - -o "$f" "$BASE/bones_seed_smpl.tar.part_$p"
done
echo "[extract] -> $DEST/data"
mkdir -p "$DEST/data"
cat "$DEST"/parts/bones_seed_smpl.tar.part_* | tar xf - -C "$DEST/data"
SMPL="$DEST/data/smpl_filtered"
echo "[ok] $SMPL ($(find "$SMPL" -name '*.pkl' | wc -l) pkl) — parts/ 는 지워도 됨(~30GB 회수)"
