#!/usr/bin/env bash
# 새 컴퓨터에서 teleop 스택 준비: 업스트림을 고정 커밋으로 clone + overlay 적용.
# 사용: bash setup.sh [작업루트=..]   (이 레포 폴더 안에서 실행)
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="${1:-$(dirname "$HERE")}"

GROOT_URL=https://github.com/NVlabs/GR00T-WholeBodyControl.git
GROOT_PIN=b042411fae38ee4d1af9aac82a37a1f8d14d6dd0
WBC_URL=git@github.com:byungokhan/wbc_h12.git
WBC_PIN=e279627ba8a2489bb46dc152a222f8c8e8ecfb27

echo "== 1/3 GR00T-WholeBodyControl =="
if [ ! -d "$ROOT/GR00T-WholeBodyControl" ]; then
    git clone "$GROOT_URL" "$ROOT/GR00T-WholeBodyControl"
fi
cd "$ROOT/GR00T-WholeBodyControl"
git fetch --all && git checkout "$GROOT_PIN"
cp -rv "$HERE/groot_support/gear_sonic/." gear_sonic/
git apply --check "$HERE/groot_support/patches/0001-pico-manager-h2-support.patch" \
    && git apply "$HERE/groot_support/patches/0001-pico-manager-h2-support.patch" \
    || echo "[경고] patch 적용 실패 — 이미 적용됐거나 업스트림 변경. UPSTREAM.md의 수동 반영 절차 참고"

echo "== 2/3 wbc_h12 (무수정, LFS) =="
if [ ! -d "$ROOT/wbc_h12" ]; then
    git clone -b h2_edu "$WBC_URL" "$ROOT/wbc_h12"
fi
cd "$ROOT/wbc_h12" && git checkout "$WBC_PIN" && git lfs pull

echo "== 3/3 안내 =="
cat <<MSG
남은 수동 단계:
  - GR00T venv 구성(.venv_teleop: torch/mujoco/pinocchio/zmq/xrobotoolkit_sdk)
    + XRoboToolkit PC Service 설치
  - 정책서버: cd $HERE/h2_policy_server && ./run.sh up  (WBC_H12_ROOT=$ROOT/wbc_h12)
  - MuJoCo 평가 레포가 있으면 mujoco_support/* 를 그 위에 복사, 씬 생성(./run.sh scene)
실행법은 README.md 참고.
MSG
