#!/usr/bin/env bash
# 시뮬 환경 자동 구성: 공식 오픈소스 MuJoCo(pip, DeepMind 배포)를 고정 버전으로
# 설치하고, wbc_h12로부터 H2 씬을 생성한다. (커스텀 심 없음 — 전부 공식 MuJoCo 위 스크립트)
#   사전: bash setup.sh 완료 (../wbc_h12 존재, lfs pull 됨)
#   사용: bash setup_sim.sh [wbc경로=../wbc_h12]
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WBC="$(cd "${1:-$HERE/../wbc_h12}" && pwd)"

echo "== 1/3 venv + 공식 MuJoCo 설치 (requirements-sim.txt, mujoco==3.3.7) =="
PY="${PYTHON:-python3}"
[ -d "$HERE/.venv_sim" ] || $PY -m venv "$HERE/.venv_sim"
source "$HERE/.venv_sim/bin/activate"
pip install -q -r "$HERE/requirements-sim.txt"
python -c "import mujoco; print('   mujoco', mujoco.__version__, 'OK (official pip)')"

echo "== 2/3 H2 씬 생성 (wbc_h12 → MuJoCo, 이 컴퓨터 절대경로로) =="
PYTHONPATH="$HERE/h2_policy_server" python "$HERE/mujoco_support/tools/make_scene.py" \
    --wbc-root "$WBC" --out "$HERE/mujoco_support/assets/h2_edu_scene.xml"

echo "== 3/3 완료 =="
cat <<MSG
다음부터 실행은 (docs/INSTALL.md 3절):
  source .venv_sim/bin/activate && export PYTHONPATH=\$PWD/h2_policy_server
  # 터미널A 정책서버:  ./h2_policy_server/run.sh up   (또는 bare — INSTALL.md 2절)
  # 터미널B 러너(PICO 없이 검증):
  python mujoco_support/sim/h2_vr_teleop_sim.py \\
      --fixture mujoco_support/sim/teleop_stand_fixture.json --autostart --seconds 15 --viewer
MSG
