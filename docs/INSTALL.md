# 설치 가이드 — 폴더 배치, MuJoCo 버전, 환경별 설치

## 0. 최종 폴더 배치 (이 모양이 되면 끝)

```
<작업루트>/                        예: ~/workspace/
├── h2_sonic_teleop/              ← 이 레포 (clone)
│   ├── h2_policy_server/         ← 정책서버 (동봉)
│   ├── mujoco_support/           ← 러너+씬생성기 (동봉; 여기서 바로 실행)
│   │   ├── sim/h2_vr_teleop_sim.py, h2_edu_sim.py, teleop_stand_fixture.json
│   │   ├── tools/make_scene.py
│   │   └── assets/h2_edu_scene.xml   ← 1-3단계에서 "생성"됨 (레포엔 없음)
│   └── ...
├── wbc_h12/                      ← setup.sh가 clone (h2_edu, e279627b, lfs)
└── GR00T-WholeBodyControl/       ← setup.sh가 clone + overlay (PICO 쓸 때만 필요)
```

> 정책서버의 `run.sh`(도커 모드)는 wbc_h12가 **형제 폴더**(`../wbc_h12`)에 있다고
> 가정한다. 위 배치 기준으론 `<작업루트>/wbc_h12`가 아니라
> `h2_sonic_teleop/../wbc_h12` = 동일하므로 그대로 맞음.

## 1. 환경 A — 시뮬 러너 + 정책서버 (필수, PICO 불필요)

**버전 (검증된 조합):** Python 3.10~3.11 / **MuJoCo 3.3.7** / numpy 1.26.4 /
pyzmq 27.2.0 / onnxruntime 1.20.1

```bash
cd h2_sonic_teleop
python3 -m venv .venv_sim && source .venv_sim/bin/activate
pip install -r requirements-sim.txt
```

- MuJoCo는 pip 패키지가 전부다 (별도 엔진 설치·라이선스 없음). 뷰어(`--viewer`)는
  데스크톱 세션(X/Wayland) 필요; 헤드리스 서버면 `MUJOCO_GL=egl`(GPU) 또는 `osmesa`.
- MuJoCo 3.x면 대개 동작하나, **3.3.7이 검증 버전**이다. 3.4+에서 API 경고가 나오면
  3.3.7로 핀 고정할 것.

## 2. 환경 B — 정책서버 실행 방식 2택

**(권장) 도커:** 별도 설치 없이
```bash
cd h2_sonic_teleop/h2_policy_server && ./run.sh up     # ../wbc_h12 자동 탐지
```

**bare (도커 없이, 환경 A venv 재사용):**
```bash
source .venv_sim/bin/activate
PYTHONPATH=h2_policy_server python h2_policy_server/server/h2_policy_server.py \
    --wbc-root ../wbc_h12
```

## 3. 시뮬 테스트 실행 (수동 3단계)

```bash
source .venv_sim/bin/activate
export PYTHONPATH=$PWD/h2_policy_server     # 러너가 h2_policy_if를 import

# ① 씬 생성 (최초 1회 / wbc 갱신 시. 절대경로가 박히므로 "그 컴퓨터에서" 생성)
python mujoco_support/tools/make_scene.py --wbc-root ../wbc_h12 \
    --out mujoco_support/assets/h2_edu_scene.xml

# ② 정책서버 (위 2절 중 하나, 별도 터미널)

# ③ 러너 — PICO 없이 파이프라인 검증 (고정 타깃 제자리서기)
python mujoco_support/sim/h2_vr_teleop_sim.py \
    --fixture mujoco_support/sim/teleop_stand_fixture.json \
    --autostart --seconds 15 --viewer
#   성공 기준: 15초간 서서 균형(STANDING), falls=0, pelvis_z≈0.99
#   헤드리스면 --viewer 빼고 --out /tmp/r.json 으로 결과 확인
```

PICO 라이브까지 가려면 README의 "실행 (시뮬 teleop)" 절 참고
(환경 C: GR00T `.venv_teleop` — Python 3.10, torch 2.14+cu, mujoco 3.14(뷰어/모델용),
pinocchio 2.7.0, scipy, xrobotoolkit_sdk + XRoboToolkit PC Service 별도 설치).

## 4. 자주 걸리는 것

| 증상 | 원인/해결 |
|---|---|
| 러너가 `h2_policy_if` import 실패 | `PYTHONPATH=$PWD/h2_policy_server` 누락 |
| 씬 로드 시 STL 못 찾음 | 씬을 다른 컴퓨터에서 생성해 절대경로가 틀림 → ①을 그 컴퓨터에서 재실행 |
| STL이 수백 바이트 | wbc_h12에서 `git lfs pull` 안 함 |
| 서버가 기동 거부(상수 불일치) | wbc_h12가 지정 커밋(e279627b)이 아님 |
| 뷰어 창 안 뜸(헤드리스) | `--viewer` 제거, 결과는 `--out` JSON으로 |
