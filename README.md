# H2 SONIC Teleop (PICO 4 → Unitree H2)

PICO 4 전신 트래킹으로 Unitree H2를 조종하는 teleoperation 파이프라인.
동결된 SONIC H2 EDU 정책(`model_step_066000`)을 **한 글자도 수정하지 않고**,
정책이 원래 가진 3모드 입구(g1/teleop/smpl) 중 사람 신호를 직접 받는
teleop/smpl 모드를 사용한다. 현재는 MuJoCo 시뮬레이션이 "로봇 몸" 역할을 하며,
실물 적용은 몸 부분만 교체하면 된다 (→ `docs/SIM_TO_REAL_GUIDELINE.md`).

## 구조 (한 장)

```
 사람 (PICO4 + 전신 트래커)
   │  raw 포즈: 24관절 위치+회전 (XRoboToolkit, ~90 Hz)
   ▼
┌ ① 매니저 = 번역기 ──────────────┐  GR00T-WholeBodyControl + groot_support/ 패치
│ raw → SMPL 번역 + 사람↔로봇     │  버튼: A+B+X+Y=켜기+체형보정, A+X=송출(POSE),
│ 체형 보정(calibration)          │        양손 그립=추종 토글
└──────────────┬──────────────────┘
               │ SMPL 참조신호 묶음 (ZMQ 방송 :5556, 50 Hz)
               ▼
┌ ② 러너 = 지휘자 ────────────────┐  mujoco_support/sim/h2_vr_teleop_sim.py
│ ⓐ SMPL → 정책 포맷(1790차원)    │
│ ⓑ 로봇 현재상태 히스토리 부착 ◀──┼──┐
│ ⓒ 정책서버 왕복                 │  │
│ ⓓ 몸 구동 (PD + 물리)           │  │
└──────┬────────────▲─────────────┘  │
       │ 관측        │ 목표관절각 31개 │
       ▼            │ (state 아님, "명령")
┌ ③ 정책서버 = 두뇌 자판기 ────────┐  │  h2_policy_server/
│ 동결 ONNX 실행만 (encode/decode) │  │  두뇌 원본은 창고에서 기동 시 로드
└──────────────────────────────────┘  │
       │ (창고: wbc_h12 — 실행 안 됨, │
       │  ONNX·URDF·상수 공급만)      │
       ▼                              │
┌ 몸 (교체 가능) ──────────────────────┤
│ [시뮬] MuJoCo: PD+중력 mj_step       │ 새 state (관절각·각속도·IMU)
│ [실물] H2: rt/lowcmd → 보드 PD,      │ 가 ②ⓑ로 되돌아감 = 피드백 루프
│        state ← rt/lowstate           │
└──────────────────────────────────────┘
```

핵심 규칙:
- **정책 출력은 "지금 상태"가 아니라 "목표각 명령"** — 실제 움직임은 몸의 PD가 만든다.
- **몸의 state가 매 스텝 되돌아와야 루프가 닫힌다** (그래서 로봇은 밀려도 균형을 잡는다).
- **GMR(리타게터)은 이 라이브 파이프라인에 없다** — 오프라인 모션파일 제작 전용.
  teleop/smpl 모드는 신경망이 리타게팅을 내장 학습한 입구라서 사람 신호 직통.

## 폴더 구성 원칙

업스트림(남의/원본 레포)은 이 레포에 **포함하지 않는다**. 대신 그 위에 얹는
overlay만 보관한다 → 업스트림이 업데이트돼도 overlay 재적용으로 따라간다.

| 폴더 | 성격 | 적용 대상 |
|---|---|---|
| `h2_policy_server/` | **우리 것 통째** (어디서도 못 받음) | — |
| `groot_support/` | overlay: 신규 6파일 + 수정 patch 1개 | NVlabs/GR00T-WholeBodyControl |
| `mujoco_support/` | overlay: teleop 러너 + 도구 | 자체 MuJoCo 평가 레포 (또는 단독 사용) |
| `wbc_h12_support/` | **비어 있음** = 무수정 선언 + clone 지침 | byungokhan/wbc_h12 (h2_edu) |
| `docs/` | sim→real 전환 가이드 | — |

업스트림 버전 고정과 overlay 적용은 `UPSTREAM.md` / `setup.sh` 참고.

## 실행 (시뮬 teleop)

> 처음이면 먼저 **`docs/INSTALL.md`** — 폴더 배치, MuJoCo 버전(3.3.7), 환경별 설치, PICO 없이 돌리는 검증 절차.

사전: XRoboToolkit PC Service 설치, PICO 앱 Status=WORKING, `setup.sh`로 업스트림 준비.

```bash
# 터미널 ① 두뇌 자판기
cd h2_policy_server && ./run.sh up

# 터미널 ② 번역기 (GR00T venv)
cd <GR00T> && source .venv_teleop/bin/activate
python gear_sonic/scripts/pico_manager_thread_server.py --manager --robot h2 --num_frames_to_send 10
#   기준자세(정면 직립)로 서서 A+B+X+Y  → "Calibration completed"
#   A+X                              → "PLANNER -> POSE" (송출 시작)

# 터미널 ③ 지휘자+몸
cd <Mujoco평가레포>   # mujoco_support/sim/* 이 sim/에 들어간 상태
docker compose -f docker/docker-compose.yaml -f docker/docker-compose.local.yaml \
  run --rm --entrypoint "" h2-mujoco \
  python sim/h2_vr_teleop_sim.py --viewer --mode smpl
#   "PICO stream RECEIVING" 확인 → 양손 그립 꽉 (또는 창 클릭 후 S) → TRACKING
#   R = 넘어지면 리스폰, 그립/S = 추종↔제자리서기 토글
```

모드: `--mode smpl` = 전신(발 포함) 모방 / `--mode teleop` = 손+가슴 3점만, 다리는 자동 균형.

## 검증 이력 (이 번들의 신뢰 근거)

- 서버 `act()` ≡ 러너의 encode+proprio+decode: 오차 1.9e-6 (float32 노이즈)
- teleop 모드: 고정 타깃으로 15 s 정립 균형 (낙상 0)
- smpl 모드: SMPL 워킹 클립만으로 4.2 s에 4.16 m 보행 (오프라인 재생 검증)
- 실기: PICO 착용 전신 추종 동작 확인 (STANDING↔TRACKING, 낙상→R 리스폰 포함)
- 선행 sim2sim 캠페인: 두 물리엔진(MuJoCo/PhysX) 교차검증 47/48 판정 일치,
  폐루프 지연 경계 <20 ms (현 정책 왕복 ~6 ms)
