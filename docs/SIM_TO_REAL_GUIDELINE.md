# Sim → Real 전환 가이드라인

목표: `mujoco_support/sim/h2_vr_teleop_sim.py`(시뮬 러너)를 실물 H2 러너로 개조한다.

## 0. 대원칙 — "몸만 교체"

러너는 두 부분으로 되어 있고, 바꾸는 건 뒷부분뿐이다:

| 러너의 구성 | 시뮬 | 실물 | 전환 작업 |
|---|---|---|---|
| 번역기+루프: `PicoStream`, `build_smpl_tokens`/`build_teleop_tokens`, `ProprioHistory`, 정책서버 왕복, STANDING↔TRACKING 상태기계 | — | — | **그대로 복사 (수정 금지)** — 전부 검증된 코드 |
| 몸-출력: 목표각 → `data.ctrl` + `mj_step` | MuJoCo | DDS `rt/lowcmd` | 교체 |
| 몸-입력: `data.qpos/qvel` + 쿼터니언/각속도 | MuJoCo | DDS `rt/lowstate` | 교체 |
| 리셋: `reset_robot()` (순간이동) | 가능 | **불가능** | 삭제 → 데드맨+호이스트로 대체 |

구체적으로 메인 루프에서 바뀌는 줄은 ~15줄이다:

```python
# [시뮬]                                      # [실물]
q_mj  = data.qpos[7:]                         q_mj  = [ls.motor_state[i].q  for i in MOTOR2MJ]
v_mj  = data.qvel[6:]                         v_mj  = [ls.motor_state[i].dq for i in MOTOR2MJ]
quat  = data.qpos[3:7]                        quat  = ls.imu_state.quaternion      # 순서 확인!
angv  = data.qvel[3:6]  (body frame)          angv  = ls.imu_state.gyroscope       # body frame
...
for _ in range(4):                            lowcmd.motor_cmd[i].q  = target[MJ2MOTOR[i]]
    data.ctrl[:] = kp*(t-q) - kd*dq           lowcmd.motor_cmd[i].kp/kd = KP/KD_MUJOCO[...]
    mujoco.mj_step(model, data)               lowcmd.crc = crc(lowcmd); pub.Write(lowcmd)
                                              # PD는 로봇 보드가 500 Hz로 수행 — tau_ff=0
```

핵심: **PD 계산을 우리가 하지 않는다.** 시뮬에선 τ를 직접 넣었지만, 실물은
목표각+Kp/Kd만 보내고 로봇 메인보드가 500 Hz PD를 돈다. Kp/Kd는 그대로
`h2_joints.KP_MUJOCO`/`KD_MUJOCO` (훈련값 — 바꾸면 정책 전제가 깨진다).

## 1. 선행 확인 (코드 작성 전, 로봇 앞에서)

1. **전원/내장모드 건강검진**: 순정 리모컨으로 damping→기립 확인 (우리 코드와 무관하게 로봇이 멀쩡한지)
2. **low-level 진입**: 매뉴얼의 콤보(H1-2는 L2+R2) 또는 SDK `MotionSwitcherClient.ReleaseMode()`
3. **네트워크**: PC 고정 IP 192.168.123.2/24, 로봇 내부 IP ping
4. **lowstate 수신(읽기 전용)**: 외부 PC에서 멀티캐스트 discovery가 로봇 스위치를 못
   넘는 알려진 문제 → wbc_h12 `origin/deploy`의 `docs/H1_2_REAL_DDS.md` 해법(unicast peer) 적용
5. **★모터 인덱스맵 실측★**: 31관절을 하나씩 손으로 움직여 `lowstate.motor_state[i]`의
   i ↔ 관절이름 표 작성. 이 표가 위 코드의 `MOTOR2MJ`/`MJ2MOTOR`가 된다.
   토큰·정책·Kp/Kd는 전부 **MuJoCo 순서(h2_edu.xml = 다리L/R, 허리3, 머리2, 팔L/R)** 기준 —
   관절 순서 오류는 1초 내 전신 붕괴를 만드는 1순위 버그다 (sim2sim에서 실증됨).
6. IMU 쿼터니언 순서(wxyz/xyzw)와 각속도 프레임(body) 확인 — 정책은 wxyz, body-frame 가정.

## 2. 신규 코드 작성 순서 (각 단계는 이전 단계 통과 후)

| 단계 | 산출물 | 성공 판정 | 로봇에 쓰는 것 |
|---|---|---|---|
| R0 | `h2_lowstate_echo.py` | 손으로 움직인 관절의 각도가 화면에서 변함 | 없음 (읽기만) |
| R1 | `h2_single_joint_wiggle.py` | 손목 1관절만 enable(mode=1, kp≈30), sin 스윙 | 1관절 소토크 |
| R2 | 실물 러너 `h2_vr_teleop_real.py` — 위 "몸 교체" 적용 | 아래 R3~ | 전관절 |
| R3 | **호이스트 공중 + STANDING**: 추종 없이 제자리서기 토큰만 | 다리가 기립 자세를 잡고 유지 | — |
| R4 | 호이스트 살짝 내려 **발 닿은 STANDING** | 균형 유지 (시뮬의 15 s 정립과 동일 거동) | — |
| R5 | g1모드 **모션 재생**: `idle_loop` → `walk_forward_loop` (sample_motions 동봉 pkl) | 클립 완주 | — |
| R6 | **PICO teleop**: teleop 모드(손+가슴) → smpl 모드(전신) | 추종 | — |

R5의 g1모드는 정책서버의 기존 `act()` 경로 그대로라 러너에서 motion 이름만 주면 된다
(시뮬 평가 러너 `h2_edu_sim.py`가 하던 것의 실물판).

## 3. 안전 장치 (R2부터 필수)

전례: wbc_h12 `origin/deploy` 브랜치의 H1-2 체계를 그대로 복제한다.

- **데드맨 클러치** (Xbox 패드 L2): 떼면 즉시 전관절 `kp=0, kd=8`(damping) — 러너는
  계속 돌되 명령만 damping으로. 다시 누르면 **현재 자세에서** 1.5~10 s soft-ramp로 복귀
  (순간 점프 금지 — 시뮬의 'R 리스폰'이 실물에선 이 ramp다)
- **비상정지**: Select/'O' = damping 후 프로세스 종료. 호이스트는 R4까지 항상 체결
- **소프트 리밋**: 송신 직전 목표각을 `h2_joints` 관절한계로 clip + 직전 명령 대비
  변화율 제한(예: 스텝당 0.1 rad) — 토큰/통신 글리치 방어
- **워치독**: lowstate 0.1 s 미수신 또는 PICO 스트림 0.5 s 미수신 → damping
- **시작 상태**: 러너의 STANDING/TRACKING 상태기계 유지하되, 시작은 반드시
  "데드맨 OFF(damping)" — L2를 눌러야 STANDING 진입

## 4. 실물 전 리허설 (로봇 없이)

GR00T의 `gear_sonic/utils/mujoco_sim/unitree_sdk2py_bridge.py` = DDS lowcmd/lowstate를
MuJoCo로 받아주는 **가짜 로봇**. R2에서 만든 실물 러너를 이 브리지에 물리면
"DDS까지 포함한 전체 경로"를 시뮬로 검증할 수 있다 (전임자 H1-2 워크플로와 동일).

## 5. 성능 경계 (sim2sim 캠페인에서 확정된 운용 수칙)

- 시퀀스는 **기립 중립에서 시작** (숙인 자세 참조로 전환하면 진입 낙상)
- 폐루프 지연 **<20 ms** 유지 (정책 왕복 ~6 ms 실측, 여유 있음 — 단 Wi-Fi로 PICO만, 로봇은 유선)
- **waist 3축 토크 실시간 감시** — 모든 낙상의 선행 신호 (포화 27–81%)
- 위험 동작: 깊이 숙이는 리치·테이블급 양손 운반 → 데모에서 제외하거나 보조
- 추가 참고: wbc_h12 `h2_tools/models/README.md`의 "Before the first hardware run" 섹션
  (전임자가 남긴 체크리스트 — 본 가이드와 정합)
