# 업스트림 레포와 버전 고정 (pins)

이 레포는 아래 업스트림을 **포함하지 않는다** (재배포 금지/불필요).
`setup.sh`가 지정 커밋으로 clone하고 overlay를 얹는다.

| 업스트림 | URL | 고정 커밋 | 브랜치 | overlay |
|---|---|---|---|---|
| GR00T-WholeBodyControl | https://github.com/NVlabs/GR00T-WholeBodyControl | `b042411fae38ee4d1af9aac82a37a1f8d14d6dd0` | main | `groot_support/` |
| wbc_h12 (정책·훈련·창고) | git@github.com:byungokhan/wbc_h12.git | `e279627ba8a2489bb46dc152a222f8c8e8ecfb27` | **h2_edu** | 없음 (무수정) — `wbc_h12_support/README.md` |
| (선택) 자체 MuJoCo 평가 레포 | 사내/개인 보관 | — | — | `mujoco_support/` |

## overlay 적용 규칙

### groot_support → GR00T 클론 위에
```bash
# 1) 신규 파일: 경로 구조 그대로 복사
cp -r groot_support/gear_sonic/ <GR00T>/gear_sonic/
# 주의: gear_sonic/data/robot_model/instantiation/h2.py 는 업스트림 .gitignore(data/)에
#       걸리므로, GR00T 쪽에서 커밋하려면 git add -f 필요 (이 레포가 원본 보관처이므로 보통 불필요)

# 2) 수정 파일: patch 적용
cd <GR00T> && git apply <이레포>/groot_support/patches/0001-pico-manager-h2-support.patch
```
patch 내용: `pico_manager_thread_server.py`에 ① calibration FK를 로봇별 hook으로 추출
(G1 기본 동작 불변) ② `--robot h2` 플래그. 업스트림이 갱신돼 patch가 실패하면
해당 함수(`_capture_calibration`의 FK 호출부, `run_pico_manager`의 ThreePointPose 생성부)만
같은 취지로 손수 반영하면 된다 (총 ~25줄).

### mujoco_support → MuJoCo 평가 레포 위에 (또는 단독)
```bash
cp mujoco_support/sim/*  <Mujoco>/sim/
cp mujoco_support/tools/* <Mujoco>/tools/
```
러너는 같은 폴더의 `h2_edu_sim.py`에서 `reset_robot`/`base_angular_velocity_local`을
import하고, 씬(`assets/h2_edu_scene.xml`)은 `./run.sh scene`으로 wbc_h12에서 생성된다.

### wbc_h12 — 무수정
```bash
git clone -b h2_edu git@github.com:byungokhan/wbc_h12.git && cd wbc_h12
git checkout e279627b && git lfs pull        # ONNX(두뇌)가 LFS
```
런타임이 읽어가는 것: `h2_tools/models/*.onnx`(정책서버), 
`gear_sonic/data/assets/robot_description/urdf/h2_edu/`(매니저 보정),
`mjcf/h2_edu.xml`(상수 검증·씬 생성), `external_dependencies/unitree_sdk2_python`(실물용 SDK).

## 환경 메모
- GR00T venv: `.venv_teleop` (torch, mujoco, pinocchio, zmq, xrobotoolkit_sdk)
- 정책서버/MuJoCo 러너: 각자 docker (`h2_policy_server/run.sh up`, Mujoco `docker-compose`)
- XRoboToolkit PC Service: `/opt/apps/roboticsservice/` (별도 설치)
