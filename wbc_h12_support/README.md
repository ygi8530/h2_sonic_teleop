# wbc_h12_support — 비어 있음 (의도된 것)

wbc_h12는 **무수정 원칙**으로 운용한다. 추가 파일도, 패치도 없다.
이 폴더는 그 사실의 선언이자, 혹시 미래에 wbc_h12 위에 얹을 것이
생기면 들어올 자리다.

## 가져오는 법
```bash
git clone -b h2_edu git@github.com:byungokhan/wbc_h12.git
cd wbc_h12 && git checkout e279627ba8a2489bb46dc152a222f8c8e8ecfb27
git lfs pull   # 정책 ONNX가 git-lfs
```

## 이 파이프라인이 wbc_h12에서 읽어가는 것 (전부 읽기 전용)
- `h2_tools/models/model_step_066000_{encoder,decoder}.onnx` — 정책서버가 서빙하는 두뇌
- `gear_sonic/data/assets/robot_description/urdf/h2_edu/` — 매니저 체형보정(FK)용 치수
- `gear_sonic/data/assets/robot_description/mjcf/h2_edu.xml` — 상수 검증, MuJoCo 씬 생성 원본
- `h2_tools/sample_motions/*.pkl` — g1모드 모션재생 데모용 (GMR로 이미 변환돼 동봉됨)
- `external_dependencies/unitree_sdk2_python/` — 실물 단계에서 쓸 공식 SDK (h1_2 low-level 예제 포함)
- (실물 참고) `origin/deploy` 브랜치 — H1-2 실물 배포 전례: C++ 디플로이어, 데드맨 운용,
  DDS 트러블슈팅(`docs/H1_2_REAL_DDS.md`), 단계별 투입 계획(`docs/deploy_plan.md`)
