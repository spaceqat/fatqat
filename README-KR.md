# FatQat

[English](README.md) | [简体中文](README.zh-CN.md) | 한국어

FatQat은 공통된 작성 인터페이스인 `Program`을 중심으로 설계된 양자 컴퓨팅 툴킷입니다.
FatQat은 프로그램을 한 번 작성한 뒤, 이를 다양한 하드웨어 또는 물리적 추상화 수준에서 모델링할 수 있습니다.

> FatQat은 현재 활발히 개발 중임에 유의하여 주십시오. 릴리스 간에 인터페이스가 변경될 수 있습니다.
> 재현성이 중요하다면 사용할 버전을 정확히 지정하여 주십시오.

| 추상화 수준 | 시뮬레이션 목적 |
| --- | --- |
| 일반 시뮬레이션 | 논리 상태, 샘플, 관측량, 노이즈, 매개변수 스윕 분석 |
| 하드웨어 프로파일 시뮬레이션 | 네이티브 연산 및 큐비트의 배치, 연결성, 수용량, 또는 점유율 |
| 해밀토니안 에뮬레이션 | 펄스, 커플링, 누설, 타이밍, 연속 시간 잡음 추적 |

각 실행 대상은 동일한 `Program` 유형으로써 입력받고, 공통된 `Job`/`Result` 흐름으로 결과를 반환합니다.
각 대상은 자신이 물리적·수학적으로 구현할 수 있는 연산인지 개별적으로 검증합니다.

## [AI가 코드를 대신 작성하는 시대에서, 인간은 어디에 가장 크게 기여할 수 있을까요?] (https://fatqat.readthedocs.io/en/latest/contributing/contributing-in-the-ai-era/)

**AI는 코드 작성을 쉽게 해 줍니다. 여러분의 질문, 판단, 검토는 그 코드를 가치 있게 만들 것입니다.**

유용한 질문을 던지고, 겪은 문제를 공유하고, 설계를 검토하고, 설명을 개선하거나 코드를 기여해 주십시오.
**[기여 가이드 읽기 →](https://fatqat.readthedocs.io/en/latest/contributing/contributing-in-the-ai-era/)**

## 설치

FatQat은 Python 3.12 이상을 요구합니다. 다음 명령으로 설치하십시오.

```sh
python -m pip install fatqat
```

개발 버전을 사용하려면 저장소를 복제한 뒤 소스에서 설치하세요.

```sh
git clone https://github.com/spaceqat/fatqat.git
cd fatqat
python -m pip install .
```

## 첫 번째 Program 실행하기

다음 벨 상태 예제는 회로 작성부터 실행 결과 확인까지의 전체 과정을 보여 줍니다.

```python
import fatqat as fq
import fatqat.operations as ops

program = fq.Program(2, 2)
program.add(ops.H, 0)
program.add(ops.CX, (0, 1))
program.measure_all()

result = fq.simulator.Simulator().run(
    program,
    shots=1000,
    simulation_config={"seed": 7},
).result()

print(result.get_counts())
```

측정 결과에는 `00`과 `11`만 나타납니다. 두 큐비트가 이 벨 상태로 얽혀 있어 측정된 두 비트가 서로 항상 같기 때문입니다.
[빠른 시작 가이드](https://fatqat.readthedocs.io/en/latest/guide/quickstart/)에서는 이 `Program`을 회로도로 그린 뒤 실행하고, 측정 빈도를 그래프로 표시하는 방법을 소개합니다.

## 같은 작성 모델로 기능을 더욱 확장하십시오.

`Program`은 레지스터와 순서가 있는 명령어를 기록합니다. 게이트, 측정, 초기화, 고전적 조건,
재사용 가능한 매개변수, 논리 큐디트(qudit, 다준위 양자 시스템), 서로 다른 국소 차원이 혼합된 시스템,
회로 시각화, 직접적인 물리 제어를 지원합니다. 회로와 펄스에 별도의 언어를 사용하는 대신,
이 기능들을 하나의 모델에서 함께 다룹니다.

[Program 가이드](https://fatqat.readthedocs.io/en/latest/guide/program/)는 이러한 개념을 단계별로 소개합니다.
[물리 모델링 수준 선택하기](https://fatqat.readthedocs.io/en/latest/guide/execution-models/)에서는
동일한 회전 연산을 수정 없이 세 가지 실행 수준에서 실행하는 방법을 보여 줍니다.

이어서 다음 문서들에 관심이 있으실 수도 있습니.

- [양자 프로그램을 모사 하십시오.](https://fatqat.readthedocs.io/en/latest/guide/simulation/): 양자 상태, 샘플링, 매개변수 스윕.
- [실행 결과를 해석하십시오.](https://fatqat.readthedocs.io/en/latest/guide/interpret-results/): 측정 빈도, 상태, 맵, 관측량.
- [이상적인 실행과 잡음이 있는 실행을 비교하십시오.](https://fatqat.readthedocs.io/en/latest/guide/ideal-and-noisy/).
- [성능과 확장성을 측정하십시오](https://fatqat.readthedocs.io/en/latest/guide/performance/).
- [하드웨어 프로파일을 테스트 해보십시오.](https://fatqat.readthedocs.io/en/latest/guide/hardware-profile-simulation/).
- [프로그램으로부터 물리적 동역학까지,](https://fatqat.readthedocs.io/en/latest/guide/hamiltonian-emulation/): 이후 트랜스몬 또는 중성 원자 실행 흐름으로 확장할 수 있습니다.
- [FatQat을 OpenQASM 및 Qiskit과 연동해 보십시오.](https://fatqat.readthedocs.io/en/latest/guide/interoperability/).

[튜토리얼 모음](https://fatqat.readthedocs.io/en/latest/tutorials/)은 알고리즘과 물리 모델을 다루는 더 상세한 사례를 제공합니다.
[API 레퍼런스](https://fatqat.readthedocs.io/en/latest/api/)에서는 정확한 함수 시그니처, 지원 연산,
데이터 형상(shape), 단위, 입력 검증 규칙을 확인할 수 있습니다.

릴리스별 변경 사항은 [변경 이력](https://github.com/spaceqat/fatqat/blob/main/CHANGELOG.md)을 참고하십시오.
위 문서 사이트의 링크는 현재 영문 문서로 연결됩니다.

## 개발

기본적인 로컬 개발을 위해 소스와 핵심 테스트 의존성을 설치하세요.

```sh
python -m pip install --upgrade pip
python -m pip install --editable . --group dev
python -m pytest
```

기여를 준비할 때는 전체 테스트 및 코드 검사 환경을 설치하세요.

```sh
python -m pip install --editable . --group test-full --group lint
```

`test-full`에는 `dev` 의존성과 선택 사항인 Qiskit 연동 의존성이 포함됩니다.
`lint`는 Black과 Pylint를 추가합니다.

변경 사항을 제출하기 전에 AI 보조 작업에 관한 정책을 포함한
[FatQat 기여 가이드](https://github.com/spaceqat/fatqat/blob/main/CONTRIBUTING.md)를 읽어 주세요.
AI 도구를 사용할 수 있지만, 모든 기여자는 자신이 제출하는 작업을 이해하고 책임져야 하며,
작업과 그에 관한 프로젝트 내 소통을 직접 주도해야 합니다.

문서를 수정할 때는 [버전을 고정한 환경 설정 및 빌드 절차](https://github.com/spaceqat/fatqat/blob/main/docs/mkdocs/README.md)를 따르세요.

주요 저장소 디렉터리는 다음과 같습니다.

- `src/fatqat/` — 패키지 소스 코드.
- `tests/` — 동작 중심의 테스트 모음.
- `docs/mkdocs/` — Material 기반 사용자 가이드, 실행 가능한 튜토리얼, API 레퍼런스.
