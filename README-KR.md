<div align="center">

<img src="docs/mkdocs/en/assets/images/space-qat-logo.png" alt="" width="120">

# FatQat

**하나의 Program, 알고리즘에서 하드웨어까지.**

[![Tests](https://github.com/spaceqat/fatqat/actions/workflows/tests.yml/badge.svg?branch=main)](https://github.com/spaceqat/fatqat/actions/workflows/tests.yml)
[![PyPI](https://img.shields.io/pypi/v/fatqat)](https://pypi.org/project/fatqat/)
[![Python](https://img.shields.io/pypi/pyversions/fatqat)](https://pypi.org/project/fatqat/)
[![Documentation](https://img.shields.io/readthedocs/fatqat)](https://fatqat.readthedocs.io/ko/latest/)
[![License](https://img.shields.io/pypi/l/fatqat)](https://github.com/spaceqat/fatqat/blob/main/LICENSE)

[빠른 시작 가이드](https://fatqat.readthedocs.io/ko/latest/guide/quickstart/) · [문서](https://fatqat.readthedocs.io/ko/latest/) · [튜토리얼 모음](https://fatqat.readthedocs.io/ko/latest/tutorials/)

[English](README.md) | [简体中文](README.zh-CN.md) | 한국어

</div>

FatQat은 양자 프로그램을 작성하고, 컴파일하고, 실행하기 위한 Python 툴킷입니다.
`Program`을 한 번 작성하면, 이상적인 시뮬레이터부터 장치의 물리 모델까지 모든 백엔드에서
동일한 인터페이스로 실행하실 수 있습니다.

- **알고리즘 개발자**는 실제 장치에 가까운 조건에서 알고리즘을 검증하고, 하드웨어에서의 실행을
  준비할 수 있습니다.
- **하드웨어 엔지니어**는 제어 신호, 커플링, 노이즈에 대한 장치의 응답을 시뮬레이션하고, 성능을
  제한하는 요인을 찾아낼 수 있습니다.
- **학생**은 단계별 예제와 튜토리얼을 통해 첫 번째 회로부터 실제 장치의 물리까지 양자 컴퓨팅을
  직접 실습하며 배울 수 있습니다.

현재 제공되는 백엔드는 세 가지 추상화 수준을 지원합니다.

| 추상화 수준 | 답할 수 있는 질문 |
| --- | --- |
| 일반 시뮬레이션 | 알고리즘이 의도대로 동작하는가? 노이즈는 결과를 어떻게 바꾸는가? |
| 하드웨어 프로파일 시뮬레이션 | 장치의 네이티브 연산, 연결성, 원자 점유 요건을 만족하는가? |
| 해밀토니안 에뮬레이션 | 펄스가 가해지는 동안 누설, 커플링, 타이밍은 어떻게 나타나는가? |

`Program`은 단순한 게이트 회로 이상입니다. 하나의 `Program`에 다음을 함께 담을 수 있습니다.

- 큐비트, 큐디트(qudit, 다준위 양자 시스템), 광학 모드에 대한 게이트, 회로 중간 측정,
  피드포워드, 매개변수
- 원자 적재, 짝짓기, 손실과 같은 장치 연산
- 보정된 게이트와 함께 사용하는, 직접 작성한 제어 펄스

또한 FatQat은 논리 회로를 초전도 및 중성 원자 장치에 맞게 컴파일하고, OpenQASM 및 Qiskit과
회로를 주고받을 수 있습니다.

## 설치

FatQat은 Python 3.12 이상을 요구합니다. 다음 명령으로 설치하십시오.

```sh
python -m pip install fatqat
```

## 첫 번째 Program 실행하기

```python
import fatqat as fq
import fatqat.operations as ops

program = fq.Program(2, 2)
program.add(ops.H, 0)
program.add(ops.CX, (0, 1))
program.measure_all()

result = fq.simulator.Simulator().run(program, shots=1000).result()
print(result.get_counts())  # '00'과 '11'만 나타납니다
```

## 더 알아보기

- [빠른 시작 가이드](https://fatqat.readthedocs.io/ko/latest/guide/quickstart/):
  첫 번째 Program을 작성하고, 회로도로 그리고, 실행해 보십시오.
- [사용자 가이드](https://fatqat.readthedocs.io/ko/latest/guide/):
  시뮬레이션, 하드웨어 제약, 펄스 동역학을 작업별로 소개합니다.
- [튜토리얼 모음](https://fatqat.readthedocs.io/ko/latest/tutorials/):
  오류 정정, 양자 시뮬레이션, 최적화, 펄스 제어 사례를 제공합니다.
- [API 레퍼런스](https://fatqat.readthedocs.io/ko/latest/api/):
  정확한 함수 시그니처, 단위, 입력 검증 규칙을 확인하실 수 있습니다.

릴리스별 변경 사항은 [변경 이력](https://github.com/spaceqat/fatqat/blob/main/CHANGELOG.md)을 참고해 주십시오.

## 기여하기

AI는 코드 작성을 쉽게 해 줍니다. 여러분의 질문, 사용 사례, 검토는 어떤 코드가 작성할 가치가
있는지 결정하는 데 도움이 됩니다.
[어떻게 기여하실 수 있는지 읽어 보신 뒤](https://fatqat.readthedocs.io/ko/latest/contributing/contributing-in-the-ai-era/),
개발 환경 설정과 AI 보조 작업에 관한 정책은
[FatQat 기여 가이드](https://github.com/spaceqat/fatqat/blob/main/CONTRIBUTING.md)를 읽어 주십시오.

FatQat은 [Apache License 2.0](https://github.com/spaceqat/fatqat/blob/main/LICENSE)에 따라
배포됩니다.
