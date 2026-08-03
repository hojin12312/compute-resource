# 부록 C · V4-Flash-0731은 어느 티어인가

[← 본문으로 돌아가기](../README.md#64-deepseek-v4---hybrid-attention-csa--hca)

본문 [6.4장](../README.md#64-deepseek-v4---hybrid-attention-csa--hca)은 V4를 **가격**으로
소개합니다 — 출력 1M 토큰 $0.28, 캐시 히트는 미스의 2%. 그러면 곧바로 다음 질문이 옵니다.
**그 가격에 붙는 성능은 어느 수준인가.**

이 부록은 DeepSeek이 2026-07-31 공개한 V4-Flash 정식판(0731)의 발표 벤치마크 **아홉 개**와,
같은 날 Artificial Analysis(AA)가 독립 실측한 종합 지수 **하나**를 놓고 그 질문에 답한
것입니다. 아키텍처를 다룬 [부록 A](./deepseek-v4.md)와 짝이 되는 문서로, 여기서는 **같은
모델이 어느 이웃들 사이에 놓이는가**만 봅니다.

근거 표기는 셋입니다. **[1차]** 는 공식 발표·리더보드·논문에서 직접 확인한 것,
**[2차]** 는 집계 사이트나 매체를 경유한 것, **[미검증]** 은 단일 출처이거나 다른 문장에서
역산한 것입니다. 부록 A·B가 쓰는 **[유도]** 는 여기 거의 없습니다 — 이 부록에 실린 수치는
대부분 남이 측정한 것이고, 이 문서의 일은 계산이 아니라 **누가 어떤 조건에서 쟀고 어디까지
검증됐는지 가르는 것**입니다.

**모든 수치는 2026-07-31 스냅샷입니다.** 리더보드와 종합 지수는 상시 갱신되므로 읽는 시점에
따라 값과 순위가 다를 수 있습니다.

## 1. 한 줄 요약

| 항목 | 내용 |
|---|---|
| 무엇이 바뀌었나 | **아키텍처·파라미터 수 동일**(284B / 13B 활성), 포스트트레이닝만 재수행 [1차] |
| 향상 폭 | 대조값이 있는 다섯 벤치마크에서 +14.8 \~ +47.1pp. DeepSWE는 7.5배 [미검증] |
| 독립 실측(AAII) | **49.9**(보도자료 표기 "50"). GLM-5.2 · Muse Spark 1.1(51)과 동급, Kimi K3(57)보다 7점 아래, Claude Opus 5(61)보다 11점 아래 [1차] |
| 실제 티어 | **프론티어 2군** — GPT-5.5 · Claude Sonnet 5 · Claude Fable 5 · GPT-5.6 Terra · Grok 4.5의 이웃 |
| 최상위와 격차 | Claude Opus 5 · GPT-5.6 Sol · Kimi K3 대비 **5\~20pp 아래** |
| 오픈웨이트 순위 | Kimi K3 다음 2위권. GLM-5.2는 항목에 따라 앞서고 뒤섭니다 |
| 가격 | $0.14 / $0.28 per M토큰(캐시 히트 $0.0028), 동시성 2,500 [2차] |
| 가장 큰 유보 | **발표 벤치마크 아홉 개는 전부 DeepSeek 자체 하네스 자체보고입니다.** 독립 측정은 AAII 하나뿐이고, 아홉 개 어디에도 0731의 제3자 측정치가 아직 없습니다 |

**0731 점수의 출처는 두 곳입니다.** 발표 아홉 개는
**[DeepSeek API 업데이트 로그](https://api-docs.deepseek.com/updates/) 한 곳뿐**이고(HF
모델카드와 기술 보고서는 아직 Preview 시점입니다), AAII는
**[Artificial Analysis 모델 페이지](https://artificialanalysis.ai/models/deepseek-v4-flash/)** 입니다.

## 2. 모든 수치에 붙는 단서 — 하네스

DeepSeek이 붙인 공식 각주입니다 [1차]. 공개 벤치마크의 Code Agent 과제는 **DeepSeek Harness
minimal mode**(출시 예정)를 프레임워크로, max effort · top_p=0.95 · temperature=1.0 으로
측정했습니다.

세 가지가 걸립니다.

1. **그 하네스가 공개되지 않았습니다.** 각 벤치마크의 공식 하네스(Terminus 2 ·
   mini-swe-agent · Toolathlon Default · Zapier 공식 셋업)와 다릅니다.
2. **하네스 분산이 런 분산보다 큽니다.** 같은 GPT-5.4가 CyberGym에서 OpenAI 하네스 79.0 대
   Zhipu의 Codex CLI 66.3으로 12.7pp 벌어집니다. AutomationBench public 셋에서는 같은
   GLM-5.2가 측정 주체에 따라 20.33 대 12.9로 7.4pp 갈립니다. **3\~5pp 차이는 무승부로 봐야
   합니다.**
3. **각주의 적용 범위가 모호합니다.** "Code Agent 과제"라고만 적혀 Toolathlon · ALE ·
   AutomationBench에도 같은 하네스를 썼는지 명시가 없습니다.

## 3. 자체보고를 어디까지 믿을 것인가

아홉 개가 모두 자체보고이므로, 먼저 이 회사의 자체보고 이력을 봅니다.

### 3.1 모델카드는 아직 Preview 시점입니다

`deepseek-ai/DeepSeek-V4-Flash`와 `deepseek-ai/DeepSeek-V4-Flash-DSpark` 두 레포 모두
**Preview 시점 카드를 그대로 쓰고 있습니다** [1차]. 카드 본문이 "We present a **preview
version** of DeepSeek-V4 series"로 시작하고, DSpark 레포는 머리에 "**not a new model** —
같은 체크포인트에 투기적 디코딩 모듈만 붙였다"고 명시합니다(DSpark 자체는
[arXiv:2607.05147](https://arxiv.org/abs/2607.05147), 본문
[5.4장](../README.md#54-투기적-디코딩-순차-병목을-우회하는-법) 각주에 정리돼 있습니다).

**벤치마크 세트가 아예 다릅니다.** 그래서 0731 점수와 직접 대조할 수 있는 항목이 없습니다.

| | 모델카드 (Preview 계열) | 0731 릴리스 노트 |
|---|---|---|
| 터미널 | Terminal Bench **2.0** | Terminal Bench **2.1** |
| 툴 사용 | Toolathlon (구 계열) · MCPAtlas | Toolathlon **verified** |
| 코딩 | SWE Verified · SWE Pro · SWE Multilingual | NL2Repo · DeepSWE · DSBench 2종 |
| 그 외 | BrowseComp · HLE w/ tools · GDPval-AA | Cybergym · Agent Last Exam · Automation Bench |

카드의 모드별 점수입니다. 전부 [1차] 자체보고이고, 다음 절의 대조에 쓰입니다.

| 벤치마크 | Non-Think | Think High | **Think Max** |
|---|---|---|---|
| Terminal Bench 2.0 | 49.1 | 56.6 | **56.9** |
| SWE Verified | 73.7 | 78.6 | **79.0** |
| SWE Pro | 49.1 | 52.3 | **52.6** |
| SWE Multilingual | 69.7 | 70.2 | **73.3** |
| Toolathlon (구 계열) | 40.7 | 43.5 | **47.8** |
| MCPAtlas | 64.0 | 67.4 | **69.0** |
| HLE w/ tools | — | 40.3 | **45.1** |
| BrowseComp | — | 53.5 | **73.2** |

카드에서 확인한 아키텍처·학습 정보(32T 토큰 사전학습, mHC, Muon 옵티마이저, FP4+FP8 혼합
정밀도, 1M 컨텍스트)는 본문 6.4장과 [부록 A](./deepseek-v4.md)가 기술 보고서
[arXiv:2606.19348](https://arxiv.org/abs/2606.19348) 원문과 설정 파일에서 더 정확하게
정리해 두었습니다. **그쪽 기록을 우선합니다.**

### 3.2 대조가 두 번 있고, 둘 다 어긋나지 않습니다

**Toolathlon 하나가 유일한 대조 창구입니다.** 공식 리더보드가 0731 공개 하루 전인
2026-07-30에 `DeepSeek V4 Flash (max)`를 **50.9** 로 측정했고 [1차], DeepSeek이 자체보고한
Preview 값은 **49.7** 입니다 [미검증]. **1.2pp 차**입니다.

구 계열에도 같은 대조가 있습니다. 카드의 `Toolathlon = 47.8`(Max)에 대해 구 공식 리더보드는
같은 모델을 **48.2** 로 측정했습니다 — **0.4pp 차**입니다. 더 결정적으로, 같은 카드의 경쟁사
열(Opus-4.6 47.2 · GPT-5.4 xHigh 54.6 · Gemini-3.1-Pro 48.8 · K2.6 50.0 · GLM-5.1 40.7)이
구 공식 리더보드 값과 **전부 일치**합니다. 이 벤치마크에서 DeepSeek은 공식 값을 그대로
인용하고 자사 수치도 공식 측정과 거의 같습니다.

**즉 이 회사가 자체 하네스로 점수를 부풀린 이력은 확인되지 않습니다.** 그래서 아래 표들의
자체보고를 완전히 버릴 이유는 없고, 70.3 같은 상승분도 하네스 효과가 아니라 재포스트트레이닝
효과로 보는 편이 타당합니다.

### 3.3 그래도 0731 자체의 독립 측정은 하나뿐입니다

이 부록을 쓴 2026-07-31은 **0731 공개 당일**이고, 그 시점에 **아홉 벤치마크 어디에도 제3자
측정치가 올라오지 않았습니다.** 독립 근거는 AA의 종합
지수([6장](#6-유일한-독립-실측--artificial-analysis-intelligence-index)) 하나뿐입니다.
Toolathlon 팀은 API 엔드포인트만 받으면 대신 평가하는 공개 서비스를 운영하므로 수주 내 공식
수치가 나올 가능성이 높습니다 — **이 항목이 나머지 여덟 개의 신뢰도를 판단할 프록시입니다.**

## 4. Preview 대비 향상 폭

Preview 수치의 출처는 officechai 한 곳뿐입니다 [미검증]. DeepSeek 공식 changelog와
igeekphone 기사에는 0731 점수만 실려 있고 Preview 대조값이 없습니다. officechai가
"DeepSeek's own charts"를 근거로 든 만큼 공식 발표 차트 이미지에서 읽은 것으로 보이나,
텍스트 1차 출처로는 확인하지 못했습니다.

| 벤치마크 | V4-Flash-Preview | V4-Flash-0731 | 향상 | 배수 |
|---|---|---|---|---|
| Terminal Bench 2.1 | 61.8 | **82.7** | **+20.9** | 1.34배 |
| NL2Repo | 39.4 | **54.2** | **+14.8** | 1.38배 |
| Cybergym | 38.7 | **76.7** | **+38.0** | 1.98배 |
| DeepSWE | 7.3 | **54.4** | **+47.1** | 7.45배 |
| Toolathlon verified | 49.7 | **70.3** | **+20.6** | 1.41배 |
| Agent Last Exam | 데이터 없음 | **25.2** | — | — |
| Automation Bench (Public) | 데이터 없음 | **25.1** | — | — |
| DSBench-FullStack (내부) | 데이터 없음 | **68.7** | — | — |
| DSBench-Hard (내부) | 데이터 없음 | **59.6** | — | — |

상위 모델인 **V4-Pro-Preview는 Terminal Bench 2.1에서 72.1** 이었습니다 [미검증].
0731이 자사 상위 모델을 10.6pp 앞선다는 것이 이번 발표의 홍보 축입니다.

**아키텍처를 바꾸지 않고 포스트트레이닝만으로 DeepSWE를 7.3에서 54.4로 올렸다는 주장이 가장
이례적입니다.** 다만 방향은 정합적입니다 — DeepSWE 공식 리더보드 v1.0에서 DeepSeek-V4-Pro가
8%였으므로, V4 계열이 이 벤치마크에서 실제로 매우 약했던 것은 별도 근거로 확인됩니다.
DeepSWE는 bash 하나만 주고 벤더 편집 프리미티브(`apply_patch`·`str_replace`)를 배제하는데,
논문도 이 표준화가 특정 모델 패밀리를 native ceiling 아래로 누를 수 있다고 한계로 적었습니다.

Terminal Bench 쪽도 어긋나지 않습니다. 카드의 **2.0** Max가 56.9이고 officechai가 인용한
Preview의 **2.1** 값이 61.8인데, 2.1은 환경·지시문 버그를 고쳐 점수가 올라가는 방향이므로
56.9 → 61.8은 정합적입니다. **다만 다른 벤치마크이므로 교차검증이라고 부를 수는 없습니다.**

## 5. 벤치마크별 경쟁 위치

아홉 개입니다. 각 표에 **Preview**와 **0731**을 함께 넣어 두 위치의 이웃이 보이게 했고,
DeepSeek 행은 굵게 표시했습니다.

### 5.1 Terminal Bench 2.1 — 82.7

터미널 환경 89개 과제(Laude Institute · Stanford, [arXiv](https://arxiv.org/abs/2604.02251)).
v2.0의 환경·지시문 오류를 고친 재검증판입니다.

| 모델 | 점수 | 출처 · 하네스 | 근거 |
|---|---|---|---|
| GPT-5.6 Sol (xhigh) | 89.5 | Artificial Analysis · Terminus 2 | [1차] |
| Claude Opus 5 (Adaptive, Max Effort) | 89.1 | Artificial Analysis · Terminus 2 | [1차] |
| Kimi K3 | 88.3 | llm-stats (0.883) | [2차] |
| GPT-5.6 Sol (max) | 88.0 | Artificial Analysis · Terminus 2 | [1차] |
| GPT-5.6 Sol | 85.77 | vals.ai | [2차] |
| Claude Opus 4.8 | 85.0 | DeepSeek 발표 차트 인용 | [미검증] |
| Claude Opus 5 | 84.64 | vals.ai | [2차] |
| GPT-5.5 | 83.4 | morphllm · Codex CLI | [2차] |
| **DeepSeek-V4-Flash-0731** | **82.7** | **DeepSeek 자체보고** | **[1차]** |
| GLM-5.2 | 82.7 | llm-stats (0.827) | [2차] |
| GLM-5.2 | 81.0 | GLM-5.2 모델카드 · Terminus-2 | [1차] |
| Kimi K3 | 80.90 | vals.ai | [2차] |
| Claude Fable 5 | 80.52 | vals.ai | [2차] |
| Claude Opus 4.8 | 78.9 | morphllm · Claude Code | [2차] |
| **DeepSeek-V4-Pro-Preview** | **72.1** | DeepSeek 발표 차트 인용 | [미검증] |
| GLM-5.1 | 63.5 | GLM-5.2 모델카드 | [1차] |
| **DeepSeek-V4-Flash-Preview** | **61.8** | DeepSeek 발표 차트 인용 | [미검증] |

**읽는 법**: 82.7은 llm-stats 집계의 GLM-5.2와 소수점까지 같습니다. Preview는 GLM-5.1
근처였으므로 이번 향상으로 GLM 한 세대를 건너뛴 셈입니다. 최상위권과는 6\~7pp.

**주의**: 같은 벤치마크인데 집계처마다 값이 다릅니다(Kimi K3가 llm-stats 88.3 대 vals.ai
80.9). 하네스와 반복 횟수가 다르기 때문이므로 **한 출처 안에서만 순위를 읽어야 합니다.**

### 5.2 NL2Repo — 54.2

NL2Repo-Bench([arXiv:2512.12730](https://arxiv.org/abs/2512.12730), ByteDance Seed + M-A-P).
빈 워크스페이스에 자연어 명세만 주고 설치 가능한 Python 라이브러리 전체를 만들게 합니다.
104과제, 점수 = 원본 pytest 스위트 평균 통과율.

| 모델 | 점수 | 측정 주체 | 근거 |
|---|---|---|---|
| Claude Opus 4.8 | 69.7 | Z.ai · DeepReinforce | [2차] |
| **DeepSeek-V4-Flash-0731** | **54.2** | **DeepSeek 자체보고** | **[1차]** |
| GPT-5.5 | 50.7 | Z.ai | [2차] |
| GLM-5.2 | 48.9 | Z.ai 자체보고 | [1차] |
| Ornith-1.0-397B | 48.2 | DeepReinforce 자체보고 | [1차] |
| Qwen3.7-Max | 47.2 | Z.ai · DeepReinforce | [2차] |
| GLM-5.1 | 42.7 | Z.ai | [1차] |
| MiniMax M3 | 42.1 | Z.ai · DeepReinforce | [2차] |
| MiniMax M2.7 | 39.8 | MiniMax 자체보고 | [1차] |
| **DeepSeek-V4-Flash-Preview** | **39.4** | DeepSeek 발표 차트 인용 | [미검증] |
| Qwen3.5-397B | 36.8 | DeepReinforce | [2차] |
| DeepSeek-V4-Pro | 35.5 | Z.ai | [2차] |
| Gemini 3.1 Pro | 33.4 | Z.ai | [2차] |

**읽는 법**: Opus 4.8 외 전부를 앞선 유일한 벤치마크입니다. Preview는 MiniMax M2.7 근처
하위권이었습니다.

**주의**: **독립 검증 리더보드가 없습니다.** 위 값 대부분이 Z.ai가 GLM-5.2 카드에서 측정한
경쟁사 점수이고, Kimi K3 · Opus 5 · GPT-5.6 계열 데이터는 아예 없습니다. 논문 저자가 직접
측정한 계열(OpenHands-CodeAct, 2025-12\~2026-01)에서는 Claude-Sonnet-4.5가 40.2로 천장
이었으므로, 벤더 하네스(400K ctx · 48k 출력 · max effort) 계열과 **직접 비교할 수 없습니다.**

### 5.3 Cybergym — 76.7

CyberGym Level 1 pass@1(UC Berkeley Dawn Song 랩,
[arXiv:2506.02548](https://arxiv.org/abs/2506.02548), ICLR 2026). 1,507개 실제 취약점 /
188개 프로젝트. 취약점 설명과 패치 전 코드로 크래시 PoC를 만들면 성공입니다.

| 모델 / 시스템 | 점수 | 비고 | 근거 |
|---|---|---|---|
| MDASH + MAI-Cyber-1-Flash + GPT-5.4 | 95.95 | Microsoft, 리더보드 미등재 | [미검증] |
| Wiz Atlas (GPT-5.5 + Opus 4.6) | 90.9 | 멀티모델 시스템, 리더보드 1위 | [1차] |
| MDASH (GPT-5.4 + Opus 4.6 + Sonnet 4.6) | 88.45 | 멀티모델 시스템 | [1차] |
| GPT-5.5-Cyber | 85.6 | 단일모델 SOTA | [1차] |
| Gemini 3.5 Flash Cyber | 83.2 | CodeMender 5회 호출 | [2차] |
| GPT-5.5 | 81.8 | | [1차] |
| GPT-5.4 | 79.0 | | [1차] |
| **DeepSeek-V4-Flash-0731** | **76.7** | **DeepSeek 자체보고** | **[1차]** |
| Claude Opus 4.7 | 73.1 | OpenAI가 측정한 경쟁사 점수 | [2차] |
| MopMonk Agent + MiniMax M3 | 73.1 | | [1차] |
| GLM-5.1 (Claude Code) | 68.7 | 오픈웨이트 자체보고 최고 | [1차] |
| Claude Opus 4.6 | 66.6 | 시스템 카드 | [1차] |
| GLM-5 | 43.2 | 리더보드 | [2차] |
| Kimi K2.5 | 41.3 | 리더보드 | [2차] |
| OpenHands + GPT-5 | 39.4 | 리더보드 | [2차] |
| Gemini 3.1 Pro (Gemini CLI) | 38.8 | Zhipu 측정 | [2차] |
| **DeepSeek-V4-Flash-Preview** | **38.7** | DeepSeek 발표 차트 인용 | [미검증] |

**읽는 법**: 76.7은 GPT-5.4와 Opus 4.7 사이 빈 구간에 단독으로 놓입니다. Preview는
Gemini 3.1 Pro · GLM-5 근처였으므로 두 배 가까이 올랐습니다.

**주의**: **80점 이상은 대부분 모델 점수가 아니라 시스템 점수입니다.** Wiz Atlas · MDASH ·
Sangfor · Xuanwu는 멀티모델 오케스트레이션이고 Crystalline은 사전 시딩한 지식베이스를
씁니다. 단일 모델끼리만 보면 GPT-5.5 → GPT-5.4 → 0731 순입니다. 또 DeepSeek은 Level과
trial 수를 명시하지 않았습니다(리더보드 관행상 Level 1 / pass@1로 추정하나 미확인).

### 5.4 DeepSWE — 54.4

Datacurve의 오염 방지 롱호라이즌 코딩 벤치마크(113과제 / 91 리포 / 5개 언어,
[arXiv:2607.07946](https://arxiv.org/html/2607.07946)). 과제를 직접 작성해 업스트림에
머지하지 않고, **하네스를 mini-swe-agent로 고정**하는 것이 설계 철학입니다.

| 모델 | pass@1 | 과제당 비용 | 근거 |
|---|---|---|---|
| claude-opus-5 [max] | 74% ±4 | $11.84 | [1차] |
| gpt-5.6-sol [max] | 73% ±3 | $8.39 | [1차] |
| claude-fable-5 [max] | 70% ±4 | $21.63 | [1차] |
| gpt-5.6-terra [max] | 70% ±3 | $3.96 | [1차] |
| kimi-k3 [max] | 69% ±5 | $4.65 | [1차] 오픈웨이트 1위 |
| gpt-5.6-luna [max] | 67% ±4 | $0.61 | [1차] |
| gpt-5.5 [xhigh] | 67% ±6 | $7.23 | [1차] |
| claude-opus-4.8 [max] | 59% ±2 | $13.22 | [1차] |
| **DeepSeek-V4-Flash-0731** | **54.4** | — | **[1차]** 자체 하네스 |
| claude-sonnet-5 [max] | 54% ±4 | $26.40 | [1차] |
| grok-4.5 [high] | 54% ±2 | $2.42 | [1차] |
| muse-spark-1.1 [xhigh] | 53% ±3 | $2.36 | [1차] |
| gpt-5.4 [xhigh] | 52% ±2 | $5.65 | [1차] |
| gemini-3.6-flash [high] | 49% ±5 | $3.53 | [1차] |
| glm-5.2 [max] | 44% ±2 | $3.92 | [1차] |
| gemini-3.5-flash [medium] | 37% ±2 | $7.34 | [1차] |
| claude-sonnet-4.6 [high] | 30% ±4 | $5.52 | [1차] |
| gemini-3.1-pro [high] | 12% ±2 | $9.48 | [1차] |
| DeepSeek-V4-Pro (v1.0 계열) | 8% ±2 | — | [2차] |
| **DeepSeek-V4-Flash-Preview** | **7.3** | — | [미검증] |

**읽는 법**: Claude Sonnet 5 · Grok 4.5와 오차범위 내 동급입니다. SOTA와 20pp, 오픈웨이트
1위 Kimi K3와 14pp. Preview는 Gemini 3.1 Pro보다도 아래인 사실상 최하위였습니다.

**주의**: 공식 리더보드에 DeepSeek 모델이 **한 개도 없습니다.** 54.4는 mini-swe-agent가 아닌
DeepSeek 자체 하네스 값이고, 하네스 고정이 이 벤치마크의 존재 이유이므로 같은 표에 놓는
것은 엄밀히 동일 조건 비교가 아닙니다. 또 v1.0과 v1.1은 채점 방식이 달라 섞을 수 없습니다
(GPT-5.5 xhigh가 v1.0 70.0 → v1.1 67.04).

### 5.5 Toolathlon verified — 70.3

The Tool Decathlon(HKUST NLP, [arXiv:2510.25726](https://arxiv.org/abs/2510.25726),
ICLR 2026). 108과제 / 32앱 / 604툴, 평균 20턴. Verified는 서브셋이 아니라 채점 인프라를
대수선한 재검증 릴리스(83/108 과제 패키지 변경)이며 **점수 계열이 새로 시작**됩니다.

| 모델 | Pass@1 | 측정일 | 근거 |
|---|---|---|---|
| Kimi K3 (max) | 76.5 ±1.9 | 2026-07-16 | [1차] |
| Claude Opus 4.8 (max) | 76.2 ±3.4 | 2026-06-30 | [1차] |
| Muse Spark 1.1 (xhigh) | 75.6 ±0.4 | 2026-07-09 | [1차] |
| GPT-5.5 (xhigh) | 73.5 ±1.2 | 2026-06-30 | [1차] |
| Claude Sonnet 5 (max) | 71.6 ±1.2 | 2026-07-01 | [1차] |
| **DeepSeek-V4-Flash-0731** | **70.3** | 2026-07-31 | **[1차]** 자체보고 |
| Gemini 3.5 Flash (high) | 67.3 ±1.2 | 2026-06-30 | [1차] |
| Gemini 3.1 Pro (high) | 61.1 ±1.3 | 2026-07-01 | [1차] |
| GLM 5.2 (max) | 59.9 ±1.9 | 2026-06-30 | [1차] |
| Kimi K2.6 / K2.7 Code | 58.0 | 2026-07-15 / 06-30 | [1차] |
| Gemini 3.5 Flash-Lite (high) | 57.1 ±3.6 | 2026-07-30 | [1차] |
| Hy3 (high) | 56.8 ±3.7 | 2026-07-30 | [1차] |
| DeepSeek V4 Pro (max) | 55.9 ±1.2 | 2026-06-30 | [1차] |
| Inkling-Small (xhigh) | 54.4 ±3.2 | 2026-07-30 | [1차] |
| **DeepSeek V4 Flash (max) — 공식 리더보드 측정** | **50.9 ±1.5** | **2026-07-30** | **[1차]** |
| **DeepSeek-V4-Flash-Preview — 자체보고** | **49.7** | — | [미검증] |
| MiMo V2.5 | 49.1 ±3.9 | 2026-07-30 | [1차] |
| MiniMax M2.7 | 47.5 ±3.8 | 2026-07-30 | [1차] |
| Qwen3.5 397B-A17B | 40.7 ±2.0 | 2026-07-30 | [1차] |

**읽는 법**: 아홉 벤치마크 중 **유일하게 공식 측정과 자체보고를 나란히 볼 수 있는 항목**이고,
그 대조가 [3.2장](#32-대조가-두-번-있고-둘-다-어긋나지-않습니다)의 근거입니다. 70.3이면
Claude Sonnet 5와 통계적으로 구분되지 않고(108과제 기준 1과제 ≈ 0.93pp) 오픈웨이트 2위가
됩니다.

**주의**: 리더보드가 체크포인트 날짜를 명시하지 않으므로 50.9가 Preview라는 것은 날짜 기반
추론입니다. 0731이 공식 리더보드에 오르면 재확인이 필요합니다.

### 5.6 Agent Last Exam — 25.2

Agents' Last Exam(UC Berkeley RDI, [arXiv:2606.05405](https://arxiv.org/abs/2606.05405)).
전문가가 실제로 며칠\~몇 주 걸려 끝낸 프로젝트를 과제화한 컴퓨터 사용 에이전트 벤치마크입니다.
55개 서브도메인 / 1,490 인스턴스, 공개분 152개. 지표는 Full Pass Rate.

| 모델 (하네스) | Full Pass Rate | 근거 |
|---|---|---|
| GPT-5.6 Sol (Codex, xhigh) | 30.6 | [1차] SOTA |
| Kimi K3 (kimi_code) | 28.3 | [1차] |
| GPT-5.6 Terra (Codex, max) | 28.0 | [1차] |
| Grok 4.5 (grok_build) | 27.0 | [1차] |
| Claude Opus 4.8 (Claude Code) | 27.0 | [1차] |
| GPT-5.5 (Codex, xhigh) | 26.6 | [1차] |
| Claude Fable 5 (Claude Code, xhigh) | 25.7 | [1차] |
| **DeepSeek-V4-Flash-0731** | **25.2** | **[1차]** 자체보고 |
| GPT-5.5 (Codex, 기본) | 24.2 | [1차] |
| Claude Fable 5 (Claude Code, 기본) | 22.0 | [1차] |
| GLM-5.2 (Claude Code, max) | 20.4 | [1차] |
| Claude Opus 4.7 (Cursor) | 20.4 | [1차] |
| Gemini 3.1 Pro (gemini_cli) | 15.8 | [1차] |
| DeepSeek-V4-Pro (OpenClaw) | 12.4 | [1차] |
| GLM-5.1 (OpenClaw) | 11.5 | [1차] |
| GPT-5.4 (Codex) | 7.2 | [1차] |
| **DeepSeek-V4-Flash-Preview** | **데이터 없음** | — |

**읽는 법**: **벤치마크 자체의 천장이 30% 수준**이라 25.2는 SOTA의 82%이고 Fable 5 ·
GPT-5.5와 같은 구간입니다. 상위 난이도 티어(38과제)로 좁히면 1위 Grok 4.5가 10.5%,
GPT-5.6 Sol이 5.3%이고 Gemini 3.1 Pro와 GPT-5.4는 0%입니다. 즉 25% 근처는 전부 **"쉬운
층만 푼"** 점수입니다.

**주의 둘.** **DeepSeek이 지표와 split을 밝히지 않았습니다** — 전체 152과제 Pass Rate인지
CLI 전용 서브셋(ALE-CLI, 약 105과제, SOTA 28.6)인지 불명입니다. ALE 논문이 "DeepSeek V4는
native vision이 없어 GUI-as-SubAgent가 필요하다"고 적었으므로 CLI split일 개연성이 있습니다.
그리고 **25.2는 ALE 논문 본문이 Codex+GPT-5.5의 ALE-CLI 값으로 발표한 수치와 정확히
같습니다.** 우연인지 타겟팅인지 판별할 수 없으므로 이것으로 어떤 결론도 내리지 않습니다.

### 5.7 Automation Bench (Public) — 25.1

Zapier의 SaaS 워크플로 자동화 벤치마크
([arXiv:2604.18934](https://arxiv.org/abs/2604.18934)). 600과제 public 셋(6개 업무 도메인 ×
100), 47개 시뮬레이션 앱 / 약 500 엔드포인트. **부분점수 없는 all-or-nothing 채점입니다.**

| 모델 | Pass Rate | 측정 주체 | 근거 |
|---|---|---|---|
| Kimi K3 (max) | 30.8 | Moonshot 자체측정 | [1차] |
| Claude Opus 4.8 (max) | 30.33 | Zapier | [1차] |
| GPT-5.6 Sol (max) | 29.17 | Zapier | [1차] |
| GPT-5.6 Terra (max) | 25.83 | Zapier | [1차] |
| Claude Fable 5 (max) | 25.83 | Zapier | [1차] |
| **DeepSeek-V4-Flash-0731** | **25.1** | **DeepSeek 자체보고** | **[1차]** |
| Claude Sonnet 5 (max) | 24.00 | Zapier | [1차] |
| GPT-5.5 (xhigh) | 22.7 | Moonshot | [1차] |
| GLM 5.2 (max) | 20.33 | Zapier | [1차] |
| Gemini 3.5 Flash (high) | 14.83 | Zapier | [1차] |
| **DeepSeek-V4-Flash-Preview** | **데이터 없음** | — | — |

**읽는 법**: GPT-5.6 Terra · Claude Fable 5와 0.7pt 차 — Zapier가 밝힌 런 간 분산 1% 이내
이므로 무승부입니다. SOTA와 5\~6pp.

**주의**: DeepSeek이 말한 **(Public)은 Artificial Analysis의 AutomationBench-AA와 다른
것입니다.** AA는 Zapier private 셋 657과제에서 objective 단위 비율로 채점해(가드레일 위반
과제만 0점) 같은 모델도 1.5\~2배 높게 나옵니다(Opus 4.8 48.5 · GLM-5.2 27.8). **AA 표와
나란히 놓으면 안 됩니다.**

절대 수준이 낮은 것은 설계상 그렇습니다. 25%는 업무 워크플로 넷 중 셋을 끝까지 실패한다는
뜻입니다. 지배적 실패 모드는 성능 부족이 아니라 **거짓 완료 선언**입니다 — Zapier 분석에서
Opus 실패의 72%, Gemini 91%, GPT-5.4 84%가 "다 했다"고 보고했으나 최종 상태가 틀렸습니다.
ALE의 실패 원인 분류(잘못된 전략 30% + 조기 포기 17%)도 같은 패턴을 지적합니다.

### 5.8 DSBench-FullStack (내부) — 68.7

DeepSeek 내부 풀스택 개발 테스트셋입니다 [1차]. **외부 비교군이 존재하지 않습니다.**

| 모델 | 점수 | 근거 |
|---|---|---|
| Claude Opus 4.8 | 71.6 | DeepSeek 발표 차트 인용 [미검증] |
| **DeepSeek-V4-Flash-0731** | **68.7** | **[1차]** 자체보고 |
| **DeepSeek-V4-Flash-Preview** | **데이터 없음** | — |

과제 수·채점 방식·공개 계획 모두 밝혀지지 않았습니다. Opus 4.8 대비 2.9pp 열위라는 것이
DeepSeek 주장의 전부이고, 제3자가 재현할 수단이 없습니다. **발표 자료에 인용할 근거로는
쓰지 않는 편이 안전합니다.**

### 5.9 DSBench-Hard (내부) — 59.6

DeepSeek 내부 Coding Agent 고난도 문제셋입니다 [1차]. 마찬가지로 외부 비교군이 없습니다.

| 모델 | 점수 | 근거 |
|---|---|---|
| Claude Opus 4.8 | 71.7 | DeepSeek 발표 차트 인용 [미검증] |
| **DeepSeek-V4-Flash-0731** | **59.6** | **[1차]** 자체보고 |
| **DeepSeek-V4-Flash-Preview** | **데이터 없음** | — |

Opus 4.8과의 격차가 **12.1pp**로, NL2Repo(15.5pp) 다음으로 큽니다. 자체 내부 셋에서 격차가
이만큼 크게 나왔다는 점은 오히려 이 셋의 난이도 설계가 정직했다는 신호로 읽을 수 있습니다.

## 6. 유일한 독립 실측 — Artificial Analysis Intelligence Index

AA의 종합 지능 지표(v4.1, 9개 평가 가중합: GDPval-AA v2 · τ³-Bench Banking ·
Terminal-Bench 2.1 · SciCode · Humanity's Last Exam · GPQA Diamond · CritPt ·
AA-Omniscience · AA-LCR)입니다. **이 절만 DeepSeek 자체보고가 아니라 AA가 2026-07-31 자체
하네스로 직접 측정한 값**이므로 근거의 성격이 다릅니다.

| 모델 | AAII | 근거 |
|---|---|---|
| Claude Opus 5 | 61 | [1차] AA 실측 · 전체 1위 |
| GPT-5.6 Sol | (Fable 5와 함께 top 3) | [1차] AA 실측 |
| Claude Fable 5 | (Opus 5 · Sol과 top 3) | [1차] AA 실측 |
| Kimi K3 | 57 | [1차] AA 실측 · 오픈웨이트 1위 |
| GPT-5.6 Luna (max effort) | 51 | [1차] AA 실측 |
| GLM-5.2 | 51 | [1차] AA 실측 |
| Muse Spark 1.1 | 51 | [1차] AA 실측 |
| Gemini 3.6 Flash | 50 | [1차] AA 실측 |
| **DeepSeek-V4-Flash-0731** | **49.9** (보도자료 표기 "50") | **[1차]** AA 실측 |
| DeepSeek V4 Pro | 44 (기사의 "6점 차" 역산) | **[미검증]** — AA가 숫자로 명시하지 않음 |
| DeepSeek-V4-Flash-Preview (AA 4월 공식 아티클) | **47** | **[1차]** 원문 "V4 Flash (Max) scores 47" |
| DeepSeek-V4-Flash-Preview (AA 라이브 페이지, 2026-07-30) | **40** | **[1차]** 본문 6.4 스펙 카드가 읽은 값 |
| DeepSeek-V4-Flash-Preview (officechai "+10점" 역산) | **40** | **[미검증]** — 4월 아티클의 47과 **불일치** |

**읽는 법**: 0731은 GLM-5.2 · Muse Spark 1.1과 사실상 동급(±1점)이고 Gemini 3.6 Flash와도
거의 일치합니다. GPT-5.6 Luna가 가격 80% 인하 직후에도 딱 1점 앞서지만, 캐시 히트 단가가
DeepSeek이 훨씬 저렴해 **동일 지능 대비 과제당 비용은 0731이 약 60% 낮다**고 AA는
설명합니다. 오픈웨이트 1위 Kimi K3와는 7점, 전체 1위 Claude Opus 5와는 11점 차입니다.

**주의 — Preview 기준점 자체가 두 출처에서 어긋납니다.** AA가 2026-04에 낸 공식 아티클
원문은 "V4 Flash (Max) scores 47"이라고 명시하는데, officechai의 0731 기사는 "a jump of
10 points over the original DeepSeek V4 Flash from April"이라고만 적어 역산하면 40이 됩니다.
47과 40 중 어느 쪽이 정확한 대조군인지 이 문서만으로는 결론낼 수 없습니다. 가능성은 둘입니다 —
(a) officechai가 어림수로 서술하며 오차가 생겼거나, (b) AA가 4월 이후 Preview 모델을
재평가해 47에서 40으로 내렸을 수 있습니다(AA는 지수를 상시 갱신하는 곳입니다).
**정황은 (b) 쪽입니다** — 본문 [6.4장](../README.md#64-deepseek-v4---hybrid-attention-csa--hca)
스펙 카드가 쓴 AAII v4.1 **2026-07-30 라이브 값이 V4 Flash를 40으로 표기**했습니다(0731 공개
하루 전, 각주 `[^v4-spec]` 8항). 같은 페이지가 4월에 47이었다면 그 사이에 내려간 것이 됩니다.
다만 **AA가 하향 사실을 문장으로 밝힌 것은 확인하지 못했고**, "+10점"이라는 문구 자체도
officechai 표현입니다.
V4 Pro의 "44"도 같은 이유로 "6 points clear of DeepSeek V4 Pro"라는 문장의 역산입니다 —
AA 4월 아티클의 V4 Pro(Max) 점수는 **52** 였으므로 Pro 쪽도 같은 하향을 겪었다는 뜻이 됩니다.

**세부 평가 변동**(officechai 인용, AA 실측 [1차]): GDPval-AA v2가 1189 → 1559로 370점
상승(인간 기준선 1,000). Terminal-Bench 2.1이 17점 올라 79% — [5.1장](#51-terminal-bench-21--827)의
자체보고 82.7과는 **다른 수치이고, 하네스가 달라 직접 비교할 수 없습니다.** τ³-Bench
Banking이 8점 올라 31%, CritPt +9점(17%), SciCode +5점(50%), HLE +5점(37%).
AA-Omniscience는 −23에서 −16으로 개선됐지만 정답률은 37%로 그대로이고 환각률만 84%로
낮아진 것 — **"더 아는 것"이 아니라 "모르면 답을 거부하는 빈도가 늘어난 것"** 이라고 AA는
분석합니다. 지수 전체 실행에 쓴 출력 토큰은 206M으로 Preview의 234M보다 12% 줄었습니다.
0731의 가중치는 아직 미공개이며, 공개되면 GDPval-AA v2 기준 오픈웨이트 2위(Kimi K3 1687
Elo 다음, GLM-5.2 1510보다 위)가 될 것으로 AA는 예상합니다.

## 7. 종합 — 실제 티어

일곱 개 공개 벤치마크와 AA의 독립 실측 하나를 나란히 놓으면 위치가 일관됩니다.

| 벤치마크 | 0731 | 동급 모델 | 최상위와 격차 |
|---|---|---|---|
| Terminal Bench 2.1 | 82.7 | GLM-5.2 (82.7) · GPT-5.5 (83.4) | −6.8 (GPT-5.6 Sol 89.5) |
| NL2Repo | 54.2 | GPT-5.5 (50.7) | −15.5 (Opus 4.8 69.7) |
| Cybergym | 76.7 | GPT-5.4 (79.0) · Opus 4.7 (73.1) | −8.9 (GPT-5.5-Cyber 85.6) |
| DeepSWE | 54.4 | Sonnet 5 (54) · Grok 4.5 (54) | −19.6 (Opus 5 74) |
| Toolathlon verified | 70.3 | Sonnet 5 (71.6) | −6.2 (Kimi K3 76.5) |
| Agent Last Exam | 25.2 | Fable 5 (25.7) · GPT-5.5 (26.6) | −5.4 (GPT-5.6 Sol 30.6) |
| Automation Bench | 25.1 | Terra (25.83) · Fable 5 (25.83) | −5.7 (Kimi K3 30.8) |
| **AAII (AA 독립 실측)** | **49.9** | **GLM-5.2 (51) · Muse Spark 1.1 (51) · Gemini 3.6 Flash (50)** | **−11.1 (Claude Opus 5, 61)** |

가장 자주 등장하는 이웃이 **GPT-5.5 · Claude Sonnet 5 · Claude Fable 5 · GPT-5.6 Terra ·
Grok 4.5**입니다. 이게 실제 티어입니다. "Opus 4.8급"이라는 언론 표현은 벤치마크마다 맞기도
하고 틀리기도 합니다 — AutomationBench에서 −5.2, DeepSWE에서 −4.6이지만 NL2Repo에서 −15.5,
DSBench-Hard에서 −12.1입니다.

**AAII만 자체보고가 아닌 항목입니다.** 일곱 개 자체보고가 가리키는 이웃(GPT-5.5 · Sonnet 5 ·
Fable 5 · Terra · Grok 4.5)과 AA의 독립 실측이 가리키는 이웃(GLM-5.2 · Muse Spark 1.1 ·
Gemini 3.6 Flash, Kimi K3보다 7점 아래)이 서로 다른 비교군이지만, **"프론티어 2군, 최상위와
5\~20pp 아래"라는 결론은 두 근거 계열 모두에서 같습니다.**

**그래서 본문 6.4장의 논지는 성능 순위가 아니라 성능 대비 단가입니다.** 284B 중 13B만
활성화되는 MoE로 $0.14 / $0.28인데, $3 / $15의 Sonnet 5와 $10 / $50의 Fable 5와 같은
점수대를 냅니다. 같은 티어를 한 자릿수 낮은 단가로 판다는 것이 이 모델의 성과이고, 스펙
카드의 왼쪽 패널이 벤치마크 성적표가 아니라 **점수와 출력 단가를 상하로 짝지은 차트**인
이유입니다(6.5장 GLM-5.2 카드는 성적이 논지라 그 자리가 벤치마크입니다).

## 8. 미확인·후속 확인 항목

| 항목 | 상태 |
|---|---|
| **HF 모델카드의 0731 점수** | **없습니다.** 두 레포 모두 Preview 시점 카드이고 벤치마크 세트도 다릅니다 [1차] |
| Preview 다섯 개 점수(61.8 · 39.4 · 38.7 · 7.3 · 49.7) | officechai 단일 출처 [미검증]. 공식 차트 이미지 원본 확인 필요 |
| Agent Last Exam · Automation Bench의 Preview 값 | 데이터 없음 |
| DSBench-FullStack · DSBench-Hard 사양 | 미공개 (과제 수·채점·공개 계획 전부) |
| DeepSeek Harness minimal mode 스펙 | 미공개 ("출시 예정") |
| 0731의 제3자 독립 측정 | 발표 아홉 개 전부 없음 (AAII만 독립) |
| 0731 가중치 공개 | 미공개 — API 전용. 모델카드가 갱신되면 0731 자체 점수표가 나올 가능성 |
| Toolathlon 공식 리더보드의 0731 등재 | 대기 중 — 등재되면 자체보고 검증의 첫 기준점 |
| Agent Last Exam의 split·지표 정의 | DeepSeek 미명시 (전체 152 대 ALE-CLI 약 105) |
| Cybergym의 Level·trial 수 | DeepSeek 미명시 (Level 1 / pass@1 추정) |
| **AAII Preview 대조값 47 vs 40** | **불일치.** AA 공식 아티클(4월) 원문은 47, officechai의 "+10점"을 역산하면 40. **본문 스펙 카드가 읽은 7월 30일 라이브 값도 40**이라 4월 이후 하향이 정황상 유력하지만, AA가 그것을 문장으로 밝힌 것은 확인하지 못함 |
| **V4 Pro의 AAII 44** | officechai "6점 차" 문장의 역산일 뿐 AA가 숫자로 명시하지 않음. AA 4월 아티클의 52와 배치되므로 Pro 쪽 재측정 여부 확인 필요 |
| Terminal-Bench 2.1: AA 실측 79 vs 자체보고 82.7 | 하네스가 달라 직접 비교 불가. 병기할 때는 하네스 차이를 함께 적을 것 |
| AAII v4.1의 개별 원자료(GDPval-AA v2 1559 등) | officechai 인용문 경유로만 확인, AA 사이트 원 표·차트 직접 열람 필요 |

## 9. 출처

**1차**

- [DeepSeek API 업데이트 로그](https://api-docs.deepseek.com/updates/) — 0731 아홉 점수와 측정 각주
- [Toolathlon 공식 리더보드](https://hkust.mintlify.app/docs/leaderboard) — Verified 21개 항목, DeepSeek V4 Flash 50.9 포함
- [DeepSWE 공식 리더보드](https://deepswe.datacurve.ai/) — 18개 설정, 2026-07-25 갱신
- [Zapier AutomationBench GitHub](https://github.com/zapier/AutomationBench) — public 600과제 공식 점수표
- [Artificial Analysis Terminal-Bench v2.1](https://artificialanalysis.ai/evaluations/terminalbench-v2-1) — Terminus 2 하네스
- [Agents' Last Exam 리더보드 API](https://agents-last-exam.org/api/demo/leaderboard) · [논문](https://arxiv.org/abs/2606.05405)
- [CyberGym 공식 사이트](https://www.cybergym.io/cybergym/) · [논문](https://arxiv.org/abs/2506.02548)
- [GLM-5.2 모델카드](https://huggingface.co/zai-org/GLM-5.2) — NL2Repo · DeepSWE · Terminal Bench 경쟁사 측정표
- [Kimi K3 모델카드](https://huggingface.co/moonshotai/Kimi-K3) — AutomationBench public 셋 측정
- [DeepSeek-V4-Flash 카드](https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash) · [DSpark 카드](https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash-DSpark) — **Preview 시점 점수표**(0731 아님). Toolathlon 47.8 · Terminal Bench 2.0 56.9 등
- [V4 기술 보고서 arXiv:2606.19348](https://arxiv.org/abs/2606.19348) — 아키텍처·효율 수치(상세 분석은 [부록 A](./deepseek-v4.md))
- [Artificial Analysis · V4 Flash 0731 모델 페이지](https://artificialanalysis.ai/models/deepseek-v4-flash/) — AAII 49.9와 세부 평가 원자료
- [Artificial Analysis 공식 아티클 (2026-04, V4 Pro·Flash 최초 출시)](https://artificialanalysis.ai/articles/deepseek-is-back-among-the-leading-open-weights-models-with-v4-pro-and-v4-flash) — **Preview AAII 원문 수치(V4 Pro Max 52 · V4 Flash Max 47)**

**2차**

- [officechai 0731 기사](https://officechai.com/ai/deepseek-releases-deepseek-v4-flash-0731-gives-opus-4-8-level-performance-at-a-fraction-of-the-price/) — **Preview 대조값의 유일한 출처**
- [officechai AAII 기사](https://officechai.com/ai/deepseek-v4-flash-0731-scores-50-on-artificial-analysis-intelligence-index-creates-big-spike-on-pareto-frontier/) — AAII 50점, 세부 평가 변동, 인접 모델
- [dataconomy V4 Flash 0731 페이지](https://dataconomy.com/ai-models/deepseek-v4-flash-ga/) — AAII 49.9/50 교차확인, 가격·컨텍스트 재확인
- [igeekphone 0731 기사](https://www.igeekphone.com/deepseek-launches-official-v4-flash-api-public-preview-confirms-v4-pro-release-coming-soon/) — 아홉 점수 표 교차확인 (Preview 값 없음)
- [aibase 0731 기사](https://www.aibase.com/en/news/30040) — 세 점수 교차확인
- [llm-stats](https://llm-stats.com/) · [vals.ai](https://vals.ai/) · [morphllm](https://www.morphllm.com/ai-coding-agent) — 집계 사이트

**인용 비권장**

llm-stats의 Toolathlon(Kimi K3를 73.2로 표기, 공식 76.5와 불일치)과 AutomationBench
(GPT-5.6 Terra 15.2 · Sonnet 5 13.5, Zapier 공식 25.83 · 24.00과 정면 충돌). 두 항목 모두
사이트가 "verified 0건"이라고 명시합니다.
