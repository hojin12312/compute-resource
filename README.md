# Compute Resource

LLM 컴퓨트 리소스 관련 잡상식을 얻어가는 자료입니다.

## 목차

- [1. Kimi K3는 어떻게 미국 증시를 흔들었나?](#1-kimi-k3는-어떻게-미국-증시를-흔들었나)
- [2. 추론 시장으로의 전환](#2-추론-시장으로의-전환)
- [3. LLM을 쓰기 위해서는 얼마만큼의 메모리가 필요할까요?](#3-llm을-쓰기-위해서는-얼마만큼의-메모리가-필요할까요)
- [4. KV캐시란 뭔가요?](#4-kv캐시란-뭔가요)
- [5. LLM 모델의 연산: 프리필과 디코딩](#5-llm-모델의-연산-프리필과-디코딩)
  - [5.1 프리필](#51-프리필)
  - [5.2 디코딩](#52-디코딩)
  - [5.3 KV 캐시 히트](#53-kv-캐시-히트)
- [6. 아키텍처 살펴보기](#6-아키텍처-살펴보기)
  - [6.1 DeepSeek R1 - Mixture of Experts (MoE)](#61-deepseek-r1---mixture-of-experts-moe)
    - [그래서 왜 빠른가요?](#그래서-왜-빠른가요)
  - [6.2 Qwen3-Next-80B-A3B - Gated DeltaNet](#62-qwen3-next-80b-a3b---gated-deltanet)
    - [왜 Gated DeltaNet은 KV캐시가 필요 없나요?](#왜-gated-deltanet은-kv캐시가-필요-없나요)
  - [6.3 DeepSeek V4 - Hybrid Attention (CSA + HCA)](#63-deepseek-v4---hybrid-attention-csa--hca)
  - [6.4 Kimi K3 - Kimi Delta Attention (KDA)](#64-kimi-k3---kimi-delta-attention-kda)
- [7. Quantization](#7-quantization)
  - [7.1 Model 양자화](#71-model-양자화)
  - [7.2 KV 캐시 양자화 (균일 양자화만 존재)](#72-kv-캐시-양자화-균일-양자화만-존재)
  - [7.3 그래서 양자화하면 뭐가 좋나요?](#73-그래서-양자화하면-뭐가-좋나요)
- [8. AI 소버린](#8-ai-소버린)
  - [8.1 국가 단위의 AI 소버린](#81-국가-단위의-ai-소버린)
  - [8.2 개인이 소유한 AI](#82-개인이-소유한-ai)
- [9. 부록: NVIDIA 데이터센터 GPU 로드맵](#9-부록-nvidia-데이터센터-gpu-로드맵)
  - [9.1 Vera Rubin 플랫폼](#91-vera-rubin-플랫폼)
- [10. 참고 자료](#10-참고-자료)

## 1. Kimi K3는 어떻게 미국 증시를 흔들었나?

> "싸게, 그러나 뛰어나게"

2026년 7월 16일, 중국 Moonshot AI가 세계인공지능대회(WAIC)에서 Kimi K3를 발표했습니다.
총 2.8조(2.8T) 파라미터, 896개 전문가 중 16개만 활성화하는 MoE 구조, 100만 토큰
컨텍스트, 그리고 새로운 Kimi Delta Attention(KDA) 아키텍처를 갖춘 모델입니다.
발표 직후 반도체 지수가 흔들렸는데, 이는 2025년 1월 DeepSeek-R1이 촉발했던
"딥시크 모먼트"와 패턴이 거의 똑같습니다.

발표 다음날인 7월 17일 프리마켓에서 나스닥 선물이 약 1.7%, S&P 500 선물이
약 0.8% 하락했고 NVDA는 2% 이상 밀렸습니다. 블룸버그 아시아 반도체 지수는
6% 넘게 빠졌습니다. 2026년 6월 22일 이후로 누적하면 반도체 섹터 시가총액
약 3.3조 달러가 증발한 상태였는데, 이 흐름 자체는 Kimi K3 발표 이전부터
진행 중이던 밸류에이션 조정이지만 발표가 하락을 가속시켰다는 분석입니다[^tftc].

시장이 반응한 이유는 단순합니다. Kimi K3는 GPU 수출 규제로 최신 가속기를
구하기 어려운 환경에서, 구세대 GPU만으로 학습됐다고 알려져 있습니다.
"최첨단 GPU 없이도 프론티어급 모델을 만들 수 있다"는 서사가 재확인되면,
빅테크의 초대형 GPU 구매 계획(capex)의 근거가 흔들립니다. Nvidia 매출은
결국 "AI를 하려면 우리 GPU가 필요하다"는 전제에 크게 의존하는데, 이
전제가 다시 한 번 도전받은 셈입니다.

> **이미지 생성 프롬프트**: "A financial news infographic showing a stock
> market crash timeline, comparing 'DeepSeek Moment' (January 2025) and
> 'Kimi K3 Moment' (July 2026), with NVIDIA, AMD, TSMC stock candlestick
> charts trending downward, Chinese and American flag motifs subtly in
> background, minimalist data-journalism style, dark navy background with
> red/green candlesticks, clean sans-serif typography"

전체 가중치 공개는 2026년 7월 27일로 예정되어 있어, 이 문서 작성 시점에는
공식 기술 리포트가 아직 나오지 않은 상태입니다. 아래 아키텍처 설명은
Moonshot 블로그와 커뮤니티 분석을 기준으로 정리했습니다.

그런데 "구세대 GPU로도 저렴하게, 그러면서도 뛰어난 모델을 만들었다"는
말이 정확히 무슨 뜻일까요? 모델을 어떻게 설계해야 적은 하드웨어로도
돌아가고, 반대로 하드웨어 스펙은 모델 설계에 어떤 제약을 거는 걸까요.
이 문서는 이 질문에 답하기 위해, 먼저 추론 비용이 왜 이렇게 중요해졌는지
(2장), LLM이 메모리를 얼마나 먹는지(3장)부터 차근차근 살펴봅니다.

## 2. 추론 시장으로의 전환

> "이제 모두가 AI를 쓴다. 그것도 무지막지하게"

최근 몇 달 동안 AI 에이전트는 사내 문화를 크게 뒤흔들고 있습니다. 엘리베이터를
타도, 밥을 먹어도, 옆 부서에서도 AI 얘기로 가득합니다.

점점 더 많은 사람이 AI 에이전트를 활용하기 시작했고 토큰 소비량이 폭증하고
있습니다. 이미 사용하는 사람들도 날이 갈수록 더 많은 토큰을 소비합니다.

- 챗봇 → 에이전트로 판도가 바뀌며 토큰 사용량이 폭증했습니다.
- 에이전트 생산성 향상을 경험한 회사들이 앞다투어 온프레미스 토큰 프로덕션
  환경을 구축하고 있습니다.
- 개인도 토큰 서빙을 위한 로컬 서빙 모델 구축을 시작했습니다. 애플 실리콘,
  DGX Spark 등이 품절 대란을 겪었습니다.
- 쓸 만한 모델을 돌릴 수 있는 GPU들의 가격이 폭등했습니다.
- 모델 사이즈가 점점 커지는 중이라 더 큰 메모리가 필요해지고 있습니다.

가격 폭등은 게이밍용과 로컬 AI 추론에 흔히 쓰이는 GPU 두 종류에서 모두
확인됩니다[^gpu-price].

> **이미지 생성 프롬프트**: "A simple upward-trending line graph titled
> 'Token consumption over time', showing two overlapping curves labeled
> 'Chatbot era' (gentle slope) and 'Agent era' (steep exponential slope)
> diverging sharply after a vertical dashed line labeled '2025', minimalist
> tech-report chart style, single accent color on white background"

## 3. LLM을 쓰기 위해서는 얼마만큼의 메모리가 필요할까요?

> "아주 많이. 그리고 빠른 놈으로."

2026년 출시 모델 기준 파라미터 표입니다[^model-table].

| 모델 | 총 파라미터 | 국가 | 회사 |
|---|---|---|---|
| Qwen3.6 27B | 27B | 중국 | Alibaba (Qwen팀) |
| Qwen3.8 Max | 2.4T | 중국 | Alibaba (Qwen팀) |
| DeepSeek V4 Flash | 284B | 중국 | DeepSeek |
| DeepSeek V4 Pro | 1.6T | 중국 | DeepSeek |
| GLM 5.2 | 744B | 중국 | Zhipu AI (Z.ai) |
| Kimi K3 | 2.8T | 중국 | Moonshot AI |
| Hy3 | 295B | 중국 | Tencent (Hunyuan팀) |
| MiniMax M3 | 428B | 중국 | MiniMax |
| Laguna S 2.1 | 118B | 미국 | Poolside |
| Gemma 4 31B | 31B | 미국 | Google (DeepMind) |
| Motif 3 | 314B | 한국 | Motif Technologies |
| Mistral Medium 3.5 | 128B | 프랑스 | Mistral AI |

일반적으로 원본 모델은 16비트(BF16)로 배포됩니다. 가속기 세대별로 네이티브
가속이 되는 정밀도도 다릅니다.

- A 시리즈(암페어) - FP16 네이티브 가속 가능 (FP16 텐서코어 자체는
  볼타부터 있었지만, 이 문서에서는 최근 세대 중 기준점으로 사용)
- H 시리즈(호퍼) - FP8 네이티브 가속 가능
- B 시리즈(블랙웰) - FP4 네이티브 가속 가능 (NVFP4)

원본 모델을 8비트로 압축했을 경우, 1B ≈ 1GB입니다. 보통 모델을 8비트로
단순 압축 시 출력 품질은 무손실 수준입니다. 작은 모델일수록 양자화에
약하고, 큰 모델일수록 강합니다.

예를 들어 GLM-5.2(744B) 기준, 141GB H200 8장이면 FP8 서빙이
가능합니다[^memory-calc].

하지만 이건 어디까지나 모델을 로드하는 데 필요한 메모리를 계산한 것뿐입니다.
실제로는 여러 오버헤드로 인해 추가 메모리를 필요로 하며, 그 중 가장
지배적인 건 KV캐시입니다.

> **이미지 생성 프롬프트**: "A horizontal bar chart comparing GPU memory
> footprint for two worked examples: GLM-5.2 at FP8 (744GB, spanning 8
> H200 cards at 141GB each) and Kimi K3 at FP8 on Hopper (2.8TB,
> requiring 32 H200 cards) versus Kimi K3 at native MXFP4 on Blackwell
> (1.4TB, only 8 B300 cards at 288GB each), with dashed reference lines
> marking single-card capacity for H200 and B300, clean infographic
> style, muted blue and orange color palette, engineering diagram
> aesthetic"

## 4. KV캐시란 뭔가요?

> "길어질수록, 여럿이서 쓸수록 더 많은 메모리가 필요하다"

LLM 모델은 다음 토큰을 추론하기 위해, 지금까지 입력된 모든 토큰과의 관계를
연산해야 합니다. 따라서 이전 토큰을 처리한 결과값을 항상 테이블에 펼쳐둬야
합니다. 이런 이유 때문에 컨텍스트가 길어지면 일반적으로 KV캐시는 선형으로
증가합니다. 따라서 수십 GB의 중형 모델은 딱 그 정도의 메모리만 있으면 될
것 같지만, 컨텍스트가 길어지면 가중치를 초과하는 수준으로 KV캐시가 자라는
경우가 흔하고, 동시 요청을 처리해야 하는 경우에는 이 캐시 비용이 훨씬 더
빠르게 증가합니다.

KV캐시 크기는 대략 다음 공식으로 추정할 수 있습니다[^kv-formula].

$$
\text{KV 캐시(byte)} = 2 \times \text{batch} \times \text{seq len} \times \text{layer 수} \times \text{KV head 수} \times \text{head dim} \times \text{정밀도(byte)}
$$

즉 "총 파라미터가 작다"는 것과 "KV캐시가 작다"는 것은 별개의 축이며,
최근 아키텍처 경쟁은 사실 이 KV캐시 축에서 벌어지고 있습니다[^kv-cache-opt].

> **이미지 생성 프롬프트**: "A line graph showing KV cache memory size (GB)
> growing with context length (0 to 1M tokens) for four attention types:
> standard Multi-Head Attention (steepest slope), GQA (moderate slope),
> MLA (shallow slope), and hybrid linear attention like Gated
> DeltaNet/KDA (nearly flat), log-scale y-axis, technical whitepaper
> diagram style, blue gradient lines with legend"

## 5. LLM 모델의 연산: 프리필과 디코딩

> "GPU 연산 vs 메모리 대역폭"

LLM 모델의 연산은 크게 두 단계로 구분됩니다. 입력을 처리하는 프리필,
그리고 출력을 만드는 디코딩입니다. 근본적으로는 동일한 연산이지만
입력은 병렬 처리가 가능하므로 빠르고, 출력은 직렬로만 처리가 가능하기에
느립니다.

단적인 예로, 하나의 요청을 처리할 때, 프리필이 초당 수천 개의 토큰을
처리할 수 있다면 디코딩은 수십 토큰 정도에 머무릅니다. 따라서 1:1로
비교했을 때 디코딩이 추론 과정에서 압도적인 병목이 됩니다.

### 5.1 프리필

입력 문장 "I like cats"가 3개 토큰(I / like / cats)이라고 하면, 각
토큰이 자기 앞의 토큰들만 보고 계산한다는 규칙은 미리 다 정해져
있습니다. 그림으로 그리면 계단 모양입니다(`X` = "이 토큰을 본다").

```
        I   like  cats
I       X
like    X    X
cats    X    X    X
```

입력 전체가 이미 손 안에 있으니, GPU는 이 계단 전체를 한 번에
계산합니다. 가중치를 메모리에서 딱 한 번만 읽고 그걸로 모든 토큰을
한꺼번에 처리하는 셈입니다. 그래서 프리필은 메모리를 읽는 속도보다
GPU의 순수 연산 속도가 더 중요합니다.

> **이미지 생성 프롬프트**: "A technical diagram showing 'Prefill' with
> parallel token processing arrows converging into a single GPU compute
> block, labeled 'compute-bound', clean minimalist engineering
> illustration, orange accent color"

### 5.2 디코딩

디코딩은 이 계단을 한 칸씩만 그려나가는 것과 같습니다. 모델이 "I like
cats" 다음에 "too"를 출력한다고 해봅시다.

```
step 1:  I like cats  →  "too" 계산
step 2:  I like cats too  →  다음 단어 계산
```

문제는, step 2를 시작하려면 step 1이 끝나서 "too"라는 단어가 실제로
나와 있어야 한다는 겁니다. 미리 계산해둘 수가 없습니다. 한 단어 →
한 단어, 딱 한 줄씩만 진행됩니다.

그래서 새 단어를 하나 만들 때마다 모델 전체를 처음부터 다시 메모리에서
읽어와야 합니다. 모델 사이즈가 100GB일 경우, 메모리 대역폭[^mem-bandwidth]이
500GB/s이라면 다음 토큰 연산을 위해 메모리를 읽는 데 0.2초가 필요합니다.

즉, 초당 5개의 토큰을 생성할 수 있습니다. 병렬 처리할 방법이 없기
때문에[^spec-decoding] 메모리 대역폭이 디코딩 성능을 좌우합니다.

그래서 최근 아키텍처들은 전부 이 디코딩 병목을 어떻게 낮추느냐를 놓고
경쟁합니다. MoE로 매 토큰마다 읽어야 하는 파라미터 양 자체를 줄이거나,
KV캐시를 압축해서 메모리 대역폭 소모를 줄이는 두 방향이 대표적입니다.
아래에서 각 모델이 택한 구체적인 해법을 살펴봅니다.

> **이미지 생성 프롬프트**: "A technical diagram showing 'Decoding' with a
> sequential token generation loop and repeated memory-read arrows,
> labeled 'memory-bandwidth-bound', clean minimalist engineering
> illustration, blue accent color"

### 5.3 KV 캐시 히트

> "메모리가 충분하면 연산 병목조차 줄어든다"

지금까지는 디코딩이 압도적인 병목이라고 설명했지만, 컨텍스트가 길어질수록
프리필에 의한 병목도 무시할 수 없습니다. 에이전틱 코딩처럼 대화가
멀티턴으로 이어지는 환경에서는 컨텍스트가 빠르게 자라고, 프리필 연산량은
컨텍스트 길이의 제곱으로 늘어나기 때문입니다. 다행히 이 제곱 연산을
선형으로 줄여주는 기술이 있는데, 바로 KV 캐시 히트입니다.

KV 캐시 히트란 이전 턴에서 이미 계산해 둔 KV 캐시를 그대로 재사용하는
기법입니다. 멀티턴 대화에서는 새 턴이 와도 이전 대화 내용은 그대로이고
뒤에 새 입력만 덧붙는 경우가 많습니다. 이때 이전 턴에서 계산해 둔
앞부분의 KV 캐시를 저장해 두었다가 재사용하면, 새로 프리필해야 하는
구간은 방금 추가된 입력뿐입니다. 즉 컨텍스트 전체를 매번 제곱으로
다시 계산하는 대신, 새로 늘어난 부분만 선형으로 계산하면 됩니다.

이 덕분에 멀티턴 에이전트 환경에서는 프리필에 드는 연산 비용을 크게
아낄 수 있고, 그만큼 디코딩에 의한 병목이 상대적으로 더 두드러지게
됩니다.

아래 그래프는 턴이 늘어날수록 프리필에 드는 누적 연산량이 캐시 히트
유무에 따라 얼마나 달라지는지 보여줍니다. 매 턴 새로 추가되는 입력이
1이라고 하면, 캐시가 없을 때는 매번 지금까지 쌓인 컨텍스트 전체를
처음부터 다시 계산해야 하므로 턴별 비용이 1, 2, 3, 4, 5로 늘고 이를
누적하면 1, 3, 6, 10, 15로 제곱형 곡선을 그립니다. 반면 캐시 히트가
있으면 매 턴 새로 추가된 1만 계산하면 되므로 누적 비용도 1, 2, 3, 4,
5로 완전한 직선을 그립니다. 5턴이 지났을 때 총 비용이 15 대 5, 즉
3배 차이가 나는 셈입니다.

![턴이 늘어날수록 누적되는 프리필 비용 — 캐시 없으면 제곱형(15), 캐시 히트 시 선형(5)으로 증가](./assets/kv-cache-hit-cost.png)

이 병목 구조는 실제 API 가격 정책에도 그대로 드러납니다. Anthropic,
OpenAI 등 대부분의 회사는 출력(디코딩) 토큰 가격을 입력(프리필) 토큰
가격보다 5배가량 비싸게 책정합니다. 병렬로 한 번에 처리되는 프리필과
달리, 디코딩은 토큰 하나마다 모델 전체를 메모리에서 다시 읽어야 하니
GPU 입장에서 훨씬 비싼 연산이기 때문입니다. 그리고 캐시 히트가 일어난
입력 토큰은 가격을 90% 할인해줍니다 — 이미 계산해 둔 KV 캐시를
재사용할 뿐 프리필 연산 자체가 거의 필요 없으니, 실제 비용도 그만큼
줄어드는 게 당연합니다.

> **이미지 생성 프롬프트**: "A technical diagram showing a receipt or invoice
> split into two line items, 'output tokens' priced 5x higher than 'input
> tokens', with a 'cache hit' input line stamped '90% off', clean minimalist
> engineering illustration, blue and gray accent colors"

## 6. 아키텍처 살펴보기

### 6.1 DeepSeek R1 - Mixture of Experts (MoE)

> "거대한 도서관, 그 중 당장 필요한 책만"

2025년 1월, DeepSeek R1이 출시되며 미국 증시가 뒤흔들렸습니다("딥시크
모먼트"). 반도체 칩 제재로 열악한 환경에서 미국의 프론티어 모델들과
비견되는 추론 모델을 발표했기 때문입니다.

인프라 투자에 천문학적 금액을 쏟아붓지 않아도 고성능 모델을 만들 수 있는
게 아닌가, 사람들이 의심하게 된 거죠. 심지어 R1은 비슷한 사이즈의 모델보다
추론 속도가 압도적으로 빨랐습니다.

R1은 별도 아키텍처가 아니라 DeepSeek-V3의 구조를 그대로 이어받아
강화학습으로 추론 능력을 얹은 모델입니다. 그리고 이 빠른 추론 속도의
비밀은 V3에서 물려받은 Mixture of Experts(MoE) 구조에 있습니다[^moe-history].
MoE란 전체 파라미터 중 일부만 사용하는 기법으로, 다음 토큰을 출력할 때
모델이 입력을 보고 어떤 전문가를 활성화할지 선택한 후 해당 전문가만
활성화합니다.

실제 수치로 보면 DeepSeek-V3/R1(671B 총 파라미터, 37B 활성 파라미터)은
활성 파라미터가 총 파라미터의 약 5.5%에 불과합니다.

![DeepSeek-V3/R1 총 파라미터 671B 중 활성 파라미터 37B(5.5%) 비교 막대그래프](./assets/moe-active-params.png)

> **이미지 생성 프롬프트**: "A technical diagram of a human brain viewed from
> above, divided into many small regions like a grid of neurons, where only
> a handful of scattered regions glow bright orange (active) while the rest
> stay dimmed gray (inactive), labeled 'Mixture of Experts — only a few
> regions activate per token', clean minimalist engineering illustration"

#### 그래서 왜 빠른가요?

MoE의 이 "활성 파라미터만 쓴다"는 특성은 [앞서 다룬 디코딩
병목](#5-llm-모델의-연산-프리필과-디코딩)에 직접 효과를 냅니다. 디코딩은
매 토큰마다 파라미터를 메모리에서 다시 읽어야 하는데, MoE는 그 읽어야
할 양 자체를 활성 파라미터 비율만큼 줄여줍니다. Dense 671B 모델이라면
매 토큰마다 671B를 통째로 읽어야 하지만, DeepSeek-V3/R1은 37B만 읽으면
됩니다 — 디코딩 속도가 이론상 671/37 ≈ 18배 빨라지는 셈입니다.

프리필에도 같은 라우팅이 적용되므로 총 연산량(FLOPs)이 줄어드는 이득이
있지만, 디코딩만큼 순수하게 반영되지는 않습니다. 프리필은 원래
GPU의 연산 능력(compute)이 지배적인 단계인데, Dense 모델은 전체
시퀀스를 하나의 큰 행렬곱(GEMM)으로 처리해 GPU 효율을 거의 최대로
뽑아냅니다. MoE는 이걸 토큰마다 다른 전문가로 흩어 보내는 구조로
바꾸는데, 전문가 하나에 배정되는 토큰 수가 적으면(짧은 프롬프트·작은
배치일수록) 개별 행렬곱이 작아져 GPU 활용률이 떨어지고, 토큰을
전문가별로 모으고 다시 원래 순서로 되돌리는 라우팅 오버헤드도
붙습니다. 시퀀스가 길거나 배치가 커지면 전문가별 행렬곱도 커져서 이
비효율은 줄어듭니다. 정리하면, MoE는 프리필도 이론적으로는 빠르게
하지만 그 이득이 디코딩만큼 온전히 실현되지는 않고 배치/시퀀스 길이에
좌우됩니다.

### 6.2 Qwen3-Next-80B-A3B - Gated DeltaNet

> "왜 전부 다 적어? 요약해"

Qwen3-Next-80B-A3B(총 80B/활성 3B)는 48개 레이어를 4개씩 묶어, 3개는
Gated DeltaNet(선형 어텐션), 1개는 Gated Attention(GQA, Q 16개/KV
2개 헤드, head_dim 256)으로 구성합니다. 즉 대부분의 레이어가 시퀀스
길이에 선형으로 스케일링되는 연산을 쓰고, 4개 중 1개만 기존 방식의
이차(quadratic) 연산을 유지해서 품질(장기 의존성 포착)과 효율(연산량/
KV캐시) 사이에서 균형을 잡습니다.

#### 왜 Gated DeltaNet은 KV캐시가 필요 없나요?

Full Attention은 매 토큰마다 지금까지 나온 모든 토큰과 다시 내적을
계산해야 합니다. 그래서 두 가지가 동시에 늘어납니다 — 내적 대상인
과거 토큰의 K/V를 전부 저장해둬야 하니 **메모리**(KV캐시)가
컨텍스트 길이에 비례해 늘고, N번째 토큰을 계산할 때마다 그 앞의
N-1개 토큰과 매번 다시 내적해야 하니 토큰 하나 만드는 데 드는
**연산량**도 컨텍스트 길이에 비례해 늘어납니다(그래서 전체 연산량은
길이의 제곱, [5.3절](#53-kv-캐시-히트) 참고).

Gated DeltaNet은 이 방식 대신 RNN처럼 토큰을 하나씩 순서대로 처리하며,
`head_dim × head_dim` 크기의 고정된 "상태(state)" 행렬 하나만 계속
업데이트합니다. 이 state 하나가 메모리와 연산량을 동시에 고정시킵니다
— 크기가 고정이니 저장할 메모리도 고정이고, 매 토큰마다 하는 일도
"이 고정 크기 state를 한 번 업데이트"뿐이라 상수 시간입니다. 그래서
토큰 하나당 연산량이 컨텍스트 길이와 무관하게 일정하고, 전체
연산량도 O(n) — 그래서 "선형 어텐션"이라 부릅니다.

이름의 "Delta"는 그냥 누적하는 게 아니라 "고쳐 쓰는" 방식이라는
뜻입니다. 순수 선형 어텐션은 outer product를 그냥 계속 더하지만

$$
S_t = S_{t-1} + v_t k_t^\top
$$

DeltaNet은 새 값을 쓰기 전에 같은 키로 이미 저장된 옛 값을 먼저
지우고 그 자리에 새 값을 씁니다

$$
S_t = S_{t-1}(I - \beta_t k_t k_t^\top) + \beta_t v_t k_t^\top
$$

이건 매 스텝마다 연상 기억의 오차를 줄이는 온라인 경사하강 한 스텝과
같아서, 단순 누적보다 정보 검색(retrieval) 성능이 좋습니다. Gated
DeltaNet은
여기에 Mamba-2 스타일 감쇠(decay) 게이트를 더해서 오래된 정보를
능동적으로 잊게 만듭니다 — 그래서 "Gated"입니다.

**단, 컨텍스트 길이에 비례한 메모리 증가가 완전히 없어지는 건
아닙니다.** Qwen3-Next-80B-A3B에서 실제로 컨텍스트 길이에 비례해 커지는
KV캐시는 48개 레이어 중 **Gated Attention 12개 레이어의 것뿐**입니다.
나머지 36개 DeltaNet 레이어는 각자 고정 크기 상태만 유지하므로
컨텍스트 길이와 무관합니다. 비유하면, DeltaNet 레이어는 그 시점까지의
과거를 고정 크기로 압축한 요약을 들고 있고, Attention 레이어는 그
시점에서 과거 토큰 하나하나에 정확히 접근할 수 있는 원본 기록(KV캐시)을
들고 있는 셈입니다. 전체 모델의 KV캐시는 이 12개 레이어분만 존재하므로,
48개 레이어 전체가 Full Attention인 모델보다 약 1/4 수준으로 줄어듭니다
(완전히 0이 되는 건 아닙니다).

이후 나온 Qwen3.6 계열은 Dense(27B 전량 활성)와 MoE(35B-A3B, 35B 중 3B
활성) 버전을 함께 내놓아, 같은 하이브리드 어텐션 구조 위에서 Dense와
MoE를 직접 비교하기 좋은 사례입니다.

> **이미지 생성 프롬프트**: "A technical diagram of a repeating stack of
> 4 layer blocks: 3 small fixed-size notebooks labeled 'Gated DeltaNet'
> being overwritten in place with each new entry, followed by 1 long
> unrolling scroll labeled 'Gated Attention (GQA)' growing longer with
> each entry, stacked vertically to show the 3:1 repeating pattern across
> 48 layers, clean minimalist engineering illustration, one accent color
> for notebooks and another for the scroll"

