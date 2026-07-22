# Compute Resource

LLM 추론/서빙 관점의 컴퓨트 리소스(연산량, 메모리) 관련 기술 동향을 정리하는 레포.

## 다룰 예정인 내용

- LLM 모델 파라미터 수와 점유 메모리와의 관계
- KV 캐시
- 프리필(prefill) / 디코딩(decoding) 단계의 연산 병목
- Dense 모델 vs MoE(Mixture of Experts) 모델

## 목차

- [Kimi K3는 어떻게 미국 증시를 흔들었나?](#kimi-k3는-어떻게-미국-증시를-흔들었나)
- [추론 시장으로의 전환](#추론-시장으로의-전환)
- [LLM을 쓰기 위해서는 얼마만큼의 메모리가 필요할까요?](#llm을-쓰기-위해서는-얼마만큼의-메모리가-필요할까요)
- [KV캐시란 뭔가요?](#kv캐시란-뭔가요)
- [LLM 모델의 연산: 프리필과 디코딩](#llm-모델의-연산-프리필과-디코딩)
- [아키텍처 살펴보기: MoE부터 최신 Attention까지](#아키텍처-살펴보기-moe부터-최신-attention까지)
- [Quantization](#quantization)
- [부록: NVIDIA 데이터센터 GPU 로드맵](#부록-nvidia-데이터센터-gpu-로드맵)
- [참고 자료](#참고-자료)

## Kimi K3는 어떻게 미국 증시를 흔들었나?

2026년 7월 16일, 중국 Moonshot AI가 세계인공지능대회(WAIC)에서 Kimi K3를 발표했습니다.
총 2.8조(2.8T) 파라미터, 896개 전문가 중 16개만 활성화하는 MoE 구조, 100만 토큰
컨텍스트, 그리고 새로운 Kimi Delta Attention(KDA) 아키텍처를 갖춘 모델입니다.
발표 직후 반도체 지수가 흔들렸는데, 이는 2025년 1월 DeepSeek-R1이 촉발했던
"딥시크 모먼트"와 패턴이 거의 똑같습니다.

발표 다음날인 7월 17일 프리마켓에서 나스닥 선물이 약 1.7%, S&P 500 선물이
약 0.8% 하락했고 NVDA는 2% 이상 밀렸습니다. 블룸버그 아시아 반도체 지수는
6% 넘게 빠졌습니다. 2026년 6월 22일 이후로 누적하면 반도체 섹터 시가총액
약 3.3조 달러가 증발한 상태였는데, 이 흐름 자체는 Kimi K3 발표 이전부터
진행 중이던 밸류에이션 조정이지만 발표가 하락을 가속시켰다는 분석입니다
([TFTC](https://www.tftc.io/kimi-k3-moonshot-ai-capex-chip-selloff), 기사
전문 확인).

시장이 반응한 이유는 단순합니다. Kimi K3는 GPU 수출 규제로 최신 가속기를
구하기 어려운 환경에서, 구세대 GPU만으로 학습됐다고 알려져 있습니다.
"최첨단 GPU 없이도 프론티어급 모델을 만들 수 있다"는 서사가 재확인되면,
빅테크의 초대형 GPU 구매 계획(capex)의 근거가 흔들립니다. Nvidia 매출은
결국 "AI를 하려면 우리 GPU가 필요하다"는 전제에 크게 의존하는데, 이
전제가 다시 한 번 도전받은 셈입니다.

기술적으로도 흥미로운 지점이 있습니다. KDA는 시퀀스 길이에 선형으로
스케일링되는 하이브리드 선형 어텐션으로, 100만 토큰급 컨텍스트에서
디코딩을 최대 6.3배 가속한다고 Moonshot은 주장합니다. 자세한 구조는
아래 [아키텍처 섹션](#아키텍처-살펴보기-moe부터-최신-attention까지)에서 다룹니다.

> **이미지 생성 프롬프트**: "A financial news infographic showing a stock
> market crash timeline, comparing 'DeepSeek Moment' (January 2025) and
> 'Kimi K3 Moment' (July 2026), with NVIDIA, AMD, TSMC stock candlestick
> charts trending downward, Chinese and American flag motifs subtly in
> background, minimalist data-journalism style, dark navy background with
> red/green candlesticks, clean sans-serif typography"

전체 가중치 공개는 2026년 7월 27일로 예정되어 있어, 이 문서 작성 시점에는
공식 기술 리포트가 아직 나오지 않은 상태입니다. 아래 아키텍처 설명은
Moonshot 블로그와 커뮤니티 분석을 기준으로 정리했습니다.

## 추론 시장으로의 전환

최근 몇 달 동안 AI 에이전트는 사내 문화를 크게 뒤흔들고 있습니다. 엘리베이터를
타도, 밥을 먹어도, 옆 부서에서도 AI 얘기로 가득합니다.

점점 더 많은 사람이 AI 에이전트를 활용하기 시작했고 토큰 소비량이 폭증하고
있습니다. 이미 사용하는 사람들도 날이 갈수록 더 많은 토큰을 소비합니다.

- 챗봇 → 에이전트로 판도가 바뀌며 토큰 사용량이 폭증했습니다.
- 에이전트 생산성 향상을 경험한 회사들이 앞다투어 온프레미스 토큰 프로덕션
  환경을 구축하고 있습니다.
- 개인도 토큰 서빙을 위한 로컬 서빙 모델 구축을 시작했습니다. 애플 실리콘,
  DGX Spark 등이 품절 대란을 겪었습니다.
- 쓸 만한 모델을 돌릴 수 있는 GPU들의 가격이 폭등했습니다. 예) RTX 5090
  200만원대 → 현재 700만원대.
- 모델 사이즈가 점점 커지는 중이라 더 큰 메모리가 필요해지고 있습니다.

> **이미지 생성 프롬프트**: "A simple upward-trending line graph titled
> 'Token consumption over time', showing two overlapping curves labeled
> 'Chatbot era' (gentle slope) and 'Agent era' (steep exponential slope)
> diverging sharply after a vertical dashed line labeled '2025', minimalist
> tech-report chart style, single accent color on white background"

## LLM을 쓰기 위해서는 얼마만큼의 메모리가 필요할까요?

Qwen3.6-27B, DeepSeek-V4-Flash, Kimi K3, GLM-5.2 — 모델 이름은 계속
바뀌지만 계산 방법은 똑같습니다. 각 모델의 실제 스펙(2026년 7월 기준,
Hugging Face 모델 카드/기술 리포트 확인)은 다음과 같습니다.

| 모델 | 구조 | 총 파라미터 | 활성 파라미터 | 컨텍스트 |
|---|---|---|---|---|
| Qwen3.6-27B | Dense (Gated DeltaNet + Attention 하이브리드) | 27B | 27B (전량) | 262K (최대 1.01M) |
| DeepSeek-V4-Flash | MoE (Hybrid Attention) | 284B | 13B | 1M |
| GLM-5.2 | MoE (DSA + IndexShare) | 744B | 약 40B | 1M |
| Kimi K3 | MoE (KDA + Stable LatentMoE) | 2.8T | 896개 전문가 중 16개 활성 (정확한 활성 파라미터 수는 미공개) | 1M |

GLM-5.2의 "약 40B"는 실제 `config.json`(`n_routed_experts: 256`,
`num_experts_per_tok: 8`, `n_shared_experts: 1`, `hidden_size: 6144`,
`num_hidden_layers: 78` 등)으로 직접 재계산해서 **같은 범위임을 확인**한
값입니다(라우팅 전문가·attention·indexer·embedding을 모두 더하면
약 44~45B가 나와 "약 40B"와 자릿수가 일치 — 정확히 40.0B로 맞아떨어지는
건 아닙니다). 총 파라미터는 역산하면 약 747B로 744B와 거의 일치합니다.
Kimi K3는 가중치가 아직 공개되지 않아(2026-07-27 예정) 같은 방식의
검증이 불가능했습니다.

일반적으로 원본 모델은 16비트(BF16)로 배포됩니다. 가속기 세대별로 네이티브
가속이 되는 정밀도도 다릅니다.

- A 시리즈(암페어) - FP16 네이티브 가속 가능 (FP16 텐서코어 자체는
  볼타부터 있었지만, 이 문서에서는 최근 세대 중 기준점으로 사용)
- H 시리즈(호퍼) - FP8 네이티브 가속 가능
- B 시리즈(블랙웰) - FP4 네이티브 가속 가능 (NVFP4)

원본 모델을 8비트로 압축했을 경우, 1B ≈ 1GB입니다. 보통 모델을 8비트로
단순 압축 시 출력 품질은 무손실 수준입니다. 작은 모델일수록 양자화에
약하고, 큰 모델일수록 강합니다.

아무튼 GLM-5.2(744B)를 FP8로 서빙하려면 단순 계산으로 약 744GB의 메모리가
필요합니다. 회사 추론 머신의 대부분을 차지하는 NVIDIA H200 그래픽카드는
한 장에 141GB의 HBM3e 메모리를 탑재하고 있습니다. 네 장이면 564GB라
부족하고, 여덟 장이면 1,128GB가 되므로 GLM-5.2 FP8 모델을 서빙할 수
있습니다.

B300(Blackwell Ultra)은 장당 288GB, 여덟 장이면 2,304GB(약 2.3TB)입니다.
Kimi K3(2.8T)를 NVFP4(4비트, 파라미터당 0.5바이트)로 양자화하면 약
1.4TB가 필요한데, H200 8장(1,128GB)으로는 부족하지만 B300 8장(2.3TB)이면
서빙할 수 있습니다.

하지만 이건 어디까지나 모델을 로드하는 데 필요한 메모리를 계산한 것뿐입니다.
실제로는 여러 오버헤드로 인해 추가 메모리를 필요로 하며, 그 중 가장
지배적인 건 KV캐시입니다.

> **이미지 생성 프롬프트**: "A horizontal bar chart comparing GPU memory
> requirements for 4 different LLMs (Qwen3.6-27B, DeepSeek-V4-Flash,
> GLM-5.2, Kimi K3) at FP8 and FP4 precision, overlaid with horizontal
> reference lines for H200 (141GB x N) and B300 (288GB x N) GPU cluster
> capacities, clean infographic style, muted blue and orange color
> palette, engineering diagram aesthetic"

## KV캐시란 뭔가요?

LLM 모델은 다음 토큰을 추론하기 위해, 지금까지 입력된 모든 토큰과의 관계를
연산해야 합니다. 따라서 이전 토큰을 처리한 결과값을 항상 테이블에 펼쳐둬야
합니다. 이런 이유 때문에 컨텍스트가 길어지면 일반적으로 KV캐시는 선형으로
증가합니다. 따라서 수십 GB의 중형 모델은 딱 그 정도의 메모리만 있으면 될
것 같지만, 컨텍스트가 길어지면 가중치를 초과하는 수준으로 KV캐시가 자라는
경우가 흔하고, 동시 요청을 처리해야 하는 경우에는 이 캐시 비용이 훨씬 더
빠르게 증가합니다.

KV캐시 크기는 대략 다음 공식으로 추정할 수 있습니다.

```
KV 캐시(byte) = 2(K,V) × batch × seq_len × layer 수 × KV head 수 × head_dim × 정밀도(byte)
```

예를 들어 Qwen3.6-27B의 Gated Attention 레이어는 KV head가 4개,
head_dim이 256입니다. 전체 64개 레이어 중 4개당 1개(16개 레이어)에만
이 구조가 있고, 나머지 3/4은 KV캐시가 필요 없는 Gated DeltaNet(선형
어텐션)입니다. 이런 하이브리드 구조는 순수 Full Attention 모델보다
KV캐시가 훨씬 적게 늘어납니다(왜 그런지는 아래
[Qwen 시리즈 섹션](#아키텍처-살펴보기-moe부터-최신-attention까지)에서
자세히 다룹니다).

반대로 DeepSeek 계열은 MLA(Multi-head Latent Attention)로 K/V를 저차원
잠재 벡터 하나로 압축해서 저장하기 때문에, 표준 GQA보다도 KV캐시가
작습니다. DeepSeek-V4는 여기서 한 단계 더 나가 Hybrid Attention(CSA+HCA)
으로, 1M 컨텍스트에서 V3.2 대비 KV캐시를 10% 수준까지 줄였다고 발표했습니다.

즉 "총 파라미터가 작다"는 것과 "KV캐시가 작다"는 것은 별개의 축이며,
최근 아키텍처 경쟁은 사실 이 KV캐시 축에서 벌어지고 있습니다.

> **이미지 생성 프롬프트**: "A line graph showing KV cache memory size (GB)
> growing with context length (0 to 1M tokens) for four attention types:
> standard Multi-Head Attention (steepest slope), GQA (moderate slope),
> MLA (shallow slope), and hybrid linear attention like Gated
> DeltaNet/KDA (nearly flat), log-scale y-axis, technical whitepaper
> diagram style, blue gradient lines with legend"

## LLM 모델의 연산: 프리필과 디코딩

LLM 모델의 연산은 크게 두 단계로 구분됩니다. 입력을 처리하는 프리필,
그리고 출력을 만드는 디코딩입니다. 근본적으로는 동일한 연산이지만
입력은 병렬 처리가 가능하므로 빠르고, 출력은 직렬로만 처리가 가능하기에
느립니다.

단적인 예로, 하나의 요청을 처리할 때, 프리필이 초당 수천 개의 토큰을
처리할 수 있다면 디코딩은 수십 토큰 정도에 머무릅니다. 따라서 1:1로
비교했을 때 디코딩이 추론 과정에서 압도적인 병목이 됩니다.

### 프리필

프리필은 왜 빠를까요? 그 이유는 다음 토큰이 이미 존재하기 때문에 이전
토큰을 참조해서 연산하는 작업을 병렬로 수행할 수 있기 때문입니다.

"오늘 점심은 사내 식당에서 먹을까요, 나가서 먹을까요?" → 한 단어가 하나의
토큰이라고 가정하겠습니다.

- 오늘 → 자기 자신만 참조하면 됩니다
- 점심은 → 오늘, 점심은 두 토큰을 참조합니다
- 사내 → 오늘, 점심은, 사내 세 토큰을 참조합니다
- ...
- 먹을까요? → 오늘, 점심은, 사내, 식당에서, 먹을까요, 나가서, 먹을까요?
  일곱 토큰을 참조합니다.

따라서 각 토큰이 어떤 토큰을 참조해야 하는지 미리 알 수 있습니다. 즉,
한 번 메모리로부터 가중치를 읽으면 전체를 한 번에 연산할 수 있습니다.
그래서 일반적으로 프리필은 GPU의 연산 능력이 지배적입니다.

### 디코딩

하지만 디코딩은 다릅니다.

- Q: 오늘 점심은 사내 식당에서 먹을까요, 나가서 먹을까요?
- A: 비도 오는데 그냥 회사에서 먹죠.

A가 예상되는 출력이라고 가정합니다.

- 비도 → 출력을 위해, 오늘, 점심은, 사내, 식당에서, 먹을까요, 나가서,
  먹을까요? 일곱 토큰을 참조합니다.
- 오는데 → 출력을 위해, 오늘, 점심은, 사내, 식당에서, 먹을까요, 나가서,
  먹을까요?, 비도 여덟 토큰을 참조합니다.

여기서 중요한 차이점은, `비도`를 만들기 전에 `오는데`를 연산할 수 없다는
겁니다.

따라서 매 토큰을 출력할 때마다 전체 모델 파라미터를 다시 읽어야 하고,
모델 사이즈가 100GB일 경우, 메모리 대역폭이 500GB/s이라면 다음 토큰
연산을 위해 메모리를 읽는 데 0.2초가 필요합니다.

즉, 초당 5개의 토큰을 생성할 수 있습니다. 병렬 처리할 방법이 없기 때문에
메모리 대역폭이 디코딩 성능을 좌우합니다.

그래서 최근 아키텍처들은 전부 이 디코딩 병목을 어떻게 낮추느냐를 놓고
경쟁합니다. MoE로 매 토큰마다 읽어야 하는 파라미터 양 자체를 줄이거나,
KV캐시를 압축해서 메모리 대역폭 소모를 줄이는 두 방향이 대표적입니다.
아래에서 각 모델이 택한 구체적인 해법을 살펴봅니다.

> **이미지 생성 프롬프트**: "A side-by-side technical diagram: left panel
> shows 'Prefill' with parallel token processing arrows converging into a
> single GPU compute block (labeled 'compute-bound'), right panel shows
> 'Decoding' with a sequential token generation loop and repeated
> memory-read arrows (labeled 'memory-bandwidth-bound'), clean minimalist
> engineering illustration, orange for compute-bound, blue for
> memory-bound"

## 아키텍처 살펴보기: MoE부터 최신 Attention까지

### DeepSeek R1 - Mixture of Experts

2025년 1월, DeepSeek R1이 출시되며 미국 증시가 뒤흔들렸습니다("딥시크
모먼트"). 반도체 칩 제재로 열악한 환경에서 미국의 프론티어 모델들과
비견되는 추론 모델을 발표했기 때문입니다.

인프라 투자에 천문학적 금액을 쏟아붓지 않아도 고성능 모델을 만들 수 있는
게 아닌가, 사람들이 의심하게 된 거죠. 심지어 R1은 비슷한 사이즈의 모델보다
추론 속도가 압도적으로 빨랐습니다.

R1은 별도 아키텍처가 아니라 **DeepSeek-V3의 구조를 그대로 이어받아**
강화학습으로 추론 능력을 얹은 모델입니다. 그 비밀은 V3와 동일한 Mixture
of Experts(MoE) 구조입니다. MoE란 전체 파라미터 중 일부만 사용하는
기법입니다. 다음 토큰을 출력할 때, 모델에서 입력을 보고 어떤 전문가를
활성화할지 선택한 후 해당 전문가만 활성화하는 겁니다.

실제 수치로 보면 DeepSeek-V3/R1(671B 총 파라미터, 37B 활성 파라미터)은
활성 파라미터가 총 파라미터의 약 5.5%에 불과합니다.

### Qwen 시리즈 - Gated DeltaNet

Qwen3.6-27B는 64개 레이어를 4개씩 묶어, 3개는 Gated DeltaNet(선형
어텐션), 1개는 Gated Attention(GQA, Q 24개/KV 4개 헤드, head_dim 256)
으로 구성합니다. 즉 대부분의 레이어가 시퀀스 길이에 선형으로
스케일링되는 연산을 쓰고, 4개 중 1개만 기존 방식의 이차(quadratic)
연산을 유지해서 품질(장기 의존성 포착)과 효율(연산량/KV캐시) 사이에서
균형을 잡습니다.

#### 왜 Gated DeltaNet은 KV캐시가 필요 없나요?

Full Attention은 매 토큰마다 지금까지 나온 모든 토큰과 다시 내적을
계산해야 해서, 과거 토큰의 K/V를 전부 저장해뒀다 재사용합니다 —
그래서 컨텍스트가 길어지면 저장량이 선형으로 늘어납니다.

Gated DeltaNet은 이 방식 대신 RNN처럼 토큰을 하나씩 순서대로 처리하며,
`head_dim × head_dim` 크기의 고정된 "상태(state)" 행렬 하나만 계속
업데이트합니다. 매 토큰마다 하는 일이 상수 시간이라 전체 연산량이
O(n) — 그래서 "선형 어텐션"이라 부릅니다.

이름의 "Delta"는 그냥 누적하는 게 아니라 "고쳐 쓰는" 방식이라는
뜻입니다. 순수 선형 어텐션은 outer product를 그냥 계속 더하지만
(`S_t = S_{t-1} + v_t k_t^T`), DeltaNet은 새 값을 쓰기 전에 같은
키로 이미 저장된 옛 값을 먼저 지우고 그 자리에 새 값을 씁니다
(`S_t = S_{t-1}(I − β_t k_t k_t^T) + β_t v_t k_t^T`). 이건 매 스텝마다
연상 기억의 오차를 줄이는 온라인 경사하강 한 스텝과 같아서, 단순
누적보다 정보 검색(retrieval) 성능이 좋습니다. Gated DeltaNet은
여기에 Mamba-2 스타일 감쇠(decay) 게이트를 더해서 오래된 정보를
능동적으로 잊게 만듭니다 — 그래서 "Gated"입니다.

**단, 컨텍스트 길이에 비례한 메모리 증가가 완전히 없어지는 건
아닙니다.** Qwen3.6-27B에서 실제로 컨텍스트 길이에 비례해 커지는
KV캐시는 64개 레이어 중 **Gated Attention 16개 레이어의 것뿐**입니다.
나머지 48개 DeltaNet 레이어는 각자 고정 크기 상태만 유지하므로
컨텍스트 길이와 무관합니다. 비유하면, DeltaNet 레이어는 그 시점까지의
과거를 고정 크기로 압축한 요약을 들고 있고, Attention 레이어는 그
시점에서 과거 토큰 하나하나에 정확히 접근할 수 있는 원본 기록(KV캐시)을
들고 있는 셈입니다. 전체 모델의 KV캐시는 이 16개 레이어분만 존재하므로,
64개 레이어 전체가 Full Attention인 모델보다 약 1/4 수준으로 줄어듭니다
(완전히 0이 되는 건 아닙니다).

Kimi K3의 KDA(Kimi Delta Attention)도 같은 원리이되, 감쇠를 헤드당
스칼라가 아니라 채널별로 더 세밀하게 조절하는 버전입니다.

같은 세대에 Qwen3.6-35B-A3B라는 MoE 버전도 있어서, 같은 계열 안에서
Dense(27B 전량 활성) vs MoE(35B 중 3B 활성)를 직접 비교하기 좋은
사례입니다.

### DeepSeek V4 - Hybrid Attention (CSA + HCA)

DeepSeek-V4는 Compressed Sparse Attention(CSA)과 Heavily Compressed
Attention(HCA)을 함께 쓰는 하이브리드 구조를 도입했습니다. **Pro
모델 기준**으로 V3.2 대비 100만 토큰 컨텍스트에서 토큰당 추론
FLOPs는 27%, KV캐시는 10% 수준까지 줄었다고 밝혔습니다(Flash에도
동일 비율이 적용되는지는 원문에 명시되지 않았습니다). 여기에
Manifold-Constrained Hyper-Connections(mHC, 잔차 연결 안정화)와
Muon 옵티마이저(학습 안정성/수렴 속도 개선)를 더했습니다.
Flash(284B 총/13B 활성)와 Pro(1.6T 총/49B 활성) 두 버전이 있습니다.

### Kimi K3 - Kimi Delta Attention (KDA)

Kimi K3는 세 가지 새 요소를 조합합니다.

1. **Kimi Delta Attention(KDA)**: 시퀀스 길이에 선형으로 스케일링하는
   하이브리드 선형 어텐션. 100만 토큰 컨텍스트에서 디코딩을 최대 6.3배
   가속한다고 발표.
2. **Attention Residuals**: 길이 축이 아니라 깊이(레이어) 축에서 이전
   레이어의 표현을 선택적으로 재사용.
3. **Stable LatentMoE**: 896개 전문가 중 16개만 활성화하는 초고희소
   MoE. Quantile Balancing이라는 방식으로 라우터 점수 분포에서 바로
   전문가 배분을 유도해, 기존처럼 민감한 보조 손실 하이퍼파라미터를
   튜닝할 필요를 없앴습니다.

공식 기술 리포트는 2026년 7월 27일 공개 예정이라, 정확한 활성
파라미터 수치는 아직 확인되지 않았습니다. 위 내용은 Moonshot 블로그와
커뮤니티 분석을 기준으로 정리했습니다.

> **이미지 생성 프롬프트**: "A comparative architecture diagram showing 4
> LLM attention lineages side by side as vertical stacked-block diagrams:
> DeepSeek (MLA to DSA to CSA+HCA), Qwen (GQA to Gated DeltaNet hybrid),
> Kimi (MLA to Kimi Delta Attention), GLM (DSA to DSA+IndexShare), each
> block labeled with year, technical research-paper figure style, black
> and white line art with one accent color per lineage"

## Quantization

### Model 양자화

**균일 양자화** (모델 전체를 같은 비트로):

- **BF16 / FP16**: 원본 배포 정밀도. 지수부 비트 수만 다릅니다
  (BF16이 더 넓은 다이나믹 레인지, FP16이 더 높은 정밀도).
- **FP8** (E4M3/E5M2): 호퍼(H100/H200)부터 네이티브 가속. 대부분
  무손실급 품질.
- **FP4 계열**: 블랙웰부터 네이티브 가속.
  - **MXFP4**: OCP(Open Compute Project) 표준. 블록 크기 32, 스케일
    factor는 E8M0(8비트 지수). GPT-OSS가 이 포맷으로 배포되며
    대중화됐고, 가중치만 양자화(weight-only)하면 암페어 이후 GPU에서도
    돌아갑니다.
  - **NVFP4**: NVIDIA 자체 포맷. 블록 크기 16(더 촘촘), 스케일 factor는
    FP8 E4M3(더 정밀). 같은 4비트여도 MXFP4보다 오차가 작다고 보고되지만,
    네이티브 가속은 블랙웰 전용입니다.

**저비트 보정/캘리브레이션 기법** (전체는 여전히 균일한 비트지만, 어떤
채널을 더 정확히 보존할지 스케일링·최적화로 정확도를 끌어올리는
방식 — 아래 두 방법 모두 결과물은 예컨대 INT4처럼 균일한 비트입니다):

- **AWQ**(Activation-aware Weight Quantization): 활성값 크기를 기준으로
  중요한 가중치 채널을 찾아 그 채널만 스케일을 보정, 나머지를 4비트로
  눌러도 손실을 최소화합니다.
- **AutoRound**: 인텔이 낸 방법으로, 블록 단위로 라운딩 값 자체를
  최적화해서 캘리브레이션 오차를 줄입니다. Qwen3.6-27B 커뮤니티 INT4
  양자화에도 실제로 쓰이고 있습니다.

**혼합 정밀도** (레이어마다 실제 비트 수 자체가 다름 — 위 두 방법과는
범주가 다릅니다):

- **Unsloth Dynamic 2.0**: 레이어별 민감도를 측정해서, 민감한 레이어는
  높은 비트로 남기고 나머지(주로 MoE 전문가 FFN)는 최대한 눌러 평균
  비트를 낮추는 방식. DeepSeek V3.1을 평균 3비트로 눌러도 원본급
  벤치마크를 유지했다고 보고됐습니다.

### KV 캐시 양자화 (균일 양자화만 존재)

KV캐시는 모델 가중치와 달리 매 요청마다 새로 생성되는 데이터라, 레이어별로
다른 비트를 미리 정해두는 혼합 정밀도 방식이 자리잡지 못했고 균일
양자화만 씁니다.

- **단순 양자화**: per-tensor/per-channel INT8·INT4. 구현은 쉽지만
  K/V에 낀 이상치(outlier) 채널 때문에 4비트 이하로 내리면 품질이
  급격히 무너집니다.
- **QuaRot** (회전 기반 양자화): 양자화 전에 하다마드(Hadamard) 회전을
  곱해 이상치를 여러 채널로 분산시킨 뒤 양자화합니다. 가중치·활성값·
  KV캐시를 통째로 4비트(W4A4KV4)까지 내려도 정확도 99% 이상을 유지한다고
  보고됐습니다.
- **TurboQuant** (Google Research, [arXiv:2504.19874](https://arxiv.org/abs/2504.19874),
  ICLR 2026): 벡터를 회전시킨 뒤 각 좌표가 베타 분포를 따른다는 성질을
  이용해 좌표별 최적 Lloyd-Max 양자화기를 설계하고, 여기에 1비트
  Johnson-Lindenstrauss(QJL) 보정을 더해 내적(attention score) 계산의
  편향을 없앱니다. 논문에 따르면 채널당 3.5비트에서는 품질 손실이
  없고, 2.5비트까지 내려도 손상이 미미합니다.

> **이미지 생성 프롬프트**: "A technical comparison diagram of
> quantization number formats (BF16, FP8, MXFP4, NVFP4) shown as
> bit-layout boxes with sign/exponent/mantissa segments color-coded,
> below them a small block-diagram showing block size difference (32 vs
> 16 elements per scale factor), clean semiconductor datasheet
> illustration style, white background with color-coded bit fields"

## 부록: NVIDIA 데이터센터 GPU 로드맵

Nvidia는 세대를 과학자 이름으로 명명합니다: Hopper → Blackwell → **Rubin** →
**Feynman**. 이 문서 본문에서 쓴 H200/B300은 각각 Hopper/Blackwell
세대입니다.

| 세대 | 시기 | 공정 | 메모리 | 대역폭 | 비고 |
|---|---|---|---|---|---|
| H100 | 2022 | TSMC 4N | HBM3 80GB | 3.35TB/s | |
| H200 | 2024 | TSMC 4N | HBM3e 141GB | 4.8TB/s | 이 문서 메모리 계산에 쓴 세대 |
| B200 (Blackwell) | 2024 | TSMC 4NP | HBM3e 192GB | 8TB/s | |
| B300 (Blackwell Ultra) | 2025 | TSMC 4NP | HBM3e 288GB | 8TB/s | 이 문서 메모리 계산에 쓴 세대 |
| Rubin R100 | 2026 | TSMC N3 | HBM4 288GB | 22TB/s | Vera CPU와 페어링 |
| Rubin Ultra | 2027 | TSMC N3 refine | HBM4 확장 | - | 밀도 향상, 테이프아웃 단계 |
| Feynman | 2028 | TSMC A16(1.6nm) | 커스텀 HBM(1TB+ 보고) | - | 3D 다이 스태킹, 광 NVLink, LP40 LPU |

### Vera Rubin 플랫폼

Rubin은 GPU 단독이 아니라 "Vera CPU + Rubin GPU" 통합 플랫폼으로
출시됩니다. Grace Blackwell(GB200)과 같은 명명 패턴입니다. 미국
천문학자 베라 루빈(암흑물질 증거 발견)의 이름을 땄습니다.

| 구성 요소 | 스펙 |
|---|---|
| Rubin GPU (R100) | TSMC 3nm급, 다이 2개·336B 트랜지스터, HBM4 288GB/22TB/s, NVFP4 추론 50 PFLOPS |
| Vera CPU | Olympus 코어 88개(Armv9.2), 176 스레드, LPDDR5X 최대 1.5TB, 227B 트랜지스터, Grace 후속 |
| NVLink-C2C | CPU-GPU 연결 1.8TB/s (PCIe Gen6의 7배) |
| Vera Rubin Superchip | Vera CPU 1개 + Rubin GPU 2개, 합산 NVFP4 추론 약 100 PFLOPS |

HBM4에서 용량은 B300과 같은 288GB지만, 대역폭이 8TB/s → 22TB/s로 거의
3배 뛰는 게 핵심입니다. 이 문서에서 반복한 "디코딩=메모리 대역폭 병목"
논리대로면, 같은 모델을 Rubin에서 돌리면 디코딩 처리량이 유의미하게
좋아질 여지가 있습니다.

Feynman의 **LP40 LPU**(Groq 인수/라이선스 기반)는 디코딩 전용
가속기로, 온칩 SRAM을 수백MB급으로 늘려 결정론적·초저지연 디코딩을
노립니다. 이 문서에서 반복적으로 다룬 디코딩 병목을 하드웨어 레벨에서
직접 겨냥한 시도입니다.

> **이미지 생성 프롬프트**: "A technical roadmap timeline infographic
> showing NVIDIA datacenter GPU generations (Hopper H200, Blackwell B300,
> Rubin R100, Rubin Ultra, Feynman) from 2024 to 2028, with a rising bar
> chart of HBM memory bandwidth (TB/s) below each generation, clean
> semiconductor industry poster style, dark background with neon
> blue/green accent bars"

이 로드맵은 GTC 2025/2026 발표 및 Wccftech·Tom's Hardware·SemiAnalysis
보도를 종합한 3rd-party 콘텐츠(reddit, tech-insider.org,
thundercompute.com, vrlatech.com) 기반입니다. NVIDIA 공식 스펙시트를
직접 fetch하지는 못했고, 여러 독립 소스 간 수치가 일치하는 것만
확인했습니다. 특히 Rubin Ultra/Feynman은 아직 확정 스펙이 아닙니다.

## 참고 자료

**직접 fetch해서 원문/원본 데이터를 확인한 자료** (신뢰도 높음):

- [Qwen/Qwen3.6-27B (Hugging Face 모델 카드)](https://huggingface.co/Qwen/Qwen3.6-27B)
- [deepseek-ai/DeepSeek-V4-Flash (Hugging Face 모델 카드)](https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash)
- [zai-org/GLM-5.2 (Hugging Face 모델 카드 + config.json 원본)](https://huggingface.co/zai-org/GLM-5.2)
- [GLM-5: from Vibe Coding to Agentic Engineering (arXiv:2602.15763)](https://arxiv.org/abs/2602.15763)
- [TurboQuant: Online Vector Quantization with Near-optimal Distortion Rate (arXiv:2504.19874, Google Research)](https://arxiv.org/abs/2504.19874)
- [Kimi K3's 2.8T-Parameter Launch Puts AI Capex Story on Trial (TFTC, 기사 전문 확인)](https://www.tftc.io/kimi-k3-moonshot-ai-capex-chip-selloff)

**검색 스니펫/2차 소스로만 확인한 자료** (교차검증 약함, 원문 미확인):

- [QuaRot: Outlier-Free 4-Bit Inference in Rotated LLMs (arXiv:2404.00456)](https://arxiv.org/abs/2404.00456) — abstract만 확인, 잘 알려진 논문이라 신뢰도는 높은 편
- [Unsloth Dynamic 2.0 GGUFs 공식 문서](https://unsloth.ai/docs/basics/unsloth-dynamic-2.0-ggufs) — 검색 스니펫으로만 확인
- Kimi K3 아키텍처 세부사항(KDA, Attention Residuals, Stable LatentMoE) — `kimi.com/blog/kimi-k3`(403), `huggingface.co/moonshotai/Kimi-K3`(401) 둘 다 직접 접근 실패. kie.ai, k3-kimi.com 등 3rd-party 재구성에 의존. 가중치·기술 리포트가 2026-07-27 공개되면 재검증 필요.
- NVFP4/MXFP4 블록 크기, NVIDIA B300 스펙 — reddit/3rd-party 블로그 기반, NVIDIA 공식 스펙시트 미확인
- Rubin/Rubin Ultra/Feynman 로드맵, Vera Rubin 플랫폼 스펙 — reddit, tech-insider.org, thundercompute.com, vrlatech.com 종합. GTC 발표 자체는 실제 있었으나 세부 수치(트랜지스터 수, PFLOPS 등)는 NVIDIA 공식 자료로 재확인 필요

위 자료는 2026년 7월 22일 기준으로 확인했습니다.
