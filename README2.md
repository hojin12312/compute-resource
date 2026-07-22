### 6.3 DeepSeek V4 - Hybrid Attention (CSA + HCA)

> "이거보다 더 싸게 만들 수 있어?"

DeepSeek-V4는 Compressed Sparse Attention(CSA)과 Heavily Compressed
Attention(HCA)을 함께 쓰는 하이브리드 구조를 도입했습니다. 둘 다
"과거 토큰을 몇 개씩 묶어 요약본 하나로 압축한다"는 아이디어를
공유하지만, 압축 강도와 그 이후 보는 방식이 다릅니다. 책 한 권을
다시 읽어야 하는 상황에 비유하면:

- **CSA**: 페이지 4장을 한 장짜리 요약으로 압축해 둔 다음, 그 요약본들
  중에서 지금 필요할 것 같은 몇 개만 골라(sparse) 자세히 읽습니다.
  압축은 약하게, 선택은 좁게 하는 방식입니다.
- **HCA**: 훨씬 더 세게 압축해서(책 한 챕터를 한 문장으로) 요약본
  개수 자체를 확 줄여버립니다. 대신 남은 요약본 수가 적으니, 골라
  읽지 않고 전부(dense) 다 읽어도 부담이 없습니다.

즉 CSA는 "많이 남기고 그중 일부만 보기", HCA는 "적게 남기고 전부
보기"로 서로 다른 지점에서 균형을 잡는 셈입니다. DeepSeek-V4는 이
둘을 레이어마다 섞어 씁니다. **Pro 모델 기준**으로 V3.2 대비 100만
토큰 컨텍스트에서 토큰당 추론 FLOPs는 27%, KV캐시는 10% 수준까지
줄었다고 밝혔습니다(Flash에도 동일 비율이 적용되는지는 원문에
명시되지 않았습니다). 여기에
Manifold-Constrained Hyper-Connections(mHC, 잔차 연결 안정화)와
Muon 옵티마이저(학습 안정성/수렴 속도 개선)를 더했습니다.
Flash(284B 총/13B 활성)와 Pro(1.6T 총/49B 활성) 두 버전이 있습니다.

이렇게 아낀 연산·메모리는 실제 API 가격에도 그대로 드러납니다[^ds-v4-pricing].

| 모델 | 입력(캐시 미스) | 캐시 히트 | 출력 |
|---|---|---|---|
| V4 Flash | $0.14 | $0.0028 | $0.28 |
| V4 Pro | $0.435 | $0.003625 | $0.87 |

(1M 토큰당, USD)

캐시 히트 가격은 출력 대비 100배(Flash) ~ 240배(Pro) 저렴한 수준입니다.
이렇게까지 저렴할 수 있는 이유는 캐시 히트가 정말 메모리를 거의
소비하지 않기 때문입니다 — 이미 계산해 둔 KV캐시를 그대로 재사용할
뿐이라 새로 읽어야 할 메모리가 거의 없고, [5.3절](#53-kv-캐시-히트)에서
설명한 것처럼 프리필 연산 자체도 새로 추가된 부분만큼만 필요합니다.

### 6.4 Kimi K3 - Kimi Delta Attention (KDA)

Kimi K3는 세 가지 새 요소를 조합합니다.

1. **Kimi Delta Attention(KDA)**: [Gated DeltaNet](#62-qwen3-next-80b-a3b---gated-deltanet)과
   같은 원리로, 과거 전체를 고정 크기 요약 노트 하나로 압축해 시퀀스
   길이에 선형으로 스케일링합니다. 100만 토큰 컨텍스트에서 디코딩을
   최대 6.3배 가속한다고 발표. 다만 이 요약 노트 방식은 "이전 대화를
   그대로 캐시해 재사용"하는 기존 prefix caching과 충돌해, Moonshot이
   직접 vLLM에 KDA용 캐싱 구현을 기여했습니다.
2. **Attention Residuals**: 옆 레이어가 아니라 몇 단계 전 레이어가
   만든 표현을 필요할 때 다시 꺼내 씁니다. 매번 새로 메모하는 대신,
   전에 써둔 메모 중 필요한 것을 골라 다시 보는 셈입니다. 구조
   자체는 서빙 시에도 그대로 쓰이지만, Moonshot이 밝힌 수치는
   학습(training) 단계 기준으로 학습 효율을 약 25% 끌어올리면서
   추가 연산 비용은 2% 미만이라는 내용입니다.
3. **Stable LatentMoE**: 896개 전문가 중 16개만 활성화하는 초고희소
   MoE. Quantile Balancing은 "상위 몇 %까지 뽑는다"는 등급 컷을
   라우터 점수 분포에서 바로 정하는 방식으로, 지원자 수(점수 분포)가
   매번 달라져도 합격선을 자동으로 다시 긋는 것과 비슷합니다. 이
   덕분에 기존처럼 민감한 보조 손실 하이퍼파라미터를 튜닝할 필요를
   없앴습니다.

그 외에도 Per-Head Muon(헤드별 독립 옵티마이저), SiTU, Gated MLA를
함께 써서 K2 대비 약 2.5배의 스케일링 효율을 얻었다고 발표했습니다.

서빙 관점에서 더 중요한 지점은, K3가 Supervised Fine-Tuning
(SFT)[^sft] 단계부터 Quantized Aware Training (QAT)[^qat]를 거쳐
**가중치는 MXFP4, 활성값은 MXFP8**로 네이티브 배포된다는 점입니다.
사후 PTQ보다 품질 저하가 작을 것으로 기대되며,
Moonshot은 서빙에 가속기 64장 이상의 슈퍼노드 구성을 권장합니다.

> **이미지 생성 프롬프트**: "A technical diagram showing three panels in a
> row for Kimi K3: panel 1 a compressed notebook labeled 'Kimi Delta
> Attention' compressing a long timeline into a fixed-size summary, panel
> 2 a hand reaching back several layers to retrieve an old note labeled
> 'Attention Residuals', panel 3 a grading curve with a cutoff line moving
> dynamically labeled 'Stable LatentMoE', clean minimalist engineering
> illustration, one accent color per panel"

> **이미지 생성 프롬프트**: "A comparative architecture diagram showing 3
> LLM attention lineages side by side as vertical stacked-block
> diagrams: DeepSeek (dense attention evolving to CSA+HCA hybrid), Qwen3-
> Next (GQA evolving to Gated DeltaNet hybrid), Kimi K3 (dense attention
> evolving to Kimi Delta Attention + Attention Residuals + Stable
> LatentMoE, with a small MXFP4/MXFP8 native-precision tag), each block
> labeled with model name, technical research-paper figure style, black
> and white line art with one accent color per lineage"

## 7. Quantization

### 7.1 Model 양자화

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

위 블록 크기·스케일 포맷 수치는 NVIDIA 공식 기술 블로그("Introducing
NVFP4 for Efficient and Accurate Low-Precision Inference",
developer.nvidia.com, 직접 fetch 확인)의 비교표 기준입니다.

위 두 FP4 포맷은 보통 학습이 끝난 모델에 사후(post-training)로
적용됩니다. 반면 Kimi K3는 Supervised Fine-Tuning (SFT) 단계부터
Quantized Aware Training (QAT)로 MXFP4/MXFP8 정밀도에 맞춰 학습해
품질 저하를 더 줄이는 접근을 씁니다([Kimi K3 아키텍처
섹션](#64-kimi-k3---kimi-delta-attention-kda) 참고).

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

### 7.2 KV 캐시 양자화 (균일 양자화만 존재)

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

> **이미지 생성 프롬프트**: "A technical diagram of a KV cache tensor
> shown as a grid of channels, with a few outlier channels highlighted in
> red spiking far above the rest; a before/after pair shows the raw grid
> on the left and, on the right, the same grid after a Hadamard rotation
> smooths the outliers evenly across channels before being compressed
> into small 4-bit blocks, clean minimalist engineering illustration,
> red accent for outliers and blue for the rotated/quantized result"

### 7.3 그래서 양자화하면 뭐가 좋나요?

양자화는 이 문서에서 계속 다룬 프리필/디코딩 병목 구조에 세 가지
방향으로 직접 이득을 줍니다.

- **연산 자체가 빨라집니다.** FP8·MXFP4·NVFP4처럼 GPU가 네이티브로
  가속하는 포맷이라면, 같은 텐서코어로 더 많은 연산을 동시에 처리할
  수 있습니다. 프리필처럼 연산량(compute)이 지배적인 단계에서 특히
  체감됩니다.
- **메모리 사용량이 줄어 디코딩이 빨라집니다.** 디코딩은 매 토큰마다
  모델 전체를 메모리에서 다시 읽어야 하는 메모리 대역폭 병목[^mem-bandwidth]
  이었죠. 가중치를 16비트에서 4비트로 낮추면 읽어야 할 바이트 수가
  1/4로 줄어드니, 같은 대역폭으로 더 짧은 시간에 다 읽을 수 있고 —
  곧 초당 생성 가능한 토큰 수가 늘어납니다.
- **KV캐시가 줄면 동시성이 늘어납니다.** KV캐시 양자화(7.2)로 캐시가
  차지하는 메모리가 줄면, 같은 GPU 메모리로 더 많은 요청의 KV캐시를
  동시에 올려둘 수 있습니다. 즉 배치 크기(동시에 처리하는 요청 수)를
  키울 수 있어, 서버 전체의 처리량(throughput)이 늘어납니다.

## 8. AI 소버린

> "더 이상 AI는 모두의 것이 아니라, 각자 자신의 것을 만들어야 하는
> 국가 전략자산이 되었다"

### 8.1 국가 단위의 AI 소버린

지금까지 살펴본 모델·아키텍처 이야기는 결국 "GPU가 부족한 환경에서도
어떻게 쓸 만한 모델을 만드느냐"는 질문으로 돌아옵니다. 그리고 이
질문 자체가 순수한 기술 문제가 아니라 국가 간 규제와 대응의 결과라는
게 이 장의 주제입니다.

**미국의 GPU 수출 규제**: 미국은 프론티어급 GPU(Blackwell 계열)의
중국 수출을 막고 있고, 그보다 낮은 성능의 H20이나 성능을 깎은 H200도
케이스별 라이선스와 관세를 거쳐야만 통과됩니다. 2025년 5월에는
전 세계를 3단계로 나눠 접근을 차등화하려던 "AI Diffusion Rule"이
폐기되고, 대신 국가별 개별 협상 방식으로 바뀌었습니다 — UAE·사우디처럼
미국과 우호적인 국가는 대량의 최신 GPU를 확보하는 반면, 그렇지 못한
국가는 접근이 불투명해지는 구조입니다.

**중국의 대응 — 자국산 칩과 자국산 모델**: 수입이 막히자 중국은
Huawei의 Ascend 칩(910C, 950 시리즈)으로 자국 내 학습·추론 인프라를
자체 조달하는 방향으로 움직였습니다. 이 문서에서 다룬 DeepSeek·Kimi·
Qwen·GLM 등 중국 오픈웨이트 모델들이 하드웨어 제약 속에서 연산량·
메모리를 극단적으로 아끼는 아키텍처(MoE, 선형 어텐션, KV캐시 압축,
네이티브 저비트 양자화)를 발전시켜온 배경이 바로 여기 있습니다 —
"싸게, 그러나 뛰어나게"[^ai-sovereign]는 애초에 선택이 아니라 제재
환경이 강제한 방향이었던 셈입니다.

**모델 자체도 전략 자산이 됩니다**: 2026년 7월, 로이터는 중국 정부가
Alibaba·ByteDance·Z.ai(Zhipu) 등과 만나 자국의 최신 AI 모델(공개
예정 오픈웨이트 모델 포함)의 해외 접근을 제한하는 방안을 논의했다고
보도했습니다. 아직 확정된 규제는 아니지만, "가장 앞선 모델은 국내
전용으로 묶어두고 그보다 뒤처진 모델만 공개한다"는 계층화 방안이
검토되고 있습니다. 미국도 마찬가지로 특정 프론티어 모델의 배포를
국가안보 사유로 일시 중단하거나, 신뢰할 수 있는 기관에만 접근을
허용하는 조치를 취한 바 있습니다. 즉 GPU 수출을 막는 것과 마찬가지로,
이제는 모델(가중치) 자체의 국외 유출도 통제 대상이 되고 있습니다.

결국 하드웨어(GPU 수출 규제) → 모델 설계(제약 속 최적화) → 모델
그 자체(배포 통제)까지, AI를 둘러싼 통제의 범위가 계속 넓어지고
있습니다. 자국 GPU도, 자국 모델도 없는 국가는 이 흐름에서 점점 더
불리한 위치에 놓이게 되며, 이것이 "모든 나라가 자기 자신의 AI를
만들어야 하는 시대"라는 이 장의 문장으로 이어집니다.

> **이미지 생성 프롬프트**: "A world map infographic showing AI compute
> sovereignty, with countries color-coded by GPU/chip access tier and
> small icons for domestic AI chips (US, China, Middle East highlighted),
> minimalist geopolitical data-journalism style, muted color palette
> with one accent color per bloc"

### 8.2 개인이 소유한 AI

같은 논리를 국가 단위에서 개인 단위로 좁혀도 방향은 똑같습니다.
2장에서 다뤘듯 개인도 애플 실리콘·DGX Spark 같은 로컬 서빙 하드웨어를
품절 대란 속에 사들이고 있는데, 이건 단순히 API 비용을 아끼려는
움직임이 아닙니다. 내 대화 기록, 내 파일, 내 업무 맥락을 계속 담아
두고 나만을 위해 돌아가는 개인 비서(자비스)를 누구나 필요로 하게 될
거라는 예상 때문입니다 — 그리고 그런 비서를 외부 서버가 아니라 내가
소유한 하드웨어 위에서 돌리고 싶다면, 결국 개인 차원의 소버린 문제와
마주하게 됩니다.

문제는 국가와 달리 개인은 자체 칩을 설계할 수 없다는 점입니다.
그래서 개인의 AI 소버린은 국가 간 GPU 배분(어느 나라가 얼마나 많은
GPU를 확보하느냐)과, 모델을 얼마나 적은 하드웨어로 돌릴 수 있게
설계하느냐(이 문서에서 계속 다룬 MoE·선형 어텐션·양자화)에 그대로
종속됩니다. 즉 "우리 모두에게 자비스가 필요하다"는 요구가 현실이
되려면, 국가 단위의 자립 경쟁과 모델의 경량화 경쟁이 먼저 해결돼야
하는 셈입니다.

그리고 이 모든 흐름 — 국가마다 자국 모델, 개인마다 자기만의 상시
비서 — 은 결국 토큰 소모를 지금보다 훨씬 더 가속화합니다. 이 문서
2장에서 다룬 토큰 사용량 폭증은 끝이 아니라 시작이었던 셈이고, 그만큼
더 많은 전력·GPU·메모리가 필요해질 가능성이 높습니다.

## 9. 부록: NVIDIA 데이터센터 GPU 로드맵

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

### 9.1 Vera Rubin 플랫폼

Rubin은 GPU 단독이 아니라 "Vera CPU + Rubin GPU" 통합 플랫폼으로
출시됩니다. Grace Blackwell(GB200)과 같은 명명 패턴입니다. 미국
천문학자 베라 루빈(암흑물질 증거 발견)의 이름을 땄습니다. Vera Rubin
NVL72는 2026-06-01 GTC 타이베이에서 양산 개시가 발표됐고, 아래 표는
NVIDIA 공식 스펙 페이지(`nvidia.com/en-us/data-center/vera-rubin-nvl72`)를
직접 fetch해서 확인한 수치입니다.

| 구성 단위 | GPU 메모리·대역폭 | NVFP4 추론 | NVLink |
|---|---|---|---|
| Rubin GPU 1개 | HBM4 288GB / 22TB/s | 50 PFLOPS | 3.6TB/s (6세대) |
| Vera Rubin Superchip (Vera CPU 1 + Rubin GPU 2) | HBM4 576GB / 44TB/s | 100 PFLOPS | 7.2TB/s |
| Vera Rubin NVL72 (Vera CPU 36 + Rubin GPU 72) | HBM4 20.7TB / 1,580TB/s | 3,600 PFLOPS | 260TB/s (스위치) |

Vera CPU는 커스텀 Olympus 코어 88개(Armv9 호환)로 NVL72 전체 CPU
코어 수는 3,168개(88×36), NVLink-C2C(CPU-GPU 연결) 대역폭은 Superchip
기준 1.8TB/s입니다. 위 수치는 "Preliminary information — 확정 전
변경 가능"이라는 각주가 달려 있어, 최종 출하 시 일부 조정될 수
있습니다.

HBM4에서 용량은 B300과 같은 288GB지만, 대역폭이 8TB/s → 22TB/s로 거의
3배 뛰는 게 핵심입니다. 이 문서에서 반복한 "디코딩=메모리 대역폭 병목"
논리대로면, 같은 모델을 Rubin에서 돌리면 디코딩 처리량이 유의미하게
좋아질 여지가 있습니다.

Rubin 플랫폼에는 GPU 외에도 추론 전용 가속기 **Groq 3 LPU**가
포함됩니다(Feynman의 LP40이 아니라 Rubin 세대부터 이미 편입).
NVIDIA 공식 페이지에 따르면 Groq 3 LPX 랙은 LPU 256개, SRAM
128GB(칩당이 아니라 랙 합산), 메모리 대역폭 40PB/s, 랙 내부 스케일업
대역폭 640TB/s로, Vera Rubin NVL72와 함께 배치 시 와트당 추론
성능 35배·1B+ 파라미터 모델 기준 매출 기회 최대 10배(Blackwell 대비)를
표방합니다. 온칩 SRAM 기반 결정론적·초저지연 디코딩이라는 이 문서의
반복 주제(디코딩=메모리 대역폭 병목)를 하드웨어 레벨에서 직접 겨냥한
설계입니다. Feynman의 LP40 LPU는 그 다음 세대 버전으로 보입니다.

> **이미지 생성 프롬프트**: "A technical roadmap timeline infographic
> showing NVIDIA datacenter GPU generations (Hopper H200, Blackwell B300,
> Rubin R100, Rubin Ultra, Feynman) from 2024 to 2028, with a rising bar
> chart of HBM memory bandwidth (TB/s) below each generation, and
> starting at the Rubin generation a second smaller icon track showing
> the companion 'Groq 3 LPU' inference accelerator riding alongside the
> GPU bars, clean semiconductor industry poster style, dark background
> with neon blue/green accent bars"

Rubin R100/Vera Rubin NVL72 항목은 NVIDIA 공식 스펙 페이지로 직접
확인했습니다(위 "Vera Rubin 플랫폼" 절 참고). H100/H200/B200/B300과
Rubin Ultra/Feynman은 GTC 2025/2026 발표 및 Wccftech·Tom's
Hardware·SemiAnalysis 보도를 종합한 3rd-party 콘텐츠(reddit,
tech-insider.org, thundercompute.com, vrlatech.com) 기반으로, 여러
독립 소스 간 수치가 일치하는 것만 확인했습니다. 특히 Rubin
Ultra(2027년 하반기 예정, Kyber 랙, GPU당 NVFP4 약 100 PFLOPS·HBM4e
1TB·대역폭 약 32TB/s 보도)와 Feynman(2028년, LP40 메모리·Rosa
CPU·BlueField-5 언급)은 NVIDIA가 세부 스펙을 아직 공식 공개하지
않아 3rd-party 재구성 그대로입니다.

## 10. 참고 자료

**직접 fetch해서 원문/원본 데이터를 확인한 자료** (신뢰도 높음):

- [Qwen/Qwen3.6-27B (Hugging Face 모델 카드)](https://huggingface.co/Qwen/Qwen3.6-27B)
- [deepseek-ai/DeepSeek-V4-Flash (Hugging Face 모델 카드)](https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash)
- [zai-org/GLM-5.2 (Hugging Face 모델 카드 + config.json 원본)](https://huggingface.co/zai-org/GLM-5.2)
- [GLM-5: from Vibe Coding to Agentic Engineering (arXiv:2602.15763)](https://arxiv.org/abs/2602.15763)
- [TurboQuant: Online Vector Quantization with Near-optimal Distortion Rate (arXiv:2504.19874, Google Research)](https://arxiv.org/abs/2504.19874)
- [Kimi K3's 2.8T-Parameter Launch Puts AI Capex Story on Trial (TFTC, 기사 전문 확인)](https://www.tftc.io/kimi-k3-moonshot-ai-capex-chip-selloff)
- [Moonshot AI Releases Kimi K3: A 2.8 Trillion Parameter Open MoE Model With Kimi Delta Attention and 1M Context (marktechpost, 기사 전문 확인, 2026-07-16)](https://www.marktechpost.com/2026/07/16/moonshot-ai-releases-kimi-k3-a-2-8-trillion-parameter-open-moe-model-with-kimi-delta-attention-and-1m-context/) — Moonshot 공식 블로그(`kimi.com/blog/kimi-k3`) 내용을 인용 보도. Per-Head Muon, SiTU, Gated MLA, QAT 기반 MXFP4/MXFP8 네이티브 서빙, 64+ 가속기 권장 구성 등 확인
- [google/gemma-4-31B (Hugging Face 모델 카드 + config.json 원본)](https://huggingface.co/google/gemma-4-31B) — Dense 구조(`enable_moe_block: false`) 직접 확인
- [poolside/Laguna-S-2.1 (Hugging Face, config.json 원본)](https://huggingface.co/poolside/Laguna-S-2.1) — 총 118B(HF 페이지 명시), 재계산한 활성 파라미터 ~8.4B로 신고치(8B)와 근접
- [Motif-Technologies/Motif-3-Beta (Hugging Face, config.json 원본)](https://huggingface.co/Motif-Technologies/Motif-3-Beta) — 재계산 총 파라미터 ~321B로 신고치(314B)와 근접
- [tencent/Hy3-preview (Hugging Face, config.json 원본)](https://huggingface.co/tencent/Hy3-preview) — 재계산 총 ~295B/활성 ~21B로 신고치와 정확히 일치
- [MiniMaxAI/MiniMax-M3 (Hugging Face, config.json 원본)](https://huggingface.co/MiniMaxAI/MiniMax-M3) — 재계산 총 파라미터 ~425B로 신고치(428B)와 근접
- [mistralai/Mistral-Medium-3.5-128B (Hugging Face, config.json 원본)](https://huggingface.co/mistralai/Mistral-Medium-3.5-128B) — 재계산 텍스트 백본 ~125B로 신고치(128B Dense)와 근접
- [deepseek-ai/DeepSeek-V4-Pro (Hugging Face, config.json 원본)](https://huggingface.co/deepseek-ai/DeepSeek-V4-Pro) — 재계산 총 파라미터 ~1.55~1.6T로 신고치(1.6T)와 근접
- [Introducing NVFP4 for Efficient and Accurate Low-Precision Inference (NVIDIA Technical Blog, 기사 전문 확인)](https://developer.nvidia.com/blog/introducing-nvfp4-for-efficient-and-accurate-low-precision-inference/) — NVFP4(블록 16, E4M3 스케일) vs MXFP4(블록 32, E8M0 스케일) 공식 비교표, NVFP4 메모리 절감(FP16 대비 3.5배, FP8 대비 1.8배) 확인
- [NVIDIA Vera Rubin NVL72 공식 스펙 페이지 (nvidia.com, 직접 fetch 확인)](https://www.nvidia.com/en-us/data-center/vera-rubin-nvl72/) — Rubin GPU 1개(HBM4 288GB/22TB/s, NVFP4 추론 50 PFLOPS)부터 NVL72 랙(HBM4 20.7TB/1,580TB/s, NVFP4 추론 3,600 PFLOPS)까지 공식 스펙표 확인. "Preliminary information" 각주 있음
- [NVIDIA GB300 NVL72 공식 스펙 페이지 (nvidia.com, 직접 fetch 확인)](https://www.nvidia.com/en-us/data-center/gb300-nvl72/) — 랙 단위 수치(NVLink 130TB/s, FP4 1,440 PFLOPS sparse)는 공식 확인, 개별 B300 GPU의 288GB/8TB/s는 이 페이지에 없어 여전히 3rd-party(Supermicro 데이터시트 등) 소스

**검색 스니펫/2차 소스로만 확인한 자료** (교차검증 약함, 원문 미확인):

- [QuaRot: Outlier-Free 4-Bit Inference in Rotated LLMs (arXiv:2404.00456)](https://arxiv.org/abs/2404.00456) — abstract만 확인, 잘 알려진 논문이라 신뢰도는 높은 편
- [Unsloth Dynamic 2.0 GGUFs 공식 문서](https://unsloth.ai/docs/basics/unsloth-dynamic-2.0-ggufs) — 검색 스니펫으로만 확인
- Kimi K3 활성 파라미터 정확한 수치 — `kimi.com/blog/kimi-k3`(403), `huggingface.co/moonshotai/Kimi-K3`(401) 둘 다 여전히 직접 접근 실패(2026-07-22 재확인). 가중치·기술 리포트가 2026-07-27 공개되면 재검증 필요. 아키텍처 구성요소(KDA, Attention Residuals, Stable LatentMoE, Per-Head Muon, SiTU, Gated MLA)는 marktechpost의 Moonshot 블로그 인용 보도로 확인했으나, 원문 블로그 직접 확인은 아직 못함
- NVIDIA B300(개별 GPU) 스펙(288GB HBM3e, 8TB/s) — Supermicro 데이터시트·Spheron 등 3rd-party 기반. NVIDIA 공식 페이지(gb300-nvl72)에는 랙 단위 수치만 있고 개별 GPU 스펙시트 미확인
- Rubin Ultra(2027 하반기, Kyber 랙)·Feynman(2028, LP40/Rosa CPU) 세부 스펙 — reddit, tech-insider.org, thundercompute.com, vrlatech.com 종합. GTC 발표 자체는 실제 있었으나 트랜지스터 수 등 세부 수치는 NVIDIA 공식 자료로 재확인 필요. (NVFP4/MXFP4 블록 크기와 Vera Rubin NVL72/Rubin GPU 스펙은 이번 세션에 NVIDIA 공식 페이지로 확인 완료 — 참고 자료 섹션 참고)
- Qwen3.8-Max(2.4T, 프리뷰, 가중치 미공개) — kie.ai, aitoolsreview.co.uk, coursiv.io 검색 스니펫으로 확인. Alibaba 자체 발표 수치이며 독립 검증 불가(MoE 여부·활성 파라미터·정확한 컨텍스트 길이 모두 미공개)
- Meta Llama 4(Scout 109B/17B 활성, Maverick 400B/17B 활성, 2025-04 출시, 2026년 모델이 아니라 표에서 제외) — reddit, explainx.ai 검색 스니펫으로 확인. Meta가 이후 프론티어 모델을 "Muse"라는 비공개 브랜드로 옮겼다는 보도(theplanettools.ai, tech-insider.org)는 검색 스니펫 수준으로만 확인, 원문 미확인

위 자료는 2026년 7월 22일 기준으로 확인했습니다.

[^tftc]: [Kimi K3's 2.8T-Parameter Launch Puts AI Capex Story on Trial (TFTC)](https://www.tftc.io/kimi-k3-moonshot-ai-capex-chip-selloff) — 기사 전문 확인.
[^gpu-price]: RTX 5090은 출시가 369만원에서 현재 700만원대로 뛰었습니다(글로벌이코노믹 g-enews.com 2025-01-07, 네이버 블로그 2026-02 검색 스니펫으로 확인, 직접 fetch는 못함). RTX PRO 6000 Blackwell(워크스테이션용, 96GB GDDR7)은 출시가 $8,565에서 현재 $13,250 수준으로 뛰었습니다(thundercompute.com 블로그 검색 스니펫으로 확인, 직접 fetch는 못함. NVLink 미지원(PCIe Gen5 x16) 등 스펙은 reddit(r/nvidia) 유출 정보와 일치). AI 데이터센터가 메모리(HBM/DRAM) 생산능력을 흡수해 GDDR7 원가 자체가 뛴 공급 측 요인이 메인이고(techpowerup.com 기사, reddit 인용, 확인), 여기에 로컬 AI 추론용으로 대용량 VRAM GPU를 찾는 수요가 게이머 수요 위에 얹힌 것도 보조 요인으로 꼽힙니다(r/LocalLLaMA 커뮤니티 글로만 정성적 확인, 정량 데이터 없음, 검색 스니펫 수준).
[^model-table]: 2025년에 나온 Meta Llama 4, xAI Grok은 2026년 모델이 아니라 표에서 제외. 오픈 웨이트 모델은 Hugging Face `config.json`을 직접 fetch해서 총 파라미터를 재계산·교차검증했고, 전부 신고치와 근접한 범위로 확인됨. 검증 방법과 그 외 세부 사항(모델별 출처, 신뢰도, 제외 근거 등)은 [참고 자료](#10-참고-자료) 섹션 참고.
[^memory-calc]: GLM-5.2(744B)를 FP8로 서빙하려면 단순 계산으로 약 744GB의 메모리가 필요합니다. 회사 추론 머신의 대부분을 차지하는 NVIDIA H200 그래픽카드는 한 장에 141GB의 HBM3e 메모리를 탑재하고 있습니다. 네 장이면 564GB라 부족하고, 여덟 장이면 1,128GB가 되므로 GLM-5.2 FP8 모델을 서빙할 수 있습니다. Kimi K3(2.8T)는 SFT 단계부터 QAT를 거쳐 가중치를 MXFP4(4비트, 파라미터당 0.5바이트)로 네이티브 배포합니다. 하지만 MXFP4/NVFP4 네이티브 가속은 블랙웰부터 지원되고(이 문서 상단의 "가속기 세대별 네이티브 정밀도" 참고), 호퍼(H200)에는 FP4 텐서코어가 없어 MXFP4 가중치를 그대로 로드해 돌릴 수 없습니다. 따라서 H200에서는 FP8로 변환해 서빙해야 하며, 이때 필요한 메모리는 파라미터당 1바이트인 약 2.8TB로, H200 8장(1,128GB)은 물론 16장(2,256GB)도 부족하고 32장(약 4.5TB)이 있어야 합니다. 반면 B300(Blackwell Ultra)은 장당 288GB이고 MXFP4/NVFP4를 네이티브 가속하므로, 저장 용량만 필요한 약 1.4TB 기준 8장(2,304GB, 약 2.3TB)이면 서빙할 수 있습니다(KV캐시 등 오버헤드는 별도).
[^kv-formula]: 공식 맨 앞의 2는 K와 V 두 개를 저장하기 때문입니다.

[^sft]: 사전학습(Pretraining)으로 "다음 토큰 맞추기"만 학습한 베이스 모델을, (질문, 좋은 답변) 형태의 예시 데이터로 추가 학습시켜 지시를 따르는 대화형 모델로 다듬는 단계입니다.

[^qat]: 모델을 학습하는 동안 저비트 양자화로 인한 오차까지 감안해서 가중치를 조정하는 방식입니다. 학습이 끝난 뒤에 양자화하는 사후 양자화(PTQ, Post-Training Quantization)보다 품질 저하가 작습니다.

[^ai-sovereign]: 2026년 7월 로이터 보도에 따르면 중국 상무부는 Alibaba·ByteDance·Z.ai(Zhipu)와 만나 최신 AI 모델(미공개 오픈웨이트 포함)의 해외 접근 제한을 논의했습니다(아직 확정 규제는 아님). 같은 시기 미국도 특정 프론티어 모델의 배포를 국가안보 사유로 일시 중단·제한한 사례가 있습니다. 즉 GPU 수출 규제뿐 아니라 모델(가중치) 자체의 국외 유출 통제도 논의 대상이 되고 있습니다.

[^kv-cache-opt]: KV캐시를 줄이는 최적화 기술들이 다양하게 존재해서, 컨텍스트가 길어져도 무작정 캐시가 불어나는 상황보다는 낫습니다. 자세한 내용은 [아키텍처 섹션](#6-아키텍처-살펴보기)에서 다룹니다.
[^spec-decoding]: 정확히는 "완전히 순차적"이라는 말에도 예외가 있습니다. 투기적 디코딩(speculative decoding)은 작고 빠른 보조 모델(draft model)이 앞으로 나올 여러 토큰을 미리 추측해서 한꺼번에 제시하고, 본 모델이 그 추측들을 병렬로(한 번의 메모리 읽기로) 검증합니다. 추측이 맞으면 여러 토큰을 한 스텝에 확정할 수 있어 디코딩이 빨라지고, 틀리면 그 지점부터 다시 순차 생성으로 돌아갑니다. 즉 매 토큰마다 전체 파라미터를 다시 읽어야 한다는 제약 자체는 그대로지만, 한 번 읽을 때 여러 토큰 분량을 뽑아낼 수 있어 실효 처리량이 늘어나는 방식입니다.
[^mem-bandwidth]: 여기서 말하는 메모리 대역폭은 CPU 캐시와는 무관합니다. 추론 연산 자체가 GPU에서 일어나므로, 정확히는 GPU의 HBM(VRAM)과 GPU 연산 코어(Tensor Core/SM) 사이에서 데이터를 주고받는 속도를 가리킵니다. 이 문서 앞부분의 [GPU 로드맵](#9-부록-nvidia-데이터센터-gpu-로드맵)에 나온 대역폭 수치(예: H200 HBM3e 4.8TB/s)가 바로 이 값입니다.
[^moe-history]: MoE 자체는 R1 이전에도 Mixtral, GPT-4 등에서 쓰이던 구조지만, "적은 인프라 투자로도 프론티어급 성능이 나온다"는 충격과 함께 R1을 통해 일반 대중에게까지 널리 알려졌습니다.

[^ds-v4-pricing]: DeepSeek 공식 API 가격(2026년 상반기 기준, `api-docs.deepseek.com/quick_start/pricing`). 2025년 말 발표된 프로모션 할인가(V4 Pro 캐시 히트 $0.003625보다 더 낮은 $0.0145 등 표기가 커뮤니티에 혼재)가 있어 시점에 따라 소폭 다를 수 있으나, "캐시 히트가 출력 대비 두 자릿수~세 자릿수 배 저렴하다"는 비율 자체는 여러 소스에서 일관되게 확인됩니다. 참고로 시간대별 성수기(피크) 요금은 이 값의 약 2배입니다.
