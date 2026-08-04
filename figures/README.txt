이 폴더의 HTML이 이 문서의 그림입니다. 이미지 파일이 아니라 **소스**입니다.

읽는 곳이 세 군데이고, 쓰는 방식이 다릅니다.
  /document/  (읽기용 문서)  iframe으로 **실시간 렌더링**
  /slides/    (발표용 덱)    iframe으로 **실시간 렌더링**
  README.md                  캡처된 ../assets/*.webp
                             — GitHub이 마크다운의 iframe을 지우기 때문입니다.

그래서 이 폴더는 배포됩니다(.gitignore에서 제외돼 있지 않습니다).
캡처는 README 한 곳만을 위한 것입니다.

레이아웃 기준
  캔버스 1376x768. 캡처 출력은 3840x2160(4K)이고 recapture.js가 zoom으로 확대해
  렌더만 다시 하므로 배치를 다시 잡을 필요는 없습니다. 이 값들은 바꾸지 마세요.
  덱 슬라이드는 1376x774라 아래 6px이 남는데, iframe 캔버스가 그림의 body 배경을
  물려받아 자동으로 채워집니다 — 캡처본과 같은 모양입니다.

고치는 법
  1. html 파일을 편집합니다.
  2. export NODE_PATH="/Users/studio/.npm/_npx/420ff84f11983ee5/node_modules"
     node figures/check.js <이름일부>          # 색 리터럴·넘침·잘림·고아줄·foot여백
     node figures/fontcheck.js <이름일부>      # 실제 렌더에 쓰인 폰트 (시스템 폰트 폴백)
  3. node figures/recapture.js <이름일부>      # README용 다크 판
     node figures/recapture.js --light <이름일부>   # README용 라이트 판
     → 그림을 고쳤으면 **두 번 돌립니다.** 한쪽만 돌리면 테마를 바꿀 때 옛 그림이 나옵니다.
  4. 결과 webp를 눈으로 열어 해당 절 내용과 대조합니다.
     스크립트가 해시 중복은 잡지만 "내용이 그 절에 맞는지"는 사람만 압니다.

  check.js가 못 잡는 것 (2026-07-31 기준)
    · 세로 넘침 — 넘침 검사가 좌우만 봅니다. 카드에 요소를 추가했으면
      `카드.bottom - 마지막요소.bottom`을 직접 재세요.
    · 폰트 — fontcheck.js가 대신 봅니다. CDP로 **실제 렌더에 쓰인 플랫폼 폰트**를
      물어보므로 선언이 아니라 결과를 봅니다.

지켜야 할 네 가지
  1) 색을 직접 쓰지 마세요.
     base.css의 :root 와 :root[data-theme="light"] **두 벌**에 변수를 추가하고
     var()로 씁니다. 리터럴이 남으면 라이트 테마에서 그 색만 다크로 남아 얼룩이 됩니다.
     같은 다크 색이 놓이는 자리에 따라 라이트 값이 갈릴 때는 역할 이름을 씁니다
     (--ink-on-cy-tint 처럼). check.js가 리터럴을 찾아 실패로 보고합니다.

  2) 시스템 폰트를 쓰지 마세요.
     루트 fonts/의 웹폰트(Pretendard Variable · IBM Plex Mono)만 씁니다.
     2026-07-30까지 base.css가 "Apple SD Gothic Neo"를 쓰고 있었고, 그동안은
     이 맥에서만 레이아웃이 맞았습니다. 서브셋은 KS X 1001 한글 2,350자를 담아
     현대 한글은 자유롭게 써도 됩니다.

     **모노 스택에는 "Pretendard Variable"을 둘째로 넣습니다** —
     `font:700 11px "IBM Plex Mono", "Pretendard Variable", monospace`.
     IBM Plex Mono에는 한글이 없어서, 모노로 지정한 라벨에 한글이 섞이면
     그 글자만 시스템 폰트(Apple SD Gothic Neo)로 폴백됩니다. 2026-07-31에
     그런 자리가 58곳 있었습니다. 폴백 목적지를 우리 웹폰트로 못박아 둡니다.

     검사는 fontcheck.js가 합니다(아래).

  3) 반복되는 작은 칸은 div가 아니라 SVG로 그리세요. (2026-08-03 전수 정비)
     recapture.js가 zoom 1.395 + DPR 2 = 2.7907배로 찍는데, div 칸은 크로미움이
     박스마다 페인트 좌표를 정수 디바이스픽셀로 스냅합니다. 그래서
     "칸 크기 x 2.7907"의 소수부가 0이 아니면 칸 크기가 제각각으로 갈립니다.
     4px 칸 실측: div 10.243~12.262 (편차 2.019px) → SVG 11.256~11.759 (0.503px).

     칸 크기뿐 아니라 피치도 보세요. v4의 .source는 칸이 10px(27.9dev, 거의 정수)인데
     피치 12px(33.49dev)이 스냅되고 있었습니다.

     DOM에서는 균일하게 보입니다 — getBoundingClientRect()로는 안 잡히고
     캡처 PNG를 디코드해 재야 보입니다. check.js도 이건 잡지 못합니다.

     지금 SVG로 그리는 곳:
       desk(책상) · k3(전문가 896·GPU 40) · gdn(상태 아이콘·문맥 트랙) ·
       expert_parallel(점 352) · v4(원본 96·막대 40) ·
       architecture_map(토큰 12·요약 4) · units(격자 200·rtok 10) · moe(전문가 256)
     div 격자로 되돌리지 마세요. 각 소스 주석에 이유를 적어 뒀습니다.

     전환할 때는 DOM에서 칸 크기·피치·열/행·시작점을 먼저 재고 viewBox와 rect
     좌표에 그대로 넣으면 화면이 바뀌지 않습니다. CSS 그라디언트는 SVG
     linearGradient로 옮깁니다(units의 반 칸 — 50% 하드 스톱 둘).

  4) 전문가 격자에는 테두리를 두르지 마세요.
     6.1의 R1(moe)과 6.6의 K3(k3)이 같은 양식입니다 —
     꺼진 점 --c-4d5661 · 켜진 점 --c-0e8fa3 · box-shadow 링 없음.
     링을 두르면 맨 윗줄 위와 맨 오른쪽줄 옆에 희미한 선으로 보입니다.
     격자는 점만으로 경계가 읽힙니다.

새 그림을 추가할 때
  1. figures/새이름.html 을 만들고 base.css를 링크합니다.
  2. recapture.js의 MAP에 'html이름' : 'assets이름' 을 추가합니다. (여기가 유일한 원천)
  3. node figures/manifest.js     → figures/manifest.json 갱신
     이 파일로 /document/ 와 /slides/ 가 에셋 이름에서 소스를 찾습니다.
     이름이 다른 것이 33개라(prices.html → memory_price_chain.webp) 손으로 적으면 틀립니다.
  4. README에 이미지를 넣고 캡처 두 번.

파일 ↔ 그림 대응 (65장 — glm_bench·k3_bench·refusal_gate·quant_calib·rotation_absorb·
                    rotation_family·quant_gain·kr_infra·kr_bench·sov_boundary·sov_personal·
                    sov_stack 열둘은 덱 전용, README 미편입 · 8장 다섯이 덱 미편입)
  shock.html             → assets/stock_shock.webp               1. Kimi K3는 어떻게 미국 증시를 흔들었나?
  capex.html             → assets/us_china_capex.webp            1.1 미국과 중국, 인프라는 8배 차이입니다
  bench.html             → assets/frontier_bench.webp            1.2 그런데 모델의 격차는 3.6점입니다
  tokens.html            → assets/token_growth.webp              2. 추론 시장으로의 전환
  lease.html             → assets/compute_lease.webp             자체 인프라만으로는 증가 속도를 따라잡지 못했습니다
  prices.html            → assets/memory_price_chain.webp        그 수요는 여러분의 노트북 가격표에 도착했습니다
  vram.html              → assets/model_capacity.webp            3. LLM을 담기 위해서는 얼마만큼의 메모리가 필요할까요?
  budget.html            → assets/node_memory_budget.webp        3. 가중치를 올리면 3분의 2가 찹니다 (노드 예산)
  readall.html           → assets/next_token_all.webp           4. 한 글자를 더 쓰려면 지금까지의 전부를 봅니다
  desk.html              → assets/kv_cache_desk.webp             4. LLM에게 일을 시키려면 얼마만큼의 메모리가 필요할까요?
  kvcalc.html            → assets/kv_cache_calc.webp             4. LLM에게 일을 시키려면 얼마만큼의 메모리가 필요할까요?
  ctx.html               → assets/context_growth.webp            그런데 대화 길이가 3년 만에 256배가 됐습니다
  units.html             → assets/compute_units.webp            5. 입력은 유닛을 다 쓰고 출력은 반 칸도 못 씁니다
  pdbench.html           → assets/prefill_decode_bench.webp      5. LLM 모델의 연산: 프리필과 디코딩
  prefill.html           → assets/prefill_parallel.webp          5.1 프리필
  decoding.html          → assets/decoding_sequential.webp       5.2 디코딩
  hit_chart.html         → assets/kv_cache_hit_chart.webp        5.3 KV 캐시 히트
  kv_tier.html           → assets/kv_tier.webp                   5.3 히트한 KV는 어디에 보관되나
  row_cost.html          → assets/cache_hit_row_cost.webp        5.3 KV 캐시 히트
  pricing.html           → assets/cache_hit_pricing_new.webp     5.3 KV 캐시 히트
  speculative.html       → assets/speculative_decoding.webp      5.4 투기적 디코딩: 순차 병목을 우회하는 법
  architecture_map.html  → assets/architecture_efficiency_map.webp 6. 아키텍처 살펴보기
  moe.html               → assets/moe_active.webp                6.1 DeepSeek R1 - Mixture of Experts (MoE) +
  moe_scope.html         → assets/moe_scope.webp                 어텐션은 담는 값의 1.7%, 읽는 양의 31%입니다
  sparse_map.html        → assets/sparse_attention_map.webp      6.1을 빼면 전부 어텐션 이야기입니다
  swa.html               → assets/swa_layers.webp                6.2 Gemma 4 - Sliding Window Attention (SWA)
  swa_reach.html         → assets/swa_reach.webp                 6.2 창문 밖 이야기를 막아주는 두 장치
  gdn.html               → assets/gdn_summary_note.webp          6.3 Qwen3-Next-80B-A3B - Gated DeltaNet
  v4_pricing.html        → assets/v4_api_pricing.webp            6.4 DeepSeek V4 - Hybrid Attention (CSA + HC
  v4_spec.html           → assets/v4_spec.webp                   6.4 DeepSeek V4 - Hybrid Attention (CSA + HC
  v4.html                → assets/v4_compression.webp            6.4 DeepSeek V4 - Hybrid Attention (CSA + HC
  gqa.html               → assets/gqa_head_sharing.webp          R1의 어텐션은 MLA입니다 (6.1 · 덱은 아직 6.5)
  indexshare.html        → assets/indexshare.webp                6.5 GLM-5.2 - IndexShare와 비동기 강화학습
  glm_spec.html          → assets/glm_spec.webp                  6.5 GLM-5.2 - IndexShare와 비동기 강화학습
  glm_bench.html         → assets/glm_bench.webp                 6.5 GLM-5.2 벤치마크 8종 심층 비교 (덱 전용)
  glm_async_rl.html      → assets/glm_async_rl.webp              6.5 학습 클러스터 절반이 추론 서버입니다
  attnres.html           → assets/attention_residuals.webp       6.6 Kimi K3 - Kimi Delta Attention (KDA)
  k3.html                → assets/k3_synthesis.webp              6.6 Kimi K3 - Kimi Delta Attention (KDA)
  k3_bench.html          → assets/k3_bench.webp                  6.6 Kimi K3 벤치마크 45종 전수 조사 (덱 전용)
  k3_spec.html           → assets/k3_spec.webp                   6.6 Kimi K3 - Kimi Delta Attention (KDA)
  refusal_gate.html      → assets/refusal_gate.webp              6.6 폐쇄 대 오픈웨이트 거절 매트릭스 (덱 전용)
  quant_map.html         → assets/quantization_map.webp          7.1 Model 양자화
  quant_intro.html       → assets/quant_two_bills.webp            7. 양자화는 압축입니다 — 두 청구서에 함께
  formats.html           → assets/precision_formats.webp          7.1 정밀도 포맷 여섯과 가속 세대
  unimix.html            → assets/uniform_vs_mixed.webp           7.1 균일 정밀도 대 혼합 정밀도
  tq_shock.html          → assets/turboquant_shock.webp          7.2 TurboQuant 발표와 메모리 증시 충격
  rotation_absorb.html   → assets/rotation_absorb.webp           7.2 회전의 흡수 경계 (덱 전용)
  rotation_family.html   → assets/rotation_family.webp           7.2 회전 계열 네 기법 (덱 전용)
  quant_gain.html        → assets/quant_gain.webp                7.3 양자화가 듣는 세 병목 (덱 전용)
  hadamard.html          → assets/kv_quant_rotation.webp         7.2 KV 캐시 양자화
  parallel_axes.html     → assets/parallel_axes.webp             8.1 나누는 네 축
  pd_tradeoff.html       → assets/pd_tradeoff.webp               그런데 분리가 처리량을 올려 주는 건 아닙니다
  split_gain.html        → assets/split_gain.webp                8.2 담기는데도 나눕니다
  expert_parallel.html   → assets/expert_parallel.webp           8.3 MoE는 전문가로 나눕니다
  pd_split.html          → assets/pd_split.webp                  8.4 프리필과 디코딩을 다른 GPU로
  sovereign.html         → assets/ai_sovereignty.webp            9.1 국가 단위의 AI 소버린
  kr_infra.html          → assets/kr_infra.webp                  9.2 국가별 가속기 장수 비교 (덱 전용)
  kr_bench.html          → assets/kr_bench.webp                  9.2 한국 모델의 AAII 지수 위치 (덱 전용)
  sov_boundary.html      → assets/sov_boundary.webp              9.3 기업이 자체 인프라를 짓는 이유 (덱 전용)
  sov_personal.html      → assets/sov_personal.webp              9.3 개인의 맥락과 두 종속 (덱 전용)
  sov_stack.html         → assets/sov_stack.webp                 9.3 국가·기업·개인 세 층의 메모리 수요 (덱 전용)
  engine_stack.html      → assets/inference_engine_stack.webp    맥에서는 이렇게 갈립니다
  roadmap.html           → assets/gpu_roadmap.webp               10. 부록: NVIDIA 데이터센터 GPU 로드맵

라이트 테마
  그림이 CSS 변수로 두 벌의 색을 갖고 있어 <html data-theme="light"> 하나로 바뀝니다.
  2026-07-30 이전에는 lightmap.js의 대응표로 소스의 색 문자열을 치환한 사본을
  figures/.light/에 만들어 거기서 캡처했습니다. 그 방식은 폐기했습니다
  (옮길 때 78장을 픽셀 단위로 대조해 결과가 같음을 확인했습니다).
  lightmap.js는 각 대응의 근거(대비비·색약 ΔE)를 남기기 위해 보존만 합니다.

되돌리기
  실시간 렌더링이 말썽이면 document/index.html의 USE_LIVE_FIGURES 와
  slides/present.js의 LIVE_FIGURES 를 false로 두면 캡처된 webp로 돌아갑니다.
