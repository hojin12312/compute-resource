# 부록 A · DeepSeek V4 깊게 파기

[← 본문으로 돌아가기](../README.md#64-deepseek-v4---hybrid-attention-csa--hca)

본문 [6.4장](../README.md#64-deepseek-v4---hybrid-attention-csa--hca)은 CSA와 HCA를 두 문단으로
줄여 설명합니다. 이 부록은 그 아래에 있는 것 — **설정 파일과 공식 추론 코드가 실제로 무엇을
하는지** — 를 따로 정리한 것입니다. 본문에 넣기에는 너무 깊고, 버리기에는 아까운 내용입니다.

## 이 문서가 본 것

| 무엇 | 경로 | 확인 시각 |
|---|---|---|
| Flash 설정 | `huggingface.co/deepseek-ai/DeepSeek-V4-Flash/raw/main/config.json` | 2026-07-30 |
| Pro 설정 | `huggingface.co/deepseek-ai/DeepSeek-V4-Pro/raw/main/config.json` | 2026-07-30 |
| 공식 추론 코드 | `huggingface.co/deepseek-ai/DeepSeek-V4-Flash/raw/main/inference/model.py` | 2026-07-30 |

표기 규칙은 `RESEARCH.md`와 같습니다 — **[1차]** 는 위 파일에서 직접 읽은 것, **[유도]** 는
그 값으로 이 문서가 계산한 것, **[미확인]** 은 확인하지 못한 것입니다. 코드는 selective fetch로
읽었으므로 파일 전체를 한 줄씩 훑은 것은 아닙니다. 아래에 인용한 부분은 직접 본 것이고,
보지 못한 부분에 대한 추측은 9장에 따로 모았습니다.

## 1. 한 줄 요약

**V4의 어텐션은 MLA가 도달하는 지점을 아키텍처로 굳힌 것입니다.** 저차원 잠재를 만들어 두고
쓸 때 헤드별 K/V로 펴는 V2·V3 방식이 아니라, **처음부터 512차원 하나만 만들어 그것을 모든
헤드의 키이자 값으로 씁니다.** 업프로젝션 행렬이 체크포인트에 없습니다. 그리고 절감의 무게가
헤드 축에서 **시퀀스 축**(4토큰·128토큰을 하나로 묶는 학습된 풀링)으로 옮겨갔습니다.

## 2. 어텐션 — MLA인가 MQA인가

### 2.1 이름이 두 개 다 나옵니다

`model.py` 안에서 같은 것을 두 이름으로 부릅니다. **[1차]**

- `class Attention`의 독스트링: *"Multi-head Latent Attention (MLA) with sliding window +
  optional KV compression."*
- `ModelArgs`에서 그 하이퍼파라미터들 위에 붙은 주석: **`# mqa`** (`q_lora_rank`, `head_dim`,
  `rope_head_dim`, `norm_eps`, `o_groups`, `o_lora_rank`, `window_size`가 그 아래 옵니다)

DeepSeek 자신이 둘을 같은 것으로 보고 있다는 뜻입니다. 왜 그런지는 [부록 B의 흡수
설명](./attention-lineage.md#6-흡수--결합법칙을-쓰는-것)에 있습니다 — **흡수한 MLA는 압축 잠재
위의 MQA와 수학적으로 같은 것**이 되기 때문입니다.

### 2.2 업프로젝션이 없습니다

어텐션 모듈의 파라미터 선언 전부입니다. **[1차]**

```python
self.wq_a = Linear(self.dim, self.q_lora_rank)                              # 4096 → 1024
self.q_norm = RMSNorm(self.q_lora_rank, self.eps)
self.wq_b = ColumnParallelLinear(self.q_lora_rank, self.n_heads * self.head_dim)  # 1024 → 64×512
self.wkv = Linear(self.dim, self.head_dim)                                  # 4096 → 512
self.kv_norm = RMSNorm(self.head_dim, self.eps)
self.wo_a = ColumnParallelLinear(self.n_heads * self.head_dim // self.n_groups,
                                 self.n_groups * args.o_lora_rank)
self.wo_b = RowParallelLinear(self.n_groups * args.o_lora_rank, self.dim)
self.attn_sink = nn.Parameter(torch.empty(self.n_local_heads, dtype=torch.float32))
```

읽을 것이 셋입니다.

- **`wkv`가 하나뿐이고 출력이 `head_dim`(512)입니다.** V2·V3의 `wkv_b`(잠재 → 헤드별 K/V)가
  존재하지 않습니다. 설정에도 `kv_lora_rank`·`qk_nope_head_dim`·`v_head_dim`이 **모두
  없습니다** — DeepSeek MLA의 서명 필드가 사라졌습니다.
- **Q는 저랭크 경로를 유지합니다.** `wq_a`(4096→1024) → `q_norm` → `wq_b`(1024→64×512).
  Q는 캐시에 남지 않으므로 이 병목은 KV캐시와 무관하고, 파라미터와 활성값을 줄이는 장치입니다.
- **출력도 저랭크이고 그룹으로 나뉩니다.** `o_groups`(Flash 8 · Pro 16)로 헤드를 묶어
  `o_lora_rank`(1024)를 통과시킵니다. [부록 B](./attention-lineage.md)에서 본 "헤드별 구분이
  가중치로 옮겨간다"의 그 자리입니다.

### 2.3 K와 V가 같은 텐서입니다

`forward`의 KV 경로와 어텐션 호출입니다. **[1차]**

```python
kv = self.wkv(x)
kv = self.kv_norm(kv)
apply_rotary_emb(kv[..., -rd:], freqs_cis)                  # 마지막 64차원에만 RoPE
act_quant(kv[..., :-rd], 64, scale_fmt, scale_dtype, True)  # 나머지 448차원은 FP8
...
o = sparse_attn(q, kv, self.attn_sink, topk_idxs, self.softmax_scale)
apply_rotary_emb(o[..., -rd:], freqs_cis, True)             # ← 출력에서 역회전
```

`sparse_attn`에 들어가는 텐서가 `q`와 `kv` 둘뿐입니다. `kv`는 토큰당 512차원 하나이고, Q는
헤드당 512차원이라 **64개(Pro는 128개) 쿼리 헤드가 같은 벡터 하나에 직접 내적**합니다.
별도의 V가 없습니다.

**마지막 줄이 결정적 증거입니다.** 키 역할을 하려면 RoPE를 걸어야 하는데, 같은 텐서를 값으로도
쓰므로 가중합이 끝난 뒤 **회전을 되돌려야** 합니다(`apply_rotary_emb`의 세 번째 인자).
K와 V가 별개 텐서라면 이 줄이 존재할 이유가 없습니다.

### 2.4 계보

| | KV 저장 형태 | 업프로젝션 | 헤드별 구분은 어디에 |
|---|---|---|---|
| V2 · V3 · V3.2 | 잠재 512 + 위치키 64 | `W^UK`·`W^UV` 있음(추론 시 흡수) | 업프로젝션 |
| **V4** | **512 하나 (K = V)** | **없음** | 쿼리 쪽 `wq_b` + 출력 쪽 `wo_a`/`wo_b` |

그래서 "V4도 MLA인가"의 답은 이렇습니다. **파라미터화로는 아니고**(업프로젝션이 없음),
**서빙 시 계산 형태로는 같습니다**(둘 다 512차원 벡터 하나에 어텐션). V2가 학습은 MLA로 하고
추론에서 흡수했던 것을, V4는 학습부터 흡수된 형태로 합니다.

**[미확인]** 이 변경이 품질에 어떤 영향을 주는지는 공개 자료로 확인하지 못했습니다. 업프로젝션을
없애면 표현력이 줄 수도 있고(`W^UK`가 헤드별로 다른 부분공간을 뽑아낼 자유가 사라짐), 반대로
학습·추론 불일치가 없어져 유리할 수도 있습니다. 리포트에 이 ablation이 있는지 확인하지 못했습니다.

## 3. 시퀀스 축 압축 — `Compressor`의 실제 동작

본문은 "4토큰을 하나로 묶는다"고 적었는데, 코드는 **묶기가 아니라 학습된 게이트 풀링**입니다. **[1차]**

```python
class Compressor(nn.Module):
    """Compresses KV cache via learned gated pooling over `compress_ratio` consecutive tokens.
    When overlap=True (ratio==4), uses overlapping windows for smoother compression boundaries."""
    self.ape   = nn.Parameter(...)          # 그룹 내 위치 임베딩 (compress_ratio × 폭)
    self.wkv   = Linear(self.dim, coff * self.head_dim)    # 압축될 값
    self.wgate = Linear(self.dim, coff * self.head_dim)    # 그 값의 점수
    self.norm  = RMSNorm(self.head_dim, args.norm_eps)
    ...
    kv = (kv * score.softmax(dim=2)).sum(dim=2)            # 그룹 안에서 softmax 가중합
```

정리하면 이렇습니다.

- **평균이 아니라 가중합입니다.** `wgate`가 토큰마다 점수를 내고 그룹(4개 또는 128개) 안에서
  softmax를 걸어 가중치로 씁니다. 즉 그룹 안에서 중요한 토큰이 요약을 더 많이 차지합니다.
  게이트 계산은 fp32로 합니다(`x = x.float()`).
- **그룹 안의 위치를 구분합니다.** 학습 파라미터 `ape`를 점수에 더합니다 — 4토큰 중 첫째와
  넷째가 다르게 취급됩니다.
- **4:1일 때는 창이 겹칩니다.** `self.overlap = compress_ratio == 4`이고, 겹치는 경우 폭이
  두 배(`coff = 2`)인 투영을 만들어 앞 절반은 겹치는 창, 뒤 절반은 일반 창에 씁니다
  (`overlap_transform`). 독스트링의 표현은 *"smoother compression boundaries"* 입니다.
  128:1에는 겹침이 없습니다.
- **압축된 요약에도 RoPE를 걸고 정밀도를 나눕니다.** 마지막 64차원에 회전을 적용하는데,
  이때 쓰는 base는 원본과 다릅니다 — `compress_rope_theta`가 **160,000**(원본 `rope_theta`는
  10,000)입니다. 그리고 회전이 걸리지 않은 448차원만 FP8로 시뮬레이션합니다.
- **디코딩에서는 상태 버퍼로 점진 압축합니다.** `kv_state`·`score_state`에 그룹이 찰 때까지
  모아 두고 `(start_pos + 1) % ratio == 0`인 스텝에만 요약 하나를 확정해 캐시에 씁니다.
  그래서 압축은 4스텝(또는 128스텝)마다 한 번 일어납니다.

## 4. 희소 선택 — 어텐션 대상을 세 갈래로 모읍니다

한 레이어가 볼 대상을 `topk_idxs`로 모아 `sparse_attn`에 넘깁니다. **[1차]**

```python
topk_idxs = get_window_topk_idxs(win, bsz, seqlen, start_pos)          # 최근 128토큰 (무압축)
if self.compress_ratio:
    if self.indexer is not None:                                        # 압축률 4 → DSA
        compress_topk_idxs = self.indexer(x, qr, start_pos, offset)
    else:                                                               # 압축률 128 → 전체
        compress_topk_idxs = get_compress_topk_idxs(ratio, bsz, seqlen, start_pos, offset)
    topk_idxs = torch.cat([topk_idxs, compress_topk_idxs], dim=-1)
```

- **`sliding_window: 128`은 모든 레이어에 있습니다.** 최근 128토큰은 압축하지 않고 그대로
  캐시에 둡니다(`kv_cache_size = window_size + max_seq_len // compress_ratio`).
- **CSA 레이어(압축률 4)만 인덱서를 가집니다.** `if self.compress_ratio == 4: self.indexer =
  Indexer(...)` else `None`. 본문 각주에 적은 "DSA 인덱서는 CSA 레이어마다 독립이고 HCA에는
  없다"가 코드로 확인됩니다.
- **HCA 레이어(압축률 128)는 고르지 않고 전부 봅니다.** `get_compress_topk_idxs`는 인과
  마스크만 적용해 **가능한 모든 압축 위치**를 반환합니다. 본문의 "남은 요약을 모두 봅니다"가
  맞습니다.
- **어텐션 싱크가 헤드마다 학습 파라미터로 있습니다**(`attn_sink`). 소프트맥스에 항상 참여하는
  가상의 자리로, 희소 어텐션에서 분포가 붕괴하는 것을 막는 장치로 알려져 있습니다.

### 인덱서 내부

```python
class Indexer(torch.nn.Module):
    """Selects top-k compressed KV positions for sparse attention via learned scoring.
    Has its own Compressor (with Hadamard rotation) to build compressed KV for scoring."""
    self.wq_b = ColumnParallelLinear(q_lora_rank, n_heads * head_dim)   # index_head_dim 128
    self.weights_proj = ColumnParallelLinear(dim, n_heads)
    self.compressor = Compressor(args, compress_ratio, self.head_dim, True)   # rotate=True
    self.register_buffer("kv_cache", torch.zeros(B, max_seq_len // ratio, head_dim))
    ...
    q = rotate_activation(q); fp4_act_quant(q, fp4_block_size, True)
    index_score = torch.einsum("bshd,btd->bsht", q, self.kv_cache[:bsz, :end_pos // ratio])
    index_score = (index_score.relu_() * weights.unsqueeze(-1)).sum(dim=2)
    topk_idxs = index_score.topk(min(self.index_topk, end_pos // ratio), dim=-1)[1]
```

- **인덱서는 자기 압축 캐시를 따로 듭니다.** 본체와 별개로 `index_head_dim`(128)짜리 압축 KV를
  유지합니다. 캐시 예산을 셀 때 빠뜨리기 쉬운 항목입니다(6장에서 계산).
- **하다마드 회전 + FP4 시뮬레이션.** 점수 계산용이라 정밀도를 과감하게 낮췄고, 회전으로
  이상치를 분산시킨 뒤 양자화합니다 — [7.2장 QuaRot](../README.md#72-kv-캐시-양자화)과 같은
  발상입니다. 코드 주석은 *"We performed QAT here"* 라고 적습니다.
- **점수는 softmax가 아니라 relu 뒤 헤드별 가중합**입니다(`weights_proj`가 헤드 가중치를 냄).
- **top-k 크기**는 Flash 512, Pro 1024입니다(`index_topk`).

## 5. 레이어 배치 — `compress_ratios` 읽는 법

배열 하나가 레이어별 압축률을 담습니다. `0`은 압축 없음(슬라이딩 윈도우만), `4`는 CSA,
`128`은 HCA입니다. 배열 길이는 `레이어 수 + 1`이고 마지막 항이 MTP 레이어 몫입니다. **[1차]**

| | Flash | Pro |
|---|---|---|
| 배열 길이 | 44 (43층 + MTP) | 62 (61층 + MTP) |
| 앞부분 | `0, 0` — 첫 두 층은 윈도우만 | `128, 128` — 첫 두 층이 HCA |
| 본체 | `4, 128`이 번갈아 | `4, 128`이 번갈아 |
| 마지막(MTP) | `0` | `0` |
| **CSA(4:1) 레이어** | **21개** | **30개** |
| **HCA(128:1) 레이어** | **20개** | **31개** |
| 윈도우만(0) | 2개 + MTP 1개 | 0개 + MTP 1개 |

**[유도]** 위 개수는 배열을 세어 얻은 값입니다. Flash는 본문 각주와 일치하고, Pro는 이 문서에서
처음 센 것입니다. 눈에 띄는 차이는 **Pro의 첫 두 층이 윈도우 전용이 아니라 HCA**라는 점입니다.

## 6. KV캐시 예산 — 코드에서 유도한 계산

엔트리 하나의 바이트부터 셉니다. **[유도]**

```
압축 엔트리 1개 = 512차원
  ├ 448차원 : FP8  (1바이트)  = 448 B      ← act_quant(kv[..., :-64])
  └  64차원 : BF16 (2바이트)  = 128 B      ← RoPE 차원은 정밀도 유지
                                 ─────
                                  576 B
```

1M 컨텍스트(1,048,576토큰), 사용자 한 명 기준입니다.

| | Flash | Pro |
|---|---|---|
| CSA 레이어 (엔트리 1M/4 = 262,144개) | 21 × 151.0MB = **3.17GB** | 30 × 151.0MB = **4.53GB** |
| HCA 레이어 (엔트리 1M/128 = 8,192개) | 20 × 4.72MB = 94MB | 31 × 4.72MB = 146MB |
| 슬라이딩 윈도우 (레이어당 128개) | 43 × 74KB = 3.2MB | 61 × 74KB = 4.5MB |
| **본체 합계** | **약 3.3GB** | **약 4.7GB** |
| 인덱서 전용 캐시 (128차원 BF16) | 21 × 67.1MB = 1.41GB | 30 × 67.1MB = 2.01GB |

본문 [스펙 카드](../README.md#64-deepseek-v4---hybrid-attention-csa--hca)에 적은 "약 3GB /
약 4GB"와 견주면 Flash는 맞고 Pro는 코드 기준이 조금 큽니다. 스펙 카드 값은 리포트의 "V3.2
대비 7%·10%"에서 환산한 것이라 유도 경로가 다르므로, **자릿수 수준에서 일치한다**고만 읽는 것이
맞습니다.

**인덱서 캐시가 공짜가 아닙니다.** Flash에서 1.4GB, Pro에서 2GB인데, 리포트의 KV 수치가 이것을
포함하는지 **[미확인]** 입니다. 코드 주석은 *"kv could also use fp8 format, though current
implementation uses bf16"* 라고 적으므로 FP8로 내리면 절반이 됩니다.

## 7. 어텐션 밖에서 새로 드러난 것

본문 6.4에 없는 항목들입니다. 전부 **[1차]** 이고, 각각이 별도 조사거리입니다.

### Hyper-Connections — 잔차를 네 벌로 들고 섞습니다

```python
class Block(nn.Module):
    """Transformer block with Hyper-Connections (HC) mixing.
    Instead of a simple residual, HC maintains `hc_mult` copies of the hidden state.
    hc_pre: reduces hc copies -> 1 via learned weighted sum (pre-weights from Sinkhorn).
    hc_post: expands 1 -> hc copies via learned post-weights + combination matrix."""
```

`hc_mult: 4`, `hc_sinkhorn_iters: 20`. 잔차 스트림을 하나가 아니라 **네 벌**로 유지하고, 블록에
들어갈 때 학습된 가중치로 하나로 줄이고(`hc_pre`) 나올 때 다시 네 벌로 펼칩니다(`hc_post`).
가중치는 Sinkhorn 정규화로 얻습니다. `Transformer`도 임베딩 직후 HC로 확장하고 head에서
줄입니다.

이것은 [6.6장 Kimi K3의 Attention Residuals](../README.md#66-kimi-k3---kimi-delta-attention-kda)와
**같은 문제를 다루는 다른 해법**입니다 — 잔차의 고정 누적을 학습 가능한 혼합으로 바꾸는 것.
K3는 깊이 방향 softmax 어텐션으로, V4는 여러 벌의 스트림과 Sinkhorn으로 풉니다. 본문 6.6이
"AttnRes는 이 표에서 앞에 나오지 않은 유일한 항목"이라고 적었는데, **V4도 같은 계열의 장치를
쓰고 있다는 사실을 반영해야 합니다.**

### 앞 3개 레이어는 해시 라우팅

`num_hash_layers: 3`. `Gate`의 독스트링이 *"Supports hash-based routing (first n_hash_layers)
where expert indices are predetermined per token ID"* 입니다. 앞 3층은 라우터가 점수를 내지 않고
**토큰 id로 전문가가 미리 정해집니다**(`self.tid2eid[input_ids]`). 라우팅 학습이 불안정한 초기
레이어를 우회하는 장치로 보이지만 근거는 **[미확인]** 입니다.

### 라우팅과 활성화의 세부

- `scoring_func: "sqrtsoftplus"` — 소프트맥스나 시그모이드가 아닙니다.
- `topk_method: "noaux_tc"` — 보조 손실 없이 편향 보정으로 균형을 잡는 V3 계열 방식.
- `routed_scaling_factor` Flash 1.5 · Pro 2.5, `norm_topk_prob: true`.
- 전문가는 `swiglu_limit: 10.0`으로 게이트와 업을 클램프하고 fp32로 계산합니다.
- `expert_dtype: "fp4"` — 전문가 가중치는 FP4, 나머지는 FP8(`quantization_config`,
  블록 128×128, 스케일 `ue8m0`). 본문 스펙 카드의 "파라미터당 0.54\~0.56바이트"가 이 혼합에서
  나옵니다.

### 그 외

- `num_nextn_predict_layers: 1` — MTP 레이어 하나. `MTPBlock`이 `e_proj`/`h_proj`로 임베딩과
  은닉을 합칩니다.
- YaRN: `factor 16`, `original_max_position_embeddings 65536` → 65K에서 1M으로 확장.
- Q에 RMS 정규화를 한 번 더 겁니다(`q *= rsqrt(q.square().mean(-1) + eps)`).

## 8. Flash와 Pro 설정 비교

전부 **[1차]** 입니다.

| 키 | Flash | Pro |
|---|---|---|
| `hidden_size` | 4096 | 7168 |
| `num_hidden_layers` | 43 | 61 |
| `num_attention_heads` | 64 | 128 |
| `head_dim` | 512 | 512 |
| `num_key_value_heads` | **1** | **1** |
| `qk_rope_head_dim` | 64 | 64 |
| `q_lora_rank` | 1024 | 1536 |
| `o_lora_rank` · `o_groups` | 1024 · 8 | 1024 · 16 |
| `n_routed_experts` | 256 | 384 |
| `num_experts_per_tok` | 6 | 6 |
| `n_shared_experts` | 1 | 1 |
| `moe_intermediate_size` | 2048 | 3072 |
| `index_topk` | 512 | 1024 |
| `index_n_heads` · `index_head_dim` | 64 · 128 | 64 · 128 |
| `sliding_window` | 128 | 128 |
| `compress_rope_theta` | 160,000 | 160,000 |
| `num_hash_layers` | 3 | 3 |
| `hc_mult` · `hc_sinkhorn_iters` | 4 · 20 | 4 · 20 |
| `max_position_embeddings` | 1,048,576 | 1,048,576 |
| `expert_dtype` | fp4 | fp4 |

**KV캐시 관점에서 둘의 차이는 레이어 수뿐입니다.** `head_dim` 512와 `num_key_value_heads` 1이
같으므로 토큰당·레이어당 저장량이 동일하고, Pro가 큰 이유는 CSA 레이어가 21개에서 30개로
늘었기 때문입니다. 헤드 수가 64에서 128로 두 배가 되어도 **KV캐시는 그대로**입니다 — 이것이
헤드 축을 분리한 설계의 값입니다.

## 9. 미확인 목록

- **`sparse_attn` 커널의 내부**. `kernel` 모듈에서 임포트하는 함수라 어떻게 top-k 인덱스를
  받아 계산하는지, 어텐션 싱크를 어떻게 섞는지 보지 못했습니다.
- **`hc_split_sinkhorn`의 구체적 동작**. 임포트만 확인했습니다.
- **업프로젝션을 없앤 것의 품질 영향**. 리포트에 ablation이 있는지 확인하지 못했습니다.
- **리포트의 KV 수치가 인덱서 캐시를 포함하는지.**
- **해시 라우팅을 앞 3층에만 쓰는 이유.**
- **Hyper-Connections의 출처 논문**. 이름으로 보아 선행 연구가 있을 것으로 보이나 확인하지
  않았습니다.
- **학습 쪽 서술 전반.** 이 문서는 추론 코드와 설정만 봤습니다.

## 10. 본문에 반영할 항목

이 부록을 쓰면서 본문([6.4장](../README.md#64-deepseek-v4---hybrid-attention-csa--hca))과
어긋나거나 빠진 것으로 확인된 것들입니다.

1. **"4토큰을 하나로 묶는다"는 표현이 부정확합니다.** 학습된 게이트 softmax 풀링이고, 4:1은
   창이 겹칩니다. 본문 수준에서는 "묶는다"로 두더라도 각주에 실제 동작을 적어야 합니다.
2. **V4의 어텐션을 "MQA"로만 적으면 절반입니다.** 업프로젝션 없는 공유 512 벡터이고 K = V라는
   점, 그리고 그것이 흡수된 MLA와 같다는 관계를 밝혀야 합니다.
3. **Hyper-Connections가 빠져 있습니다.** 6.6의 AttnRes를 "앞에 나오지 않은 유일한 항목"으로
   적은 문장과 충돌합니다.
4. **인덱서 전용 캐시**가 KV 예산 서술에 없습니다.
5. **Pro의 레이어 구성**(CSA 30 · HCA 31, 첫 두 층이 HCA)이 문서에 없습니다.
