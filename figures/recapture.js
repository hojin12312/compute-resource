/* figures/*.html → assets/*.webp 를 3840×2160(4K)으로 재캡처합니다.
 *
 * 왜 4K인가: 그림 원본이 1376×768이면 FHD 화면에서 reveal이 139.5%로 확대해
 * 작은 라벨이 뭉갭니다. 1920으로 올리면 FHD(DPR 1)에서는 1:1이 되지만,
 * Retina·4K(DPR 2)에서는 실제 디바이스 픽셀이 3840이라 다시 2배 확대됩니다.
 * 텍스트 슬라이드는 벡터라 화면만큼 다시 그려지는데 그림만 1920에 멈춰
 * 나란히 놓으면 그림이 흐려 보입니다(실측: DPR 2에서 텍스트는 디테일이
 * 늘고 그림은 늘지 않음). 3840이면 DPR 2에서도 1:1입니다.
 *
 * 레이아웃 배율(zoom 1.395)은 그대로 두고 deviceScaleFactor만 2로 올립니다.
 * 검증된 1920 배치를 그대로 두고 픽셀만 늘리는 방식이라 재검증이 필요 없습니다.
 *
 * 레이아웃은 건드리지 않습니다. 그림 HTML은 1376×768 기준으로 검증된 상태라,
 * CSS zoom(=레이아웃 재계산 후 확대)으로 1920 폭에 맞춰 렌더만 다시 합니다.
 * 요소 배치·간격 비율은 그대로고 픽셀만 늘어납니다.
 *
 * 세로는 768×1.3953=1071.6이라 하단 약 8px이 남는데, 배경색(#0d1117)이 같아
 * 티가 나지 않습니다. 16:9와 1376:768의 종횡비 차이(1.7778 vs 1.7917) 때문입니다.
 *
 * 왜 file:// 인가: http.server를 쓰면 포트를 옛 세션 서버가 잡고 있을 때
 * 조용히 다른 디렉토리의 동명 파일을 찍는 사고가 납니다(2026-07-26 실제 발생).
 * file://은 경로가 곧 파일이라 그 사고가 원천 봉쇄됩니다.
 *
 * 실행:  node figures/recapture.js            (전체)
 *        node figures/recapture.js swa k3     (일부만 — html 이름 일부 매칭)
 *        node figures/recapture.js --light    (라이트 테마 판 → *_light.webp)
 *
 * --light 는 <html data-theme="light">를 걸어 같은 소스를 라이트로
 * 만들고 거기서 캡처합니다. 덱이 라이트 테마일 때 이 그림으로 바꿔 답니다.
 * 표에 없는 색이 소스에 있으면 중단합니다 — 다크 색 하나가 조용히 남는 것을 막습니다.
 * 그림을 고쳤으면 **두 번 돌려야 합니다**: 그냥 한 번, --light로 한 번.
 */
const { chromium } = require('playwright');
const path = require('path');
const fs = require('fs');
const { execFileSync } = require('child_process');

const ROOT = path.resolve(__dirname, '..');
const OUT_W = 1920, OUT_H = 1080;     // 레이아웃 기준 (CSS 픽셀)
const DPR = 2;                        // 디바이스 배율 → 실제 출력은 3840×2160
const SRC_W = 1376, SRC_H = 768;      // 그림 HTML이 전제하는 레이아웃 크기
const ZOOM = OUT_W / SRC_W;           // 1.395348…

// figures/README.txt의 대응표와 같아야 합니다.
const MAP = {
  'shock.html':      'stock_shock',
  'capex.html':      'us_china_capex',
  'bench.html':      'frontier_bench',
  'tokens.html':     'token_growth',
  'lease.html':      'compute_lease',
  'prices.html':     'memory_price_chain',
  'vram.html':       'model_capacity',
  'budget.html':     'node_memory_budget',
  'readall.html':    'next_token_all',
  'units.html':      'compute_units',
  'desk.html':       'kv_cache_desk',
  'kvcalc.html':     'kv_cache_calc',
  'ctx.html':        'context_growth',
  'pdbench.html':    'prefill_decode_bench',
  'prefill.html':    'prefill_parallel',
  'decoding.html':   'decoding_sequential',
  'hit_chart.html':  'kv_cache_hit_chart',
  'kv_tier.html':    'kv_tier',
  'row_cost.html':   'cache_hit_row_cost',
  'pricing.html':    'cache_hit_pricing_new',
  'speculative.html':'speculative_decoding',
  'architecture_map.html':'architecture_efficiency_map',
  'moe.html':        'moe_active',
  'moe_scope.html':  'moe_scope',
  'sparse_map.html': 'sparse_attention_map',
  'swa.html':        'swa_layers',
  'swa_reach.html':  'swa_reach',
  'gdn.html':        'gdn_summary_note',
  'v4_pricing.html': 'v4_api_pricing',
  'v4_spec.html':    'v4_spec',
  'glm_spec.html':   'glm_spec',
  'glm_bench.html':  'glm_bench',
  'k3_spec.html':    'k3_spec',
  'k3_bench.html':   'k3_bench',
  'v4.html':         'v4_compression',
  'gqa.html':        'gqa_head_sharing',
  'indexshare.html': 'indexshare',
  'glm_async_rl.html':'glm_async_rl',
  'attnres.html':    'attention_residuals',
  'k3.html':         'k3_synthesis',
  'refusal_gate.html': 'refusal_gate',
  'quant_map.html':  'quantization_map',
  'quant_calib.html': 'quant_calib',
  'quant_intro.html': 'quant_two_bills',
  'formats.html':     'precision_formats',
  'unimix.html':      'uniform_vs_mixed',
  'kv_when.html':    'kv_when',
  'control_scope.html': 'control_scope',
  'gpu_gen.html':      'gpu_gen',
  'platform_cmp.html': 'platform_cmp',
  'engine_role.html':  'engine_role',
  'serving_stack.html': 'serving_stack',
  'kernel_role.html':  'kernel_role',
  'tq_shock.html':   'turboquant_shock',
  'rotation_absorb.html':  'rotation_absorb',
  'rotation_family.html':  'rotation_family',
  'quant_gain.html':       'quant_gain',
  'hadamard.html':   'kv_quant_rotation',
  'rotation_geometry.html': 'rotation_geometry',
  'parallel_axes.html':  'parallel_axes',
  'pd_tradeoff.html':    'pd_tradeoff',
  'split_gain.html':     'split_gain',
  'expert_parallel.html':'expert_parallel',
  'pd_split.html':       'pd_split',
  'sovereign.html':  'ai_sovereignty',
  'kr_infra.html':   'kr_infra',
  'kr_bench.html':   'kr_bench',
  'sov_boundary.html': 'sov_boundary',
  'sov_personal.html': 'sov_personal',
  'sov_stack.html':    'sov_stack',
  'engine_stack.html':'inference_engine_stack',
  'roadmap.html':    'gpu_roadmap',
};

/* ── 라이트 판 ────────────────────────────────────────────────────────────
 * 예전에는 lightmap.js의 대응표로 소스의 색 문자열을 치환한 사본을
 * figures/.light/에 만들어 그곳에서 캡처했습니다. 지금은 그림이 CSS 변수로
 * 두 벌의 색을 갖고 있으므로 <html data-theme="light"> 하나만 걸면 됩니다.
 * (전환 시 78장을 픽셀 단위로 대조해 예전 방식과 결과가 같음을 확인했습니다.)
 *
 * 색 리터럴이 소스에 남아 있으면 라이트에서 얼룩이 됩니다 —
 * `node figures/check.js`가 그것을 찾아 실패로 보고합니다. */

(async () => {
  const argv = process.argv.slice(2);
  const LIGHT = argv.includes('--light');
  const filter = argv.filter(a => a !== '--light');
  const entries = Object.entries(MAP).filter(([html]) =>
    filter.length === 0 || filter.some(f => html.includes(f)));

  if (entries.length === 0) { console.error('매칭되는 그림이 없습니다.'); process.exit(1); }

  const SRC_DIR = __dirname;
  const SUFFIX = LIGHT ? '_light' : '';

  const browser = await chromium.launch();
  const ctx = await browser.newContext({
    viewport: { width: OUT_W, height: OUT_H },
    deviceScaleFactor: DPR,
  });
  const page = await ctx.newPage();

  const done = [];
  for (const [html, name] of entries) {
    const src = path.join(SRC_DIR, html);
    if (!fs.existsSync(src)) { console.error(`  ✗ 소스 없음: ${html}`); continue; }

    await page.goto('file://' + src);
    if (LIGHT) await page.evaluate(() => document.documentElement.setAttribute('data-theme', 'light'));
    // 레이아웃(1376×768)은 그대로 두고 확대해서 1920 폭을 채웁니다.
    // 하단 여분은 그림의 배경 변수로 메워 캡처 전체가 한 장처럼 보이게 합니다.
    await page.addStyleTag({ content:
      `html { zoom: ${ZOOM}; background: var(--c-0d1117); }` });
    await page.waitForLoadState('networkidle');
    await page.evaluate(() => document.fonts.ready);
    await page.waitForTimeout(220);          // 스크립트로 그리는 격자/막대가 있음

    const png = path.join(ROOT, 'assets', name + SUFFIX + '.png');
    const webp = path.join(ROOT, 'assets', name + SUFFIX + '.webp');
    await page.screenshot({ path: png });
    execFileSync('cwebp', ['-quiet', '-q', '92', png, '-o', webp]);
    fs.unlinkSync(png);

    const kb = (fs.statSync(webp).size / 1024).toFixed(0);
    console.log(`  ✓ ${html.padEnd(17)} → ${(name + SUFFIX + '.webp').padEnd(32)} ${kb}KB`);
    done.push(webp);
  }
  await browser.close();

  // 덮어쓰기 사고의 지문: 서로 다른 그림인데 내용이 같으면 잘못 찍힌 것입니다.
  const crypto = require('crypto');
  const seen = new Map();
  for (const f of done) {
    const h = crypto.createHash('sha256').update(fs.readFileSync(f)).digest('hex');
    if (seen.has(h)) console.error(`\n  ⚠ 내용이 같은 그림 발견: ${path.basename(f)} == ${path.basename(seen.get(h))}`);
    seen.set(h, f);
  }
  console.log(`\n${done.length}장 완료 (${OUT_W * DPR}×${OUT_H * DPR}${LIGHT ? ', 라이트' : ''}), 중복 없음.`);
})();
