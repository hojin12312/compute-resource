/* 그림 39장 레이아웃 회귀 검사.
 *
 *   node figures/check.js              모든 그림
 *   node figures/check.js v4_spec k3   일부만
 *   node figures/check.js --light      라이트 테마로
 *
 * 검사 항목과 그 이유:
 *   1) 패널 넘침 — 인라인 width 눌림 검사는 CSS grid 넘침을 못 잡습니다. 그래서
 *      패널의 getBoundingClientRect().right와 자식의 right를 따로 비교합니다.
 *   2) 캔버스 넘침 — 1376×768 밖으로 나간 요소(캡처에서 잘립니다).
 *   3) .foot 하단 여백 — 110px을 넘으면 그림이 비어 보입니다.
 *   4) 고아 줄 — 마지막 줄에 한두 글자만 남는 것. 블록 요소의 getClientRects()는
 *      1줄로 나오므로 Range.getClientRects()로 재야 합니다.
 *   5) 잘린 텍스트 — scrollWidth가 clientWidth를 넘는 요소(… 없이 잘립니다).
 *   6) 폰트 — Pretendard와 Plex Mono가 실제로 적용됐는지. 시스템 폰트로 떨어지면
 *      이 맥에서만 맞는 레이아웃으로 되돌아갑니다.
 */
'use strict';
const path = require('path');
const fs = require('fs');
const { chromium } = require('playwright');

const SRC_W = 1376, SRC_H = 768;
const FOOT_GAP_MAX = 110;

async function main() {
  const argv = process.argv.slice(2);
  const light = argv.includes('--light');
  const NOFONT = argv.includes('--nofont');
  const jsonAt = argv.indexOf('--json');
  const jsonPath = jsonAt >= 0 ? argv[jsonAt + 1] : null;
  const filter = argv.filter((a, i) => !a.startsWith('--') && !(jsonAt >= 0 && i === jsonAt + 1));

  let files = fs.readdirSync(__dirname).filter(f => f.endsWith('.html'));
  if (filter.length) files = files.filter(f => filter.some(k => f.includes(k)));
  files.sort();

  /* ── 0) 색 리터럴 린트 (브라우저 없이) ───────────────────────────────────
     그림은 base.css의 변수 두 벌로 테마를 갖습니다. 소스에 색을 직접 쓰면
     라이트에서 다크 색이 얼룩으로 남습니다. 주석은 색을 설명하는 글이라 넘깁니다. */
  const litFiles = [...files, 'base.css'];
  const lint = [];
  for (const f of litFiles) {
    const text = fs.readFileSync(path.join(__dirname, f), 'utf8');
    const isCss = f.endsWith('.css');
    /* 주석을 같은 길이의 공백으로 덮어 위치를 유지합니다 */
    const masked = text.replace(isCss ? /\/\*[\s\S]*?\*\//g : /<!--[\s\S]*?-->|\/\*[\s\S]*?\*\//g,
      m => ' '.repeat(m.length));
    /* :root 블록의 변수 정의부는 색이 있어야 하는 곳입니다 */
    const body = f === 'base.css'
      ? masked.replace(/:root[^{]*\{[\s\S]*?\}/g, m => ' '.repeat(m.length))
      : masked;
    const found = body.match(/#[0-9a-fA-F]{6}\b|#[0-9a-fA-F]{3}\b|rgba?\([^)]*\)/g) || [];
    const uniq = [...new Set(found)];
    if (uniq.length) lint.push({ file: f, colors: uniq });
  }
  if (lint.length) {
    console.log('\n  ✗ 색 리터럴이 남아 있습니다 — base.css에 변수 두 벌을 추가하고 var()로 바꾸세요:');
    for (const l of lint) console.log('      ' + l.file.padEnd(24) + l.colors.join(' '));
  } else {
    console.log('\n  · 색 리터럴 0건 (테마 전환 안전)');
  }

  const browser = await chromium.launch();
  const page = await browser.newPage({ viewport: { width: SRC_W, height: SRC_H } });
  const rows = [];

  for (const file of files) {
    const errs = [];
    page.removeAllListeners('pageerror');
    page.on('pageerror', e => errs.push('JS: ' + e.message));

    await page.goto('file://' + path.join(__dirname, file));
    if (light) await page.evaluate(() => document.documentElement.setAttribute('data-theme', 'light'));
    await page.evaluate(() => document.fonts.ready);
    await page.waitForTimeout(120);

    const r = await page.evaluate(({ SRC_W, SRC_H, FOOT_GAP_MAX }) => {
      const out = { spill: [], canvas: [], clipped: [], orphans: [], footGap: null, fonts: {} };

      const faces = [...document.fonts].map(f => ({ family: f.family, weight: f.weight, status: f.status }));
      out.fonts.loaded = faces.filter(f => f.status === 'loaded').map(f => f.family + ' ' + f.weight);
      out.fonts.bodyFamily = getComputedStyle(document.body).fontFamily.split(',')[0].replace(/"/g, '');
      out.fonts.pretendard = faces.some(f => f.family === 'Pretendard Variable' && f.status === 'loaded');
      /* 모노는 실제로 쓰는 요소가 있을 때만 따집니다 */
      out.fonts.monoUsed = [...document.querySelectorAll('body *')].some(el =>
        getComputedStyle(el).fontFamily.indexOf('Plex Mono') === 0);
      out.fonts.mono = !out.fonts.monoUsed ||
        faces.some(f => f.family === 'IBM Plex Mono' && f.status === 'loaded');
      /* 폴백으로 떨어졌는지 실측: 같은 문장을 웹폰트와 시스템 폰트로 재서 폭 비교 */
      const probe = (ff) => { const s = document.createElement('span');
        s.style.cssText = 'position:absolute;visibility:hidden;white-space:pre;font-size:40px;font-family:' + ff;
        s.textContent = '컴퓨팅 자원의 역학관계 0123'; document.body.appendChild(s);
        const w = s.getBoundingClientRect().width; s.remove(); return Math.round(w); };
      out.fonts.wPretendard = probe('"Pretendard Variable"');
      out.fonts.wSystem = probe('sans-serif');

      /* 1) 패널 넘침 */
      document.querySelectorAll('.panel, .card, .box, .col').forEach(p => {
        const pr = p.getBoundingClientRect();
        if (pr.width === 0) return;
        p.querySelectorAll('*').forEach(c => {
          const cr = c.getBoundingClientRect();
          if (cr.width === 0) return;
          const over = Math.round(Math.max(cr.right - pr.right, pr.left - cr.left));
          if (over > 1) out.spill.push({ sel: (c.className || c.tagName) + '', over });
        });
      });

      /* 2) 캔버스 넘침 */
      document.querySelectorAll('body *').forEach(el => {
        const r = el.getBoundingClientRect();
        if (r.width === 0 || r.height === 0) return;
        if (getComputedStyle(el).position === 'fixed') return;
        const over = Math.round(Math.max(r.right - SRC_W, r.bottom - SRC_H));
        if (over > 1) out.canvas.push({ sel: (el.className || el.tagName) + '', over });
      });

      /* 3) .foot 하단 여백 */
      const foot = document.querySelector('.foot');
      if (foot) {
        const ft = foot.getBoundingClientRect().top;
        let maxBottom = 0;
        document.querySelectorAll('body *').forEach(el => {
          if (el === foot || foot.contains(el)) return;
          const r = el.getBoundingClientRect();
          if (r.width === 0 || r.height === 0) return;
          if (r.bottom > maxBottom && r.top < ft) maxBottom = r.bottom;
        });
        out.footGap = Math.round(ft - maxBottom);
      }

      /* 4) 고아 줄 — 텍스트를 직접 가진 요소만 */
      document.querySelectorAll('p, .cap, .sub, .note, .foot, li, .pdesc, .ptitle').forEach(el => {
        const t = (el.textContent || '').trim();
        if (t.length < 24) return;
        const range = document.createRange();
        range.selectNodeContents(el);
        const lines = [...range.getClientRects()].filter(r => r.width > 0);
        if (lines.length < 2) return;
        const last = lines[lines.length - 1], first = lines[0];
        if (last.width < first.width * 0.12 && last.width < 40) {
          out.orphans.push({ sel: (el.className || el.tagName) + '', lastWidth: Math.round(last.width) });
        }
      });

      /* 5) 잘린 텍스트 */
      document.querySelectorAll('body *').forEach(el => {
        if (el.children.length) return;
        const st = getComputedStyle(el);
        if (st.overflow === 'visible' && st.textOverflow !== 'ellipsis') return;
        if (el.scrollWidth > el.clientWidth + 1 && el.clientWidth > 0) {
          out.clipped.push({ sel: (el.className || el.tagName) + '', text: (el.textContent || '').slice(0, 22) });
        }
      });

      return out;
    }, { SRC_W, SRC_H, FOOT_GAP_MAX });

    if (!NOFONT) {
      if (!r.fonts.pretendard) errs.push('폰트: Pretendard 미로드');
      if (!r.fonts.mono) errs.push('폰트: Plex Mono 미로드');
      if (r.fonts.bodyFamily !== 'Pretendard Variable') errs.push('폰트: body가 ' + r.fonts.bodyFamily);
    }
    if (r.spill.length) errs.push('패널넘침 ' + r.spill.length + '건 (최대 ' + Math.max(...r.spill.map(s => s.over)) + 'px)');
    if (r.canvas.length) errs.push('캔버스넘침 ' + r.canvas.length + '건 (최대 ' + Math.max(...r.canvas.map(s => s.over)) + 'px)');
    if (r.footGap !== null && r.footGap > FOOT_GAP_MAX) errs.push('foot여백 ' + r.footGap + 'px');
    if (r.footGap !== null && r.footGap < 0) errs.push('foot겹침 ' + r.footGap + 'px');
    if (r.orphans.length) errs.push('고아줄 ' + r.orphans.length + '건');
    if (r.clipped.length) errs.push('잘림 ' + r.clipped.length + '건: ' + r.clipped.slice(0, 2).map(c => c.text).join(' / '));

    rows.push({ file, errs, detail: r });
  }

  await browser.close();

  if (jsonPath) {
    fs.writeFileSync(jsonPath, JSON.stringify(rows.map(r => ({ file: r.file, footGap: r.detail.footGap,
      spill: r.detail.spill.length, canvas: r.detail.canvas.length, clipped: r.detail.clipped.length,
      orphans: r.detail.orphans.length, wPretendard: r.detail.fonts.wPretendard,
      wSystem: r.detail.fonts.wSystem, bodyFamily: r.detail.fonts.bodyFamily })), null, 1));
    console.log('  → ' + jsonPath + ' 에 기록');
  }
  const bad = rows.filter(r => r.errs.length);
  console.log('\n' + (light ? '라이트' : '다크') + ' · ' + rows.length + '장 검사\n');
  for (const r of rows) {
    const mark = r.errs.length ? '✗' : '·';
    const gap = r.detail.footGap === null ? '  -' : String(r.detail.footGap).padStart(3);
    console.log(`  ${mark} ${r.file.padEnd(24)} foot ${gap}  ${r.errs.join(' | ')}`);
  }
  console.log('\n  통과 ' + (rows.length - bad.length) + ' / ' + rows.length + (bad.length ? '  ✗ 손봐야 할 그림 ' + bad.length + '장' : '  전부 통과'));
  process.exit(bad.length || lint.length ? 1 : 0);
}

main();
