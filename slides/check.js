/* 덱 110장 회귀 검사.
 *
 *   node slides/check.js            다크
 *   node slides/check.js --light    라이트
 *   node slides/check.js --json out.json
 *
 * 폰트를 바꾸면 자폭이 달라져 글이 슬라이드 밖으로 밀려도 조용합니다. 발표 중에는
 * 알아챌 방법이 없으므로 여기서 잡습니다.
 *
 * 검사 항목
 *   1) 슬라이드 밖으로 나간 요소 (1376×774 경계)
 *   2) 세로 넘침 (scrollHeight > clientHeight)
 *   3) 잘린 텍스트 (scrollWidth > clientWidth)
 *   4) 그림 슬라이드가 실제로 그려졌는지 (iframe이면 내부까지)
 */
'use strict';
const fs = require('fs');
const { chromium } = require('playwright');

const URL = process.env.DECK_URL || 'http://localhost:8899/slides/';

async function main() {
  const argv = process.argv.slice(2);
  const light = argv.includes('--light');
  const jsonAt = argv.indexOf('--json');
  const jsonPath = jsonAt >= 0 ? argv[jsonAt + 1] : null;

  const browser = await chromium.launch();
  const page = await browser.newPage({ viewport: { width: 1600, height: 900 } });
  const errs = [];
  page.on('pageerror', e => errs.push('JS: ' + e.message));
  const bad = [];
  page.on('response', r => { if (r.status() >= 400) bad.push(r.status() + ' ' + r.url().split('/').pop()); });

  await page.goto(URL, { waitUntil: 'networkidle' });
  await page.evaluate(() => document.fonts.ready);
  await page.waitForTimeout(600);

  if (light) {
    await page.evaluate(() => {
      if (document.documentElement.getAttribute('data-theme') !== 'light') {
        document.getElementById('tb-theme')
          ? document.getElementById('tb-theme').click()
          : document.documentElement.setAttribute('data-theme', 'light');
      }
    });
    await page.waitForTimeout(500);
  }

  const total = await page.evaluate(() => Reveal.getTotalSlides());
  const rows = [];

  for (let i = 0; i < total; i++) {
    await page.evaluate(n => Reveal.slide(0, 0, 0) && 0, 0);   /* no-op guard */
    await page.evaluate(n => {
      const s = Reveal.getSlides()[n];
      const idx = Reveal.getIndices(s);
      Reveal.slide(idx.h, idx.v || 0);
    }, i);
    await page.waitForTimeout(160);

    const r = await page.evaluate(() => {
      const s = Reveal.getCurrentSlide();
      const sr = s.getBoundingClientRect();
      const scale = sr.width / 1376;
      const out = { outside: [], vscroll: 0, clipped: [], fig: null, chapter: s.dataset.chapter || '' };

      /* .cnum(장 번호 장식)은 일부러 슬라이드 밖으로 흘려 놓은 것입니다.
         제외하지 않으면 장 구분 슬라이드 9장이 늘 위반으로 잡혀 신호가 묻힙니다. */
      const BY_DESIGN = '.cnum';

      s.querySelectorAll('*').forEach(el => {
        if (el.closest('.notes')) return;
        if (el.matches(BY_DESIGN) || el.closest(BY_DESIGN)) return;
        const st = getComputedStyle(el);
        if (st.display === 'none' || st.visibility === 'hidden') return;
        const r = el.getBoundingClientRect();
        if (r.width === 0 || r.height === 0) return;
        const over = Math.max(r.right - sr.right, sr.left - r.left, r.bottom - sr.bottom, sr.top - r.top);
        if (over > 1.5) {
          out.outside.push({ sel: (el.className || el.tagName) + '', over: Math.round(over / scale) });
        }
        if (el.children.length === 0 && el.scrollWidth > el.clientWidth + 1 && el.clientWidth > 0) {
          out.clipped.push({ sel: (el.className || el.tagName) + '', text: (el.textContent || '').slice(0, 20) });
        }
      });

      /* 세로 넘침도 .cnum이 만듭니다 — 잠시 숨겨 재고 되돌립니다. */
      const deco = s.querySelector(BY_DESIGN);
      const keep = deco ? deco.style.display : null;
      if (deco) deco.style.display = 'none';
      out.vscroll = Math.round(Math.max(0, s.scrollHeight - s.clientHeight));
      if (deco) deco.style.display = keep;

      if (s.classList.contains('fig')) {
        const frame = s.querySelector('iframe');
        const img = s.querySelector('img');
        if (frame) {
          let n = null;
          try { n = frame.contentDocument && frame.contentDocument.body ? frame.contentDocument.body.children.length : null; }
          catch (e) { n = 'blocked'; }
          out.fig = { kind: 'iframe', inner: n, src: (frame.getAttribute('src') || '').split('/').pop() };
        } else if (img) {
          out.fig = { kind: 'img', ok: img.complete && img.naturalWidth > 0, src: (img.getAttribute('src') || '').split('/').pop() };
        } else {
          out.fig = { kind: 'none' };
        }
      }
      return out;
    });

    const msgs = [];
    if (r.outside.length) msgs.push('밖으로 ' + r.outside.length + '건(최대 ' + Math.max(...r.outside.map(o => o.over)) + 'px)');
    if (r.vscroll > 2) msgs.push('세로넘침 ' + r.vscroll + 'px');
    if (r.clipped.length) msgs.push('잘림 ' + r.clipped.length + '건: ' + r.clipped.slice(0, 2).map(c => c.text).join(' / '));
    if (r.fig) {
      if (r.fig.kind === 'none') msgs.push('그림 없음');
      else if (r.fig.kind === 'iframe' && (r.fig.inner === 'blocked' || !r.fig.inner)) msgs.push('그림 iframe 비어 있음');
      else if (r.fig.kind === 'img' && !r.fig.ok) msgs.push('그림 이미지 로드 실패');
    }
    rows.push({ n: i + 1, chapter: r.chapter, msgs, fig: r.fig });
  }

  await browser.close();

  const bads = rows.filter(x => x.msgs.length);
  console.log('\n' + (light ? '라이트' : '다크') + ' · ' + total + '장 검사\n');
  for (const x of bads) console.log(`  ✗ ${String(x.n).padStart(2)}장 ${x.chapter.padEnd(18)} ${x.msgs.join(' | ')}`);
  const kinds = rows.filter(x => x.fig).reduce((a, x) => { a[x.fig.kind] = (a[x.fig.kind] || 0) + 1; return a; }, {});
  console.log('\n  그림 슬라이드: ' + JSON.stringify(kinds));
  console.log('  통과 ' + (total - bads.length) + ' / ' + total + (bads.length ? '' : '  전부 통과'));
  console.log('  4xx: ' + (bad.length ? bad.slice(0, 5).join(', ') : '없음') + ' | JS오류: ' + (errs.length ? errs.slice(0, 3).join(' / ') : '없음'));

  if (jsonPath) fs.writeFileSync(jsonPath, JSON.stringify(rows, null, 1));
  process.exit(bads.length || errs.length ? 1 : 0);
}

main();
