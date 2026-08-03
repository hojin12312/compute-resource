/* 그림 47장의 폰트 감사 — check.js가 못 잡는 것을 잡습니다.
 *
 *   export NODE_PATH="/Users/studio/.npm/_npx/420ff84f11983ee5/node_modules"
 *   node figures/fontcheck.js            전체
 *   node figures/fontcheck.js swa k3     일부만 (이름 일부 매칭)
 *
 * 왜 필요한가: check.js는 색 리터럴·넘침·잘림·고아줄을 보지만 **어떤 폰트로 실제로
 * 그려졌는지는 보지 않습니다.** 그래서 2026-07-31까지 다섯 그림이
 * `"SFMono-Regular", monospace`를 쓰고 있었고, 그 폰트는 이 맥에서도 해결되지 않아
 * **Courier로 폴백**됐습니다(글자가 상자 중앙에서 벗어나 보이는 원인). 같은 날
 * 모노 지정 요소 58곳에 한글이 들어 있는 것도 드러났습니다 — IBM Plex Mono에는 한글이
 * 없어 Apple SD Gothic Neo(시스템 폰트)로 폴백됩니다.
 *
 * 검사 방법: CDP `CSS.getPlatformFontsForNode`로 **실제 렌더에 쓰인 플랫폼 폰트**를
 * 물어봅니다. 선언이 아니라 결과를 보는 것이라 폴백을 놓치지 않습니다.
 * 허용 목록은 루트 fonts/가 싣는 두 벌뿐입니다 — Pretendard Variable · IBM Plex Mono.
 *
 * 다크·라이트 두 테마를 모두 봅니다(테마가 폰트를 바꾸지는 않지만, 테마별로 숨었다
 * 나타나는 요소가 있으면 글자도 달라집니다).
 */
'use strict';
const fs = require('fs');
const path = require('path');
const { chromium } = require('playwright');

const ALLOW = /^(Pretendard Variable|IBM Plex Mono)$/;

async function main() {
  const argv = process.argv.slice(2);
  const filter = argv.filter(a => !a.startsWith('--'));
  const files = fs.readdirSync(__dirname)
    .filter(f => f.endsWith('.html'))
    .filter(f => filter.length === 0 || filter.some(k => f.includes(k)))
    .sort();
  if (!files.length) { console.error('매칭되는 그림이 없습니다.'); process.exit(1); }

  const browser = await chromium.launch();
  const page = await browser.newPage({ viewport: { width: 1376, height: 774 } });
  const cdp = await page.context().newCDPSession(page);
  await cdp.send('DOM.enable');
  await cdp.send('CSS.enable');

  const bad = [];
  for (const f of files) {
    for (const theme of ['dark', 'light']) {
      await page.goto('file://' + path.join(__dirname, f));
      if (theme === 'light') {
        await page.evaluate(() => document.documentElement.setAttribute('data-theme', 'light'));
      }
      await page.evaluate(() => document.fonts.ready);
      await page.waitForTimeout(70);
      const { root } = await cdp.send('DOM.getDocument');
      const { nodeId } = await cdp.send('DOM.querySelector', { nodeId: root.nodeId, selector: 'body' });
      const { fonts } = await cdp.send('CSS.getPlatformFontsForNode', { nodeId });
      const off = fonts.filter(x => x.glyphCount > 0 && !ALLOW.test(x.familyName));
      if (off.length) {
        bad.push({ f, theme, off });
        console.log('  ✗ ' + f.padEnd(24) + theme.padEnd(6) +
          off.map(x => x.familyName + ' ' + x.glyphCount + '자').join(' · '));
      }
    }
  }

  /* 선언 쪽도 함께 봅니다 — 지금은 통과해도 다음에 시스템 폰트를 적어 넣는 것을 막습니다 */
  const declared = [];
  for (const f of files) {
    const src = fs.readFileSync(path.join(__dirname, f), 'utf8');
    const m = src.match(/font-family\s*:[^;}]+|font\s*:\s*[^;}]*\d(?:px|em)[^;}]+/g) || [];
    m.forEach(decl => {
      const names = decl.match(/"[^"]+"|'[^']+'/g) || [];
      names.forEach(n => {
        const clean = n.replace(/['"]/g, '');
        if (!ALLOW.test(clean)) declared.push(f + ' → ' + clean);
      });
    });
  }
  if (declared.length) {
    console.log('\n  선언에 남은 비허용 폰트:');
    [...new Set(declared)].forEach(d => console.log('    ✗ ' + d));
  }

  const ok = bad.length === 0 && declared.length === 0;
  console.log('\n  ' + files.length + '장 × 2테마 · 폴백 ' + bad.length + '건 · 선언 위반 ' +
    [...new Set(declared)].length + '건  ' + (ok ? '전부 통과' : '✗ 손봐야 합니다'));
  await browser.close();
  process.exit(ok ? 0 : 1);
}

main();
