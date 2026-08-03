/* figures/manifest.json 을 recapture.js의 MAP에서 다시 만듭니다.
 *
 * 읽기용 문서(/document/)와 발표용 덱(/slides/)이 둘 다 이 파일을 읽어
 * 에셋 이름으로 그림 소스를 찾고, **내용 해시로 캐시를 깹니다.**
 *
 *   node figures/manifest.js          쓰기
 *   node figures/manifest.js --check   어긋났는지만 확인 (CI용)
 *
 * 왜 필요한가 1 — 이름 대응: 읽기용 문서는 README의 `./assets/v4_spec.webp` 같은 참조를
 * 보고 그에 대응하는 그림 소스 `figures/v4_spec.html`을 iframe으로 띄웁니다. 그런데
 * 이름이 다른 것이 33개나 됩니다(`prices.html` → `memory_price_chain.webp`).
 * 그 대응은 recapture.js의 MAP이 유일한 원천이므로, 손으로 두 번 적지 않고
 * 여기서 뒤집어 내보냅니다.
 *
 * 왜 필요한가 2 — 캐시: GitHub Pages는 `max-age=600`으로 내보냅니다. 그림을 고쳐 푸시해도
 * 이미 열어 본 브라우저는 하드 리프레시 없이는 10분간 옛 그림을 그립니다. 그래서
 *
 *   1. 그림 파일마다 내용 해시(`v`)를 실어 보내고, 문서·덱이 `?v=<해시>`를 붙여 띄웁니다.
 *      내용이 바뀌면 URL이 바뀌므로 브라우저가 새로 받습니다.
 *   2. `manifest.json` 자체는 두 소비자가 `cache: 'no-store'`로 읽으므로 항상 최신입니다.
 *   3. `base.css`가 바뀌면 그림 HTML 안의 `href="base.css?v=…"`를 이 스크립트가 다시
 *      찍습니다. 그러면 39장의 해시가 함께 바뀌어 CSS까지 새로 받습니다.
 *
 * **그림이나 base.css를 고쳤으면 이 스크립트를 다시 돌리세요.** 안 돌리면 `--check`가
 * 실패하고, 그대로 배포하면 옛 화면이 남습니다.
 */
'use strict';
const fs = require('fs');
const path = require('path');
const crypto = require('crypto');

const ROOT = path.resolve(__dirname, '..');
const OUT = path.join(ROOT, 'figures', 'manifest.json');
const CSS = 'base.css';

const hash8 = s => crypto.createHash('sha256').update(s).digest('hex').slice(0, 8);

/* recapture.js를 실행하지 않고 MAP만 떼어 읽습니다 (그 파일은 즉시 캡처를 시작합니다) */
function readMap() {
  const src = fs.readFileSync(path.join(__dirname, 'recapture.js'), 'utf8');
  const start = src.indexOf('const MAP = {');
  const end = src.indexOf('};', start);
  if (start < 0 || end < 0) throw new Error('recapture.js에서 MAP을 찾지 못했습니다');
  const body = src.slice(start, end);
  const map = {};
  const re = /'([^']+\.html)'\s*:\s*'([^']+)'/g;
  let m;
  while ((m = re.exec(body))) map[m[1]] = m[2];
  return map;
}

/* 그림 HTML의 base.css 링크에 현재 CSS 해시를 찍습니다. 이미 맞으면 그대로 둡니다. */
function stampCss(text, cssVer) {
  const re = /(<link[^>]+href=")base\.css(?:\?v=[0-9a-f]+)?(")/g;
  if (!re.test(text)) throw new Error('base.css 링크를 찾지 못했습니다');
  return text.replace(re, '$1' + CSS + '?v=' + cssVer + '$2');
}

function build({ write }) {
  const map = readMap();
  const cssVer = hash8(fs.readFileSync(path.join(__dirname, CSS), 'utf8'));

  const figures = {};
  const v = {};
  const drift = [];

  for (const [html, asset] of Object.entries(map)) {
    if (figures[asset]) throw new Error('에셋 이름이 겹칩니다: ' + asset);
    const file = path.join(__dirname, html);
    if (!fs.existsSync(file)) throw new Error('소스가 없습니다: ' + html);

    const cur = fs.readFileSync(file, 'utf8');
    const next = stampCss(cur, cssVer);
    if (next !== cur) {
      if (write) fs.writeFileSync(file, next);
      else drift.push(html);
    }
    figures[asset] = html;
    v[html] = hash8(next);
  }

  /* README가 참조하는 그림이 전부 매핑에 있는지 */
  const md = fs.readFileSync(path.join(ROOT, 'README.md'), 'utf8');
  const refs = [...md.matchAll(/!\[[^\]]*\]\(\.\/assets\/([^)]+)\.webp\)/g)].map(m => m[1]);
  const unmapped = refs.filter(r => !figures[r]);
  if (unmapped.length) throw new Error('README가 쓰는데 매핑이 없습니다: ' + unmapped.join(', '));

  /* 캡처본(webp) 되돌림용 한 벌짜리 토큰. 그림이 하나라도 바뀌면 함께 바뀝니다. */
  const ver = hash8(Object.keys(v).sort().map(k => k + ':' + v[k]).join('|'));

  const json = {
    '_주석': 'assets 이름 → figures 소스와 내용 해시. figures/recapture.js의 MAP에서 '
           + '생성합니다 (node figures/manifest.js). 직접 고치지 마세요. '
           + 'v는 그림별 캐시 버스터, ver은 캡처본용 한 벌 토큰, css는 base.css 해시입니다.',
    ver,
    css: cssVer,
    figures,
    v,
  };
  return { json, drift };
}

const check = process.argv.includes('--check');
const { json, drift } = build({ write: !check });
const next = JSON.stringify(json, null, 1) + '\n';
const count = Object.keys(json.figures).length;

if (check) {
  const cur = fs.existsSync(OUT) ? fs.readFileSync(OUT, 'utf8') : '';
  if (drift.length) {
    console.error('  ✗ 그림 ' + drift.length + '장의 base.css 버전이 낡았습니다: '
      + drift.slice(0, 5).join(' ') + (drift.length > 5 ? ' …' : ''));
    console.error('    node figures/manifest.js 로 다시 찍으세요.');
    process.exit(1);
  }
  if (cur !== next) {
    console.error('  ✗ figures/manifest.json 이 그림 내용과 어긋났습니다 (해시 또는 MAP).');
    console.error('    node figures/manifest.js 로 다시 만드세요.');
    process.exit(1);
  }
  console.log('  · 매니페스트 최신 (' + count + '개, ver ' + json.ver + ')');
} else {
  fs.writeFileSync(OUT, next);
  console.log('  → figures/manifest.json (' + count + '개, ver ' + json.ver
    + ', css ' + json.css + ')');
}
