/* ============================================================================
   발표 도구 — 레이저 포인터 · 마우스 필기 · 라이트 테마
   reveal.js 기본에는 셋 다 없습니다. 서드파티 플러그인(chalkboard 등)을 쓰면
   CDN 의존이 생기고 톤도 덱과 어긋나므로 여기서 직접 만듭니다. 의존성 없음.

     L   레이저 포인터        D  필기(펜)        E  지우개
     C   펜 색 순환           Z  한 획 되돌리기  Shift+Z  다시
     X   이 슬라이드 필기 지우기                 T  라이트/다크 테마

   설계상 지켜야 할 것 세 가지
   1. 획은 **슬라이드 좌표(1376×774)로 저장**합니다. 화면 크기가 바뀌거나 테마를
      바꿔도 필기가 그림 위 같은 자리에 그대로 남습니다.
   2. 펜 색은 hex가 아니라 **의미 이름**('cyan'·'amber'·'violet'·'ink')으로 저장하고
      그릴 때 현재 테마의 CSS 변수로 해석합니다. 이 문서는 색이 곧 의미라,
      테마를 바꿔도 "시안으로 표시한 것"이 계속 시안이어야 합니다.
   3. 터치 기기에서는 도구를 켜지 않습니다 — 탭·스와이프가 이미 몰입 모드를
      쓰고 있어 충돌합니다. 테마만 따라갑니다.
   ========================================================================== */
(function () {
  'use strict';

  var root = document.documentElement;
  var TOUCH = window.matchMedia('(pointer: coarse)').matches;

  /* ── 그림과 테마 ──────────────────────────────────────────────────────────
     발표 자료라 기본은 다크입니다. 고른 값만 기억해 다음에 열 때 이어집니다.

     그림은 캡처된 webp가 아니라 figures/*.html 을 그 자리에서 렌더합니다. 그래서
     테마를 바꿀 때 이미지를 맞바꾸지 않고 iframe 안 문서에 같은 data-theme를 심습니다.
     확대해도 벡터라 선명하고, 그림 두 벌(다크·라이트)을 들고 다니지 않아도 됩니다.

     **후퇴 경로**: index.html에는 <img>가 그대로 들어 있습니다. 매니페스트를 못 읽거나
     iframe이 안 뜨면 그 이미지가 남습니다. 발표 중에 그림이 사라지는 것이 최악이므로
     LIVE_FIGURES = false 로 두면 예전처럼 이미지만 씁니다. */
  var LIVE_FIGURES = true;
  var THEME_KEY = 'cr-deck-theme';
  var figs = [];      /* <img> 들 (후퇴용) */
  var frames = [];    /* <iframe> 들 (실시간 렌더) */
  var figVer = {};    /* 그림 소스 → 내용 해시 */

  /* Pages가 max-age=600으로 내보내므로, 고친 그림을 배포해도 이미 열어 본 브라우저는
     하드 리프레시 없이 10분간 옛 그림을 그립니다. 그림 URL에 내용 해시를 붙여
     그 구간을 없앱니다(해시는 no-store로 읽는 manifest.json에서 옵니다).
     주소에 ?fresh를 붙이면 해시 대신 현재 시각을 씁니다. */
  var FRESH = /(^|[?&])fresh\b/.test(location.search);

  function figUrl(file) {
    var t = FRESH ? Date.now() : figVer[file];
    return '../figures/' + file + (t ? '?v=' + t : '');
  }

  function initFigures() {
    var imgs = document.querySelectorAll('.reveal .slides section.fig img');
    for (var i = 0; i < imgs.length; i++) {
      // src에는 캐시 무효화용 ?v=…가 붙어 있습니다. 경로와 쿼리를 갈라
      // 파일명만 _light로 바꾸고 쿼리는 그대로 이어 붙입니다. 쿼리를 고려하지 않고
      // /\.webp$/로 검사하면 그림이 전부 걸러져 테마 전환이 조용히 죽습니다.
      var img = imgs[i], src = img.getAttribute('src');
      var m = /^([^?#]*\.webp)(\?[^#]*)?$/.exec(src);
      if (!m) continue;
      img.dataset.dark = src;
      img.dataset.light = m[1].replace(/\.webp$/, '_light.webp') + (m[2] || '');
      // 라이트 그림이 없으면 조용히 다크로 되돌립니다 (덱이 깨지지 않게)
      img.addEventListener('error', function () {
        if (this.dataset.dark && this.getAttribute('src') !== this.dataset.dark) {
          this.setAttribute('src', this.dataset.dark);
        }
      });
      figs.push(img);
    }
  }

  /* iframe 안 문서에 테마를 심습니다. 같은 출처라 접근됩니다. */
  function paintFrame(frame, name) {
    try {
      var d = frame.contentDocument;
      if (!d || !d.documentElement) return false;
      if (name === 'light') d.documentElement.setAttribute('data-theme', 'light');
      else d.documentElement.removeAttribute('data-theme');
      return true;
    } catch (e) { return false; }
  }

  /* <img>를 같은 그림의 <iframe>으로 갈아 끼웁니다. 실패하면 <img>를 되살립니다. */
  function liveFigure(img, file, name) {
    var frame = document.createElement('iframe');
    frame.setAttribute('scrolling', 'no');
    frame.setAttribute('tabindex', '-1');
    frame.setAttribute('aria-hidden', 'true');   // 슬라이드에 alt가 이미 있습니다
    frame.setAttribute('title', img.getAttribute('alt') || '');
    frame.dataset.fallback = img.getAttribute('src') || '';

    var settled = false;
    var giveUp = setTimeout(function () { if (!settled) revert(frame, img); }, 8000);

    frame.addEventListener('load', function () {
      var ok = paintFrame(frame, root.getAttribute('data-theme') || 'dark');
      var body = null;
      try { body = frame.contentDocument && frame.contentDocument.body; } catch (e) {}
      if (!ok || !body || body.children.length === 0) { clearTimeout(giveUp); revert(frame, img); return; }
      settled = true;
      clearTimeout(giveUp);
    });
    frame.addEventListener('error', function () { clearTimeout(giveUp); revert(frame, img); });

    frame.src = figUrl(file);
    img.parentNode.replaceChild(frame, img);
    frames.push(frame);
    return frame;
  }

  function revert(frame, img) {
    var i = frames.indexOf(frame);
    if (i >= 0) frames.splice(i, 1);
    if (frame.parentNode) frame.parentNode.replaceChild(img, frame);
    figs.push(img);
    var name = root.getAttribute('data-theme') || 'dark';
    var want = name === 'light' ? img.dataset.light : img.dataset.dark;
    if (want) img.setAttribute('src', want);
  }

  function applyTheme(name) {
    root.setAttribute('data-theme', name);
    for (var i = 0; i < figs.length; i++) {
      var want = name === 'light' ? figs[i].dataset.light : figs[i].dataset.dark;
      if (want && figs[i].getAttribute('src') !== want) figs[i].setAttribute('src', want);
    }
    for (var j = 0; j < frames.length; j++) paintFrame(frames[j], name);
    var b = document.getElementById('tb-theme');
    if (b) b.setAttribute('aria-pressed', name === 'light' ? 'true' : 'false');
    draw();   // 펜 색은 테마의 CSS 변수를 따르므로 다시 그립니다
  }

  function toggleTheme() {
    var next = root.getAttribute('data-theme') === 'light' ? 'dark' : 'light';
    applyTheme(next);
    try { localStorage.setItem(THEME_KEY, next); } catch (e) {}
    flash(next === 'light' ? '라이트 테마' : '다크 테마');
  }

  initFigures();
  var stored = null;
  try { stored = localStorage.getItem(THEME_KEY); } catch (e) {}
  applyTheme(stored === 'light' ? 'light' : 'dark');

  /* 그림 소스로 갈아 끼웁니다. 매니페스트를 못 읽으면 이미지 그대로 갑니다. */
  if (LIVE_FIGURES) {
    fetch('../figures/manifest.json?t=' + Date.now(), { cache: 'no-store' })
      .then(function (r) { return r.ok ? r.json() : null; })
      .then(function (data) {
        var map = (data && data.figures) || null;
        figVer = (data && data.v) || {};
        if (!map) return;
        var theme = root.getAttribute('data-theme') || 'dark';
        var pending = figs.slice();
        figs.length = 0;
        for (var i = 0; i < pending.length; i++) {
          var img = pending[i];
          var m = /assets\/([a-z0-9_]+)\.webp/i.exec(img.dataset.dark || '');
          var file = m && map[m[1]];
          if (file) liveFigure(img, file, theme);
          else figs.push(img);      /* 매핑이 없으면 이미지로 남깁니다 */
        }
      })
      .catch(function () { /* 이미지로 갑니다 */ });
  }

  /* ── 오버레이 및 키 바인딩 (모바일 포함 활성화) ───────────────────────── */
  var layer = document.createElement('div');
  layer.className = 'ptools';
  layer.innerHTML =
    '<canvas class="ink"></canvas>' +
    '<div class="laser"><i class="lg"></i><i class="lc"></i></div>' +
    '<div class="penring"></div>' +
    '<div class="toast"></div>';
  document.body.appendChild(layer);

  var canvas = layer.querySelector('.ink');
  var ctx = canvas.getContext('2d');
  var laser = layer.querySelector('.laser');
  var lglow = layer.querySelector('.lg');
  var ring = layer.querySelector('.penring');
  var toast = layer.querySelector('.toast');

  /* ── 상태 ────────────────────────────────────────────────────────────── */
  var W = 1376, H = 774;                 // 슬라이드 좌표계
  var mode = null;                       // null | 'laser' | 'pen' | 'eraser'
  var COLORS = ['cyan', 'amber', 'violet', 'ink'];
  var COLOR_KO = { cyan: '시안', amber: '앰버', violet: '보라', ink: '먹' };
  var color = 'cyan';
  var LINE_W = 3.4;                      // 슬라이드 좌표 기준 굵기
  var ERASE_R = 16;

  var strokes = new Map();               // section → [stroke]
  var redo = new Map();                  // section → [stroke]
  var live = null;                       // 그리는 중인 획
  var down = false;
  var px = -99, py = -99;                // 마지막 포인터 (화면 좌표)

  function slide() { return Reveal.getCurrentSlide(); }
  function listFor(m, s) { if (!m.has(s)) m.set(s, []); return m.get(s); }

  /* 화면 좌표 → 슬라이드 좌표.
     reveal이 .slides에 transform: scale()을 걸어 두므로 현재 섹션의 실제 사각형과
     Reveal.getScale()로 환산해야 합니다. 이 환산을 빼먹으면 창 크기가 바뀔 때마다
     필기가 어긋납니다(덱 검증 스크립트가 같은 함정에 빠진 전례가 있습니다). */
  function geom() {
    var s = slide();
    if (!s) return null;
    var r = s.getBoundingClientRect();
    var k = Reveal.getScale() || (r.width / W) || 1;
    return { left: r.left, top: r.top, k: k };
  }
  function toSlide(cx, cy) {
    var g = geom();
    if (!g) return null;
    return [(cx - g.left) / g.k, (cy - g.top) / g.k];
  }

  function cssColor(name) {
    var v = getComputedStyle(root).getPropertyValue('--pen-' + name);
    return (v || '').trim() || '#22a7bd';
  }

  /* ── 캔버스 ──────────────────────────────────────────────────────────── */
  function sizeCanvas() {
    var dpr = Math.min(window.devicePixelRatio || 1, 2.5);
    canvas.width = Math.round(window.innerWidth * dpr);
    canvas.height = Math.round(window.innerHeight * dpr);
    canvas.style.width = window.innerWidth + 'px';
    canvas.style.height = window.innerHeight + 'px';
    canvas._dpr = dpr;
    draw();
  }

  function draw() {
    if (!ctx) return;
    ctx.setTransform(1, 0, 0, 1, 0, 0);
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    var g = geom();
    if (!g) return;
    var d = canvas._dpr || 1;
    ctx.setTransform(g.k * d, 0, 0, g.k * d, g.left * d, g.top * d);
    ctx.lineCap = 'round';
    ctx.lineJoin = 'round';

    var list = strokes.get(slide()) || [];
    for (var i = 0; i < list.length; i++) path(list[i]);
    if (live) path(live);
  }

  /* 점을 그대로 이으면 각이 지므로 중점을 통과하는 이차 곡선으로 부드럽게 만듭니다 */
  function path(st) {
    var p = st.pts;
    if (!p.length) return;
    ctx.strokeStyle = cssColor(st.color);
    ctx.lineWidth = st.w;
    ctx.globalAlpha = st.color === 'ink' ? 1 : .92;
    ctx.beginPath();
    if (p.length < 3) {
      ctx.moveTo(p[0][0], p[0][1]);
      ctx.lineTo(p[p.length - 1][0], p[p.length - 1][1]);
    } else {
      ctx.moveTo(p[0][0], p[0][1]);
      for (var i = 1; i < p.length - 1; i++) {
        ctx.quadraticCurveTo(p[i][0], p[i][1],
          (p[i][0] + p[i + 1][0]) / 2, (p[i][1] + p[i + 1][1]) / 2);
      }
      ctx.lineTo(p[p.length - 1][0], p[p.length - 1][1]);
    }
    ctx.stroke();
    ctx.globalAlpha = 1;
  }

  /* ── 지우개 — 픽셀이 아니라 획 단위로 지웁니다 (실행 취소와 결이 맞습니다) ── */
  function eraseAt(x, y) {
    var list = strokes.get(slide());
    if (!list || !list.length) return;
    var kept = [], hit = false;
    for (var i = 0; i < list.length; i++) {
      if (nearStroke(list[i], x, y)) { hit = true; } else { kept.push(list[i]); }
    }
    if (hit) { strokes.set(slide(), kept); draw(); }
  }
  function nearStroke(st, x, y) {
    var p = st.pts, r = ERASE_R + st.w / 2, r2 = r * r;
    for (var i = 0; i < p.length; i++) {
      if (i) { if (segDist2(p[i - 1], p[i], x, y) <= r2) return true; }
      else { var dx = p[0][0] - x, dy = p[0][1] - y; if (dx * dx + dy * dy <= r2) return true; }
    }
    return false;
  }
  function segDist2(a, b, x, y) {
    var vx = b[0] - a[0], vy = b[1] - a[1];
    var wx = x - a[0], wy = y - a[1];
    var L2 = vx * vx + vy * vy;
    var t = L2 ? Math.max(0, Math.min(1, (wx * vx + wy * vy) / L2)) : 0;
    var dx = a[0] + t * vx - x, dy = a[1] + t * vy - y;
    return dx * dx + dy * dy;
  }

  /* ── 모드 ────────────────────────────────────────────────────────────── */
  function setMode(m) {
    mode = (mode === m) ? null : m;
    root.classList.toggle('t-laser', mode === 'laser');
    root.classList.toggle('t-pen', mode === 'pen');
    root.classList.toggle('t-eraser', mode === 'eraser');
    root.classList.toggle('t-on', !!mode);
    layer.classList.toggle('catch', mode === 'pen' || mode === 'eraser');
    // 필기 중에는 화살표로 슬라이드가 넘어가면 곤란하지 않으므로 키보드는 그대로 둡니다
    syncCursor();
    updateBar();
    flash(mode === 'laser' ? '레이저 포인터'
        : mode === 'pen' ? '필기 · ' + COLOR_KO[color]
        : mode === 'eraser' ? '지우개' : '도구 끄기');
  }

  function syncCursor() {
    ring.style.borderColor = cssColor(color);
    ring.style.width = ring.style.height =
      (mode === 'eraser' ? ERASE_R * 2 : Math.max(14, LINE_W * 4)) + 'px';
    place();
  }

  function cycleColor() {
    color = COLORS[(COLORS.indexOf(color) + 1) % COLORS.length];
    if (mode !== 'pen') setMode('pen'); else { syncCursor(); updateBar(); flash('필기 · ' + COLOR_KO[color]); }
  }

  function undo() {
    var list = strokes.get(slide());
    if (!list || !list.length) return;
    listFor(redo, slide()).push(list.pop());
    draw();
  }
  function redoLast() {
    var r = redo.get(slide());
    if (!r || !r.length) return;
    listFor(strokes, slide()).push(r.pop());
    draw();
  }
  function clearSlide() {
    var list = strokes.get(slide());
    if (!list || !list.length) return;
    redo.set(slide(), list.slice().reverse().concat(redo.get(slide()) || []));
    strokes.set(slide(), []);
    draw();
    flash('이 슬라이드 필기 지움');
  }

  /* ── 포인터 ──────────────────────────────────────────────────────────── */
  var raf = 0;
  function place() {
    laser.style.transform = 'translate(' + px + 'px,' + py + 'px)';
    lglow.style.transform = 'translate(-50%,-50%)';
    ring.style.transform = 'translate(' + px + 'px,' + py + 'px) translate(-50%,-50%)';
  }

  window.addEventListener('pointermove', function (e) {
    px = e.clientX; py = e.clientY;
    if (!raf) raf = requestAnimationFrame(function () { raf = 0; place(); });
    barProximity(e.clientY);

    if (!down) return;
    if (mode === 'pen' && live) {
      var p = toSlide(e.clientX, e.clientY);
      if (!p) return;
      var last = live.pts[live.pts.length - 1];
      // 너무 촘촘한 점은 버립니다 — 곡선 품질은 그대로면서 획이 가벼워집니다
      if (!last || Math.abs(p[0] - last[0]) + Math.abs(p[1] - last[1]) > 1.2) {
        live.pts.push(p); draw();
      }
    } else if (mode === 'eraser') {
      var q = toSlide(e.clientX, e.clientY);
      if (q) eraseAt(q[0], q[1]);
    }
  }, { passive: true });

  layer.addEventListener('pointerdown', function (e) {
    if (TOUCH || e.pointerType === 'touch') return;
    if (mode !== 'pen' && mode !== 'eraser') return;
    e.preventDefault();
    layer.setPointerCapture(e.pointerId);
    down = true;
    var p = toSlide(e.clientX, e.clientY);
    if (!p) return;
    if (mode === 'pen') {
      live = { color: color, w: LINE_W, pts: [p] };
      redo.set(slide(), []);
    } else {
      eraseAt(p[0], p[1]);
    }
  });

  function endStroke() {
    if (!down) return;
    down = false;
    if (live) {
      if (live.pts.length > 1) listFor(strokes, slide()).push(live);
      live = null;
      draw();
    }
  }
  layer.addEventListener('pointerup', endStroke);
  layer.addEventListener('pointercancel', endStroke);
  window.addEventListener('blur', endStroke);

  /* ── 하단 도구막대 — 포인터가 아래쪽에 가면 나타납니다 ────────────────
     타이머로 여닫으면 발표 중에 깜빡여서 시선을 뺏깁니다. "아래로 내리면 보인다"는
     규칙이 예측 가능하고, 슬라이드를 가리는 시간도 최소입니다. */
  var bar = document.createElement('div');
  bar.className = 'pbar';
  bar.innerHTML =
    '<button id="tb-laser" data-k="L">레이저</button>' +
    '<button id="tb-pen"   data-k="D">필기</button>' +
    '<button id="tb-color" data-k="C"><i class="sw"></i>색</button>' +
    '<button id="tb-erase" data-k="E">지우개</button>' +
    '<button id="tb-clear" data-k="X">지움</button>' +
    '<span class="sep"></span>' +
    '<button id="tb-theme" data-k="T">테마</button>';
  layer.appendChild(bar);

  bar.addEventListener('pointerdown', function (e) { e.stopPropagation(); });
  bar.addEventListener('click', function (e) {
    var b = e.target.closest('button');
    if (!b) return;
    ({ 'tb-laser': function () { setMode('laser'); },
       'tb-pen':   function () { setMode('pen'); },
       'tb-color': cycleColor,
       'tb-erase': function () { setMode('eraser'); },
       'tb-clear': clearSlide,
       'tb-theme': toggleTheme })[b.id]();
    b.blur();
  });

  function barProximity(y) {
    bar.classList.toggle('near', y > window.innerHeight - 130);
  }
  function updateBar() {
    bar.querySelector('#tb-laser').setAttribute('aria-pressed', mode === 'laser');
    bar.querySelector('#tb-pen').setAttribute('aria-pressed', mode === 'pen');
    bar.querySelector('#tb-erase').setAttribute('aria-pressed', mode === 'eraser');
    bar.querySelector('#tb-color .sw').style.background = cssColor(color);
  }
  updateBar();
  bar.classList.add('near');                                  // 처음 3초만 보여 줍니다
  setTimeout(function () { bar.classList.remove('near'); }, 3000);

  /* ── 알림 칩 ─────────────────────────────────────────────────────────── */
  var toastT = 0;
  function flash(text) {
    toast.textContent = text;
    toast.classList.add('on');
    clearTimeout(toastT);
    toastT = setTimeout(function () { toast.classList.remove('on'); }, 1100);
  }

  /* ── 키 ──────────────────────────────────────────────────────────────
     Reveal.addKeyBinding으로 등록하면 `?` 도움말에도 함께 실립니다.
     L은 reveal 기본이 "오른쪽 이동"인데 →·SPACE·N과 겹치는 여분이라 가져옵니다.
     P(이전 슬라이드)는 건드리지 않으려고 필기는 D(Draw)로 뒀습니다. */
  function bind(code, key, desc, fn) {
    Reveal.addKeyBinding({ keyCode: code, key: key, description: desc }, fn);
  }
  bind(76, 'L', '레이저 포인터', function () { setMode('laser'); });
  bind(68, 'D', '필기(펜)',      function () { setMode('pen'); });
  bind(69, 'E', '지우개',        function () { setMode('eraser'); });
  bind(67, 'C', '펜 색 순환',    cycleColor);
  bind(88, 'X', '이 슬라이드 필기 지우기', clearSlide);
  bind(84, 'T', '라이트/다크 테마',       toggleTheme);
  bind(90, 'Z', '되돌리기 (Shift+Z 다시)', undo);

  /* ── 숫자로 슬라이드 이동 ──────────────────────────────────────────────
     reveal에 jump-to-slide가 들어 있지만 G를 눌러야 열립니다. 발표 중에
     "32번으로 가 주세요" 같은 요청이 오면 G를 기억하고 있어야 해서, 숫자를
     누르는 것만으로 바로 열리게 했습니다 — 3 · 2 · Enter면 32번입니다.
     ESC는 취소(원래 슬라이드로 복귀)입니다.

     캡처 단계에서 잡아 reveal 기본 처리보다 먼저 가로챕니다. 입력창이
     열린 뒤의 타이핑은 reveal이 처리하므로 여기서는 첫 숫자만 넣어 줍니다.
     정확히 N번째로 가려면 Reveal.initialize의 slideNumber가 'c'나 'c/t'여야
     합니다(index.html 참고) — false면 숫자를 다른 좌표로 해석합니다. */
  document.addEventListener('keydown', function (e) {
    if (e.metaKey || e.ctrlKey || e.altKey || e.shiftKey) return;
    var t = e.target;
    if (t && (/input|textarea/i.test(t.tagName || '') || t.isContentEditable)) return;
    if (!/^[0-9]$/.test(e.key)) return;
    if (typeof Reveal.toggleJumpToSlide !== 'function') return;
    e.preventDefault(); e.stopImmediatePropagation();
    Reveal.toggleJumpToSlide(true);
    var inp = document.querySelector('.jump-to-slide-input');
    // 값만 넣고 input 이벤트는 쏘지 않습니다 — 쏘면 400ms 뒤에 첫 숫자만으로
    // 한 번 이동해 버려서, 두 자리를 치는 동안 화면이 튑니다.
    if (inp) { inp.value = e.key; inp.placeholder = '슬라이드 번호'; }
  }, true);

  Reveal.addKeyBinding(
    { keyCode: 48, key: '0–9', description: '숫자로 슬라이드 이동 (Enter 확정 · ESC 취소)' },
    function () {}
  );

  document.addEventListener('keydown', function (e) {
    // 입력창(슬라이드 이동)에 타이핑하는 동안에는 도구 단축키가 끼어들지 않게 합니다
    var t = e.target;
    if (t && (/input|textarea/i.test(t.tagName || '') || t.isContentEditable)) return;
    // Shift+Z는 직접 잡습니다 — reveal이 수정자 붙은 키를 바인딩에 넘겨줄지가
    // 버전에 따라 다릅니다. 여기서 처리하고 Z 바인딩까지 흘러가지 않게 막습니다.
    if (e.keyCode === 90 && e.shiftKey && !e.ctrlKey && !e.metaKey && !e.altKey) {
      e.stopImmediatePropagation(); e.preventDefault(); redoLast(); return;
    }
    // ESC는 도구만 먼저 끕니다 — 도구를 켠 채로 개요 화면에 들어가면 헷갈립니다
    if (e.keyCode === 27 && mode) { e.stopImmediatePropagation(); e.preventDefault(); setMode(mode); }
  }, true);

  /* ── 갱신 ────────────────────────────────────────────────────────────── */
  Reveal.on('slidechanged', draw);
  Reveal.on('resize', function () { sizeCanvas(); });
  Reveal.on('overviewshown', function () { layer.classList.add('hide'); });
  Reveal.on('overviewhidden', function () { layer.classList.remove('hide'); draw(); });
  window.addEventListener('resize', sizeCanvas);
  sizeCanvas();
})();
