/* DeepSeek-v3-Lite portal runtime: boot overlay, hero decode, ASCII
   widgets. Vanilla JS, no dependencies, file://-safe. */
(function () {
    'use strict';

    var REDUCED = window.matchMedia &&
        window.matchMedia('(prefers-reduced-motion: reduce)').matches;

    /* ---- boot overlay ------------------------------------------------ */

    function finishBoot() {
        document.documentElement.classList.remove('booting');
        var ov = document.getElementById('boot-overlay');
        if (ov && ov.parentNode) ov.parentNode.removeChild(ov);
    }

    function runBoot() {
        var ov = document.getElementById('boot-overlay');
        if (!ov || REDUCED) { finishBoot(); return; }
        var bar = ov.querySelector('.boot-bar');
        var t0 = null, FILL = 500, done = false;
        function skip() { if (!done) { done = true; finishBoot(); } }
        document.addEventListener('click', skip, { once: true });
        document.addEventListener('keydown', skip, { once: true });
        function frame(t) {
            if (done) return;
            if (t0 === null) t0 = t;
            var p = Math.min(1, (t - t0) / FILL);
            var e = 1 - Math.pow(1 - p, 3);
            var n = Math.round(e * 12);
            if (bar) {
                bar.textContent = '[' + '\u2593'.repeat(n) +
                    '\u2591'.repeat(12 - n) + '] ' + Math.round(e * 100) + '%';
            }
            if (p >= 1) {
                done = true;
                ov.classList.add('boot-fading');
                setTimeout(finishBoot, 220);
                return;
            }
            requestAnimationFrame(frame);
        }
        requestAnimationFrame(frame);
    }

    /* ---- hero scramble-decode ----------------------------------------- */

    var GLYPHS = '\u2591\u2592\u2593/\\|=+*\u00b7#<>';

    function glyph() { return GLYPHS[Math.floor(Math.random() * GLYPHS.length)]; }

    function runHeroDecode() {
        var pre = document.getElementById('hero-decode');
        if (!pre) return;
        var title = pre.getAttribute('data-title') || 'DEEPSEEK-V3-LITE';
        var sub = pre.getAttribute('data-sub') || '';
        if (REDUCED) { pre.textContent = '  ' + title + '\n\n  ' + sub; return; }
        var t0 = null, LOCK = 1600, TYPE_SPEED = 28;
        function frame(t) {
            if (t0 === null) t0 = t;
            var el = t - t0;
            var locked = Math.min(title.length, Math.floor(el / LOCK * title.length));
            var line = '';
            for (var i = 0; i < title.length; i++) line += i < locked ? title[i] : glyph();
            if (locked >= title.length) {
                var k = Math.min(sub.length, Math.floor((el - LOCK) / TYPE_SPEED));
                pre.textContent = '  ' + line + '\n\n  ' + sub.slice(0, k) +
                    (k < sub.length ? '\u2588' : '');
                if (k >= sub.length) return;
            } else {
                pre.textContent = '  ' + line;
            }
            requestAnimationFrame(frame);
        }
        requestAnimationFrame(frame);
    }

    /* ---- expandable code blocks ---------------------------------------- */

    var CODE_MAX = 16, CODE_SHOWN = 12;

    function initExpandableCode() {
        document.querySelectorAll('.markdown-body .code-wrapper').forEach(function (wrap) {
            var pre = wrap.querySelector('pre');
            if (!pre) return;
            var lines = pre.textContent.split('\n');
            if (lines.length <= CODE_MAX) return;
            var hidden = lines.length - CODE_SHOWN;
            wrap.classList.add('collapsible', 'collapsed');
            var btn = document.createElement('button');
            btn.type = 'button';
            btn.className = 'code-seam';
            btn.setAttribute('aria-expanded', 'false');
            btn.textContent = '\u2500\u2500 show ' + hidden + ' more lines \u25be';
            btn.addEventListener('click', function () {
                var open = wrap.classList.toggle('collapsed') === false;
                btn.setAttribute('aria-expanded', String(open));
                btn.textContent = open ? '\u2500\u2500 collapse \u25b4'
                    : ('\u2500\u2500 show ' + hidden + ' more lines \u25be');
            });
            wrap.appendChild(btn);
        });
    }

    /* ---- 18-layer stack explorer ---------------------------------------- */

    function layerSpec(i) {
        var dense = i < 2;
        return [
            'layer      ' + String(i).padStart(2, '0') + ' / 17',
            'attention  MLA \u00b7 kv_lora_rank 192 \u00b7 rope_head 24',
            dense ? 'ffn        SwiGLU \u00b7 inter_dim 1536'
                  : 'ffn        DeepSeekMoE \u00b7 20 routed (top-4) + 1 shared \u00b7 inter 384',
            dense ? 'role       dense warmup layer' : 'role       routed MoE layer',
            'source     configs/pretrain_a100_422m.yaml'
        ].join('\n');
    }

    function stackRowText(i) {
        var dense = i < 2;
        var inner = (' ' + String(i).padStart(2, '0') + ' \u2502 MLA \u25b8 ' +
            (dense ? 'SwiGLU \u00b7 dense' : 'MoE 20+1 \u00b7 top-4')).padEnd(36);
        return '\u2502' + inner + '\u2502';
    }

    function initLayerExplorer() {
        var box = document.getElementById('widget-layer-stack');
        if (!box) return;
        var pre = box.querySelector('pre.ascii-stack');
        var panel = box.querySelector('.ascii-panel');
        if (!pre) return;
        /* Rebuild the static fallback art as focusable rows. */
        pre.textContent = '';
        pre.appendChild(document.createTextNode(
            '\u250c' + '\u2500'.repeat(36) + '\u2510\n'));
        for (var i = 17; i >= 0; i--) {
            var row = document.createElement('span');
            row.className = 'stack-row';
            row.tabIndex = 0;
            row.setAttribute('data-layer', String(i));
            row.textContent = stackRowText(i);
            pre.appendChild(row);
            pre.appendChild(document.createTextNode('\n'));
        }
        pre.appendChild(document.createTextNode(
            '\u2514' + '\u2500'.repeat(36) + '\u2518'));
        if (!panel) return;
        function select(ev) {
            var idx = parseInt(ev.currentTarget.getAttribute('data-layer'), 10);
            panel.textContent = layerSpec(idx);
            pre.querySelectorAll('.stack-row.active').forEach(function (r) {
                r.classList.remove('active');
            });
            ev.currentTarget.classList.add('active');
        }
        pre.querySelectorAll('.stack-row').forEach(function (row) {
            row.addEventListener('mouseenter', select);
            row.addEventListener('focus', select);
            row.addEventListener('click', select);
        });
    }

    /* ---- MoE routing playground ------------------------------------------ */

    function initMoePlayground() {
        var box = document.getElementById('widget-moe-routing');
        if (!box) return;
        var grid = box.querySelector('pre.moe-grid');
        if (!grid) return;
        var status = document.createElement('div');
        status.className = 'moe-status';
        status.textContent = 'shared + 4 routed active';
        var btn = document.createElement('button');
        btn.type = 'button';
        btn.className = 'moe-fire';
        btn.textContent = '\u25b8 route token';
        box.appendChild(status);
        box.appendChild(btn);
        function render(active, weights) {
            var s = 'gate(x) \u25b8 top-4 of 20 routed\n';
            for (var r = 0; r < 4; r++) {
                for (var c = 0; c < 5; c++) {
                    var idx = r * 5 + c;
                    s += (active.indexOf(idx) >= 0 ? '\u2593\u2593' : '\u2591\u2591') +
                        String(idx).padStart(2, '0') + ' ';
                }
                s += '\n';
            }
            s += 'shared  \u2593\u2593sh (always on)';
            grid.textContent = s;
            if (weights) {
                var parts = active.map(function (e, k) {
                    return 'e' + String(e).padStart(2, '0') + '=' + weights[k].toFixed(2);
                });
                status.textContent = 'shared + 4 routed active \u00b7 ' + parts.join(' ');
            }
        }
        function route() {
            var act = [];
            while (act.length < 4) {
                var e = Math.floor(Math.random() * 20);
                if (act.indexOf(e) < 0) act.push(e);
            }
            var logits = act.map(function () { return Math.random() * 2; });
            var m = Math.max.apply(null, logits);
            var ex = logits.map(function (v) { return Math.exp(v - m); });
            var sum = ex.reduce(function (a, b) { return a + b; }, 0);
            render(act, ex.map(function (v) { return v / sum; }));
        }
        btn.addEventListener('click', route);
        route();
    }

    /* ---- MLA absorption toggle --------------------------------------------- */

    var MLA_STANDARD = [
        'standard attention (materialised KV)',
        '',
        ' h[768] \u2500 W_q \u2500\u25b8 q [12 \u00d7 72]  \u2510',
        ' h[768] \u2500 W_k \u2500\u25b8 k [12 \u00d7 48]  \u251c\u2500\u25b8 attn(q, k, v) \u25b8 out',
        ' h[768] \u2500 W_v \u2500\u25b8 v [12 \u00d7 64]  \u2518',
        '',
        ' KV cache / token : 12 \u00d7 (48 + 64) = 1,344 dims'
    ].join('\n');

    var MLA_ABSORBED = [
        'absorbed attention (latent-compressed)',
        '',
        ' h[768] \u2500 W_D \u2500\u25b8 c [192] (+ rope 24) \u2500\u25b8 cached',
        ' c \u2500\u25b8 W_U \u00b7 W_q absorbed into q\u2032 \u2500\u25b8 attn \u25b8 out',
        '',
        ' KV cache / token : 192 + 24 = 216 dims',
        ' compression      : 1,344 \u25b8 216 (\u2248 6.2\u00d7)'
    ].join('\n');

    function initMlaToggle() {
        var box = document.getElementById('widget-mla-absorb');
        if (!box) return;
        var pre = box.querySelector('pre.mla-figure');
        if (!pre) return;
        var bar = document.createElement('div');
        bar.className = 'mla-toggle-bar';
        var bStd = document.createElement('button');
        bStd.type = 'button';
        bStd.textContent = '[ standard ]';
        var bAbs = document.createElement('button');
        bAbs.type = 'button';
        bAbs.textContent = '[ absorbed ]';
        bar.appendChild(bStd);
        bar.appendChild(bAbs);
        box.insertBefore(bar, pre);
        function show(absorbed) {
            pre.textContent = absorbed ? MLA_ABSORBED : MLA_STANDARD;
            bStd.setAttribute('aria-pressed', String(!absorbed));
            bAbs.setAttribute('aria-pressed', String(absorbed));
        }
        bStd.addEventListener('click', function () { show(false); });
        bAbs.addEventListener('click', function () { show(true); });
        show(true);
    }

    /* ---- reading progress strip --------------------------------------------- */

    function initProgressStrip() {
        if (!document.querySelector('.markdown-body')) return;
        var strip = document.createElement('div');
        strip.className = 'progress-strip';
        strip.setAttribute('aria-hidden', 'true');
        document.body.appendChild(strip);
        var heads = Array.prototype.slice.call(
            document.querySelectorAll('.markdown-body h2, .markdown-body h3'));
        var ticking = false;
        function update() {
            ticking = false;
            var doc = document.documentElement;
            var max = doc.scrollHeight - window.innerHeight;
            var p = max > 0 ? Math.min(1, Math.max(0, window.scrollY / max)) : 1;
            var n = Math.round(p * 10);
            var label = '';
            for (var i = 0; i < heads.length; i++) {
                if (heads[i].getBoundingClientRect().top <= 90) {
                    label = heads[i].textContent.replace(/#+\s*$/, '').trim();
                }
            }
            strip.textContent = '[' + '\u2593'.repeat(n) + '\u2591'.repeat(10 - n) +
                '] ' + Math.round(p * 100) + '%' + (label ? ' \u00b7 ' + label : '');
        }
        window.addEventListener('scroll', function () {
            if (!ticking) { ticking = true; requestAnimationFrame(update); }
        }, { passive: true });
        update();
    }

    /* ---- init ---------------------------------------------------------------- */

    function init() {
        runBoot();
        runHeroDecode();
        initExpandableCode();
        initLayerExplorer();
        initMoePlayground();
        initMlaToggle();
        initProgressStrip();
    }

    if (document.readyState === 'loading') {
        document.addEventListener('DOMContentLoaded', init);
    } else {
        init();
    }
})();
