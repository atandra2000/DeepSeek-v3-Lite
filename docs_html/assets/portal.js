/* DeepSeek-v3-Lite Documentation Portal Client-Side Behaviors
   ======================================================================
   • DeepSeek Quantum Latent Manifold & Neural Synapse Visualizer (FIG · A0)
   • 3 Interactive Mechanism Exploder Labs (MLA VRAM, MoE 20+1, MTP Tree)
   • Full 18-Layer Training Step Pipeline Telemetry (FIG · A1)
   • 18-Layer ASCII Stack Interactive Inspector (FIG · 01)
   • Dedicated Concept Widgets (MoE Playground, MLA Toggle, DualPipe, MTP)
   • Expandable Code Blocks (>14 lines) & Copy-to-Clipboard
   • Navigation Filtering & TOC Scrollspy
*/

(function () {
    'use strict';

    var reduced = window.matchMedia && window.matchMedia('(prefers-reduced-motion: reduce)').matches;

    // ------------------------------------------------------------------
    // 1. DeepSeek Quantum Latent Manifold & Neural Synapses (FIG · A0)
    //    Interactive high-DPI polar geometry simulation:
    //      - 192d Compressed Latent Key/Value Horizon (7.1× KV Cache Cut)
    //      - 24d Decoupled RoPE Rotary Orbit (θ=10K)
    //      - Matrix Absorption Projections (W_U · W_q → q') across 12 Heads
    //      - DeepSeekMoE 20+1 Dynamic Biased Routing Field (Aux-Loss-Free)
    //      - Multi-Token Prediction (MTP Depth-1) Speculative Tree
    //      - DualPipe 1F1B Bidirectional Pipeline Overlap Stream
    // ------------------------------------------------------------------
    var heroController = {
        setMode: null,
        triggerAbsorptionFlash: null,
        triggerMoEPulse: null,
        triggerMTPPulse: null
    };

    function initMlaHero() {
        var canvas = document.getElementById('heroStateCanvas');
        if (!canvas) return;

        var ctx = canvas.getContext('2d');
        if (!ctx) return;

        var container = canvas.parentElement;
        var probeEl = document.getElementById('heroProbeHUD');
        var probeTag = document.getElementById('probeTag');
        var probeCoords = document.getElementById('probeCoords');
        var probeDecay = document.getElementById('probeDecay');
        var modeBtns = document.querySelectorAll('.fig-controls .fig-btn[data-mode]');
        var speedBtn = document.getElementById('heroSpeedBtn');
        var pauseBtn = document.getElementById('heroPauseBtn');
        var modeLabel = document.getElementById('hudModeLabel');
        var formulaBar = document.getElementById('heroFormulaBar');

        var currentMode = 'mla'; // 'mla' | 'moe' | 'mtp' | 'dualpipe'
        var simSpeed = 1.0;
        var isPaused = false;
        var isHovered = false;
        var mouseX = -9999, mouseY = -9999;
        var animId = null;
        var lastTime = 0;
        var simTime = 0;
        var flashEffect = 0;

        var PALETTE = {
            paperBg: '#0e0c0a',
            paperStation: '#161310',
            paperStationHover: '#1c1813',
            rule: '#2c261c',
            ruleStrong: '#3a3226',
            terracotta: '#e07a3f',
            terracottaGlow: 'rgba(224, 122, 63, 0.75)',
            terracottaTint: 'rgba(224, 122, 63, 0.16)',
            olive: '#9a9440',
            oliveGlow: 'rgba(154, 148, 64, 0.75)',
            oliveTint: 'rgba(154, 148, 64, 0.16)',
            gold: '#c9a35c',
            goldGlow: 'rgba(201, 163, 92, 0.85)',
            goldTint: 'rgba(201, 163, 92, 0.20)',
            ink: '#d8ccb4',
            inkSoft: '#b3a68c',
            inkFaint: '#7a7160'
        };

        var shockwaves = [];
        var particles = [];
        var numParticles = 54;
        for (var p = 0; p < numParticles; p++) {
            particles.push({
                radiusFrac: 0.15 + 0.80 * Math.random(),
                theta: Math.random() * Math.PI * 2,
                speed: (0.2 + 0.6 * Math.random()) * (Math.random() > 0.5 ? 1 : -1),
                size: 1.2 + 1.4 * Math.random(),
                alpha: 0.3 + 0.5 * Math.random(),
                lane: Math.floor(Math.random() * 4)
            });
        }

        var width = 0, height = 0, cx = 0, cy = 0, radius = 0;

        function resize() {
            var rect = container.getBoundingClientRect();
            var dpr = Math.min(window.devicePixelRatio || 1, 2);
            width = rect.width;
            height = rect.height || 360;
            canvas.width = Math.round(width * dpr);
            canvas.height = Math.round(height * dpr);
            ctx.setTransform(1, 0, 0, 1, 0, 0);
            ctx.scale(dpr, dpr);
            cx = width / 2;
            cy = height / 2;
            radius = Math.min(width, height) * 0.43;
        }

        if (window.ResizeObserver) {
            new ResizeObserver(function () {
                resize();
                if (reduced || isPaused) renderFrame(0, true);
            }).observe(container);
        } else {
            window.addEventListener('resize', resize);
        }
        resize();

        container.addEventListener('mousemove', function (e) {
            var rect = canvas.getBoundingClientRect();
            mouseX = e.clientX - rect.left;
            mouseY = e.clientY - rect.top;
            isHovered = true;
        });
        container.addEventListener('mouseleave', function () {
            isHovered = false;
            mouseX = -9999; mouseY = -9999;
            if (probeEl) probeEl.style.opacity = '0';
        });

        container.addEventListener('click', function (e) {
            var rect = canvas.getBoundingClientRect();
            var clickX = e.clientX - rect.left;
            var clickY = e.clientY - rect.top;
            shockwaves.push({
                x: clickX,
                y: clickY,
                radius: 4,
                maxRadius: radius * 0.9,
                life: 1.0,
                color: currentMode === 'moe' ? PALETTE.terracotta : (currentMode === 'mtp' ? PALETTE.olive : (currentMode === 'dualpipe' ? PALETTE.gold : PALETTE.terracotta))
            });
        });

        function setMode(mode) {
            currentMode = mode;
            modeBtns.forEach(function (b) {
                b.classList.toggle('active', b.getAttribute('data-mode') === mode);
            });
            updateFormulaAndLabels();
            if (reduced || isPaused) renderFrame(0, true);
        }

        heroController.setMode = setMode;
        heroController.triggerAbsorptionFlash = function () {
            setMode('mla');
            flashEffect = 1.0;
            shockwaves.push({ x: cx, y: cy, radius: 4, maxRadius: radius, life: 1.0, color: PALETTE.terracotta });
        };
        heroController.triggerMoEPulse = function () {
            setMode('moe');
            flashEffect = 1.0;
            shockwaves.push({ x: cx, y: cy, radius: 4, maxRadius: radius, life: 1.0, color: PALETTE.terracotta });
        };
        heroController.triggerMTPPulse = function () {
            setMode('mtp');
            flashEffect = 1.0;
            shockwaves.push({ x: cx, y: cy, radius: 4, maxRadius: radius, life: 1.0, color: PALETTE.olive });
        };

        modeBtns.forEach(function (btn) {
            btn.addEventListener('click', function (e) {
                e.stopPropagation();
                var m = btn.getAttribute('data-mode') || 'mla';
                setMode(m);
            });
        });

        if (speedBtn) {
            speedBtn.addEventListener('click', function (e) {
                e.stopPropagation();
                if (simSpeed === 1.0) simSpeed = 2.0;
                else if (simSpeed === 2.0) simSpeed = 0.5;
                else simSpeed = 1.0;
                speedBtn.textContent = simSpeed + '×';
                speedBtn.setAttribute('aria-label', 'Speed: ' + simSpeed + 'x');
            });
        }

        if (pauseBtn) {
            pauseBtn.addEventListener('click', function (e) {
                e.stopPropagation();
                isPaused = !isPaused;
                pauseBtn.innerHTML = isPaused ? '&#9658;' : '&#10074;&#10074;';
                pauseBtn.setAttribute('aria-label', isPaused ? 'Resume simulation' : 'Pause simulation');
                if (!isPaused && !animId) {
                    lastTime = performance.now();
                    loop(lastTime);
                }
            });
        }

        function updateFormulaAndLabels() {
            if (modeLabel) {
                if (currentMode === 'mla') modeLabel.textContent = 'MLA · LATENT PROJECTION (192d)';
                else if (currentMode === 'moe') modeLabel.textContent = 'DeepSeekMoE · 20+1 BIASED ROUTER';
                else if (currentMode === 'mtp') modeLabel.textContent = 'MTP · SPECULATIVE TREE (DEPTH 1)';
                else if (currentMode === 'dualpipe') modeLabel.textContent = 'DUALPIPE · 1F1B OVERLAP SCHEDULE';
            }
            if (formulaBar) {
                if (currentMode === 'mla') {
                    formulaBar.innerHTML = '<span class="formula-sym">q\'<sub class="f-sub">t,i</sub></span> <span class="formula-op">=</span> <span class="formula-term term-a">W<sub class="f-sub">q</sub> h<sub class="f-sub">t</sub> &middot; W<sub class="f-sub">U</sub></span> <span class="formula-dot">&middot;</span> <span class="formula-sym">score</span> <span class="formula-op">=</span> <span class="formula-term term-b">q\'<sub class="f-sub">t,i</sub> c<sub class="f-sub">j</sub><sup class="f-sub">KV&top;</sup> + q<sub class="f-sub">t,i</sub><sup class="f-sub">R</sup> k<sub class="f-sub">j</sub><sup class="f-sub">R&top;</sup></span>';
                } else if (currentMode === 'moe') {
                    formulaBar.innerHTML = '<span class="formula-sym">s<sub class="f-sub">i,t</sub></span> <span class="formula-op">=</span> <span class="formula-term term-a">&sigma;(W<sub class="f-sub">g</sub> x<sub class="f-sub">t</sub>)<sub class="f-sub">i</sub> + b<sub class="f-sub">i</sub></span> <span class="formula-dot">&middot;</span> <span class="formula-sym">Top-4</span> <span class="formula-op">&isin;</span> <span class="formula-term term-b">{0..19} + SharedExp(x)</span>';
                } else if (currentMode === 'mtp') {
                    formulaBar.innerHTML = '<span class="formula-sym">MTP<sub class="f-sub">depth=1</sub></span> <span class="formula-op">:</span> <span class="formula-term term-a">p(x<sub class="f-sub">t+1</sub>|x<sub class="f-sub">&le;t</sub>)</span> <span class="formula-op">&otimes;</span> <span class="formula-term term-b">p(x<sub class="f-sub">t+2</sub>|x<sub class="f-sub">&le;t+1</sub>)</span> <span class="formula-dot">&middot;</span> <span class="formula-sym">Speedup</span> <span class="formula-op">=</span> <span class="formula-term">1.85&times;</span>';
                } else {
                    formulaBar.innerHTML = '<span class="formula-sym">DualPipe</span> <span class="formula-op">:</span> <span class="formula-term term-a">1F1B Fwd Stream</span> <span class="formula-op">&harr;</span> <span class="formula-term term-b">1F1B Bwd Stream</span> <span class="formula-dot">&middot;</span> <span class="formula-sym">Comm Overlap</span> <span class="formula-op">=</span> <span class="formula-term">~100%</span>';
                }
            }
        }

        function drawDialAndGrid(t) {
            ctx.save();
            ctx.strokeStyle = 'rgba(58, 50, 38, 0.35)';
            ctx.lineWidth = 1;

            // Cardinal Crosshairs
            ctx.beginPath();
            ctx.moveTo(cx - radius - 16, cy); ctx.lineTo(cx + radius + 16, cy);
            ctx.moveTo(cx, cy - radius - 16); ctx.lineTo(cx, cy + radius + 16);
            ctx.stroke();

            // 64-tick perimeter dial
            var numTicks = 64;
            for (var k = 0; k < numTicks; k++) {
                var angle = (k / numTicks) * Math.PI * 2;
                var isMajor = (k % 8 === 0);
                var isMedium = (k % 4 === 0);
                var tickLen = isMajor ? 9 : (isMedium ? 6 : 3);
                var r0 = radius;
                var r1 = radius + tickLen;

                var x0 = cx + Math.cos(angle) * r0;
                var y0 = cy + Math.sin(angle) * r0;
                var x1 = cx + Math.cos(angle) * r1;
                var y1 = cy + Math.sin(angle) * r1;

                ctx.beginPath();
                ctx.moveTo(x0, y0);
                ctx.lineTo(x1, y1);
                ctx.strokeStyle = isMajor ? PALETTE.gold : (isMedium ? PALETTE.ruleStrong : 'rgba(58, 50, 38, 0.45)');
                ctx.lineWidth = isMajor ? 1.5 : 1;
                ctx.stroke();
            }

            // Concentric boundary circles
            var rings = [
                { r: radius * 0.38, style: 'rgba(224, 122, 63, 0.35)', dash: [4, 4], label: '192d LATENT KV' },
                { r: radius * 0.65, style: 'rgba(154, 148, 64, 0.35)', dash: [2, 6], label: '24d DECOUPLED RoPE' },
                { r: radius * 0.88, style: 'rgba(201, 163, 92, 0.28)', dash: [], label: '12-HEAD QUERY ORBIT' }
            ];

            rings.forEach(function (ring) {
                ctx.beginPath();
                ctx.arc(cx, cy, ring.r, 0, Math.PI * 2);
                ctx.strokeStyle = ring.style;
                ctx.lineWidth = 1;
                ctx.setLineDash(ring.dash);
                ctx.stroke();
            });
            ctx.setLineDash([]);

            // Axis labels
            ctx.font = 'bold 8px "JetBrains Mono", monospace';
            ctx.fillStyle = PALETTE.inkFaint;
            ctx.textAlign = 'center';
            ctx.fillText('+Q_c', cx + radius + 24, cy + 3);
            ctx.fillText('-Q_c', cx - radius - 24, cy + 3);
            ctx.fillText('+K_c', cx, cy - radius - 18);
            ctx.fillText('-K_c', cx, cy + radius + 22);

            ctx.restore();
        }

        function drawMLAManifold(t, probeNode) {
            ctx.save();
            var latentR = radius * 0.38;
            var ropeR = radius * 0.65;
            var headR = radius * 0.88;

            // 1. Central 192d Latent KV Manifold Core
            var pulse = Math.sin(t * 2.4) * 0.08 + 1.0 + (flashEffect * 0.2);
            var coreGrad = ctx.createRadialGradient(cx, cy, 2, cx, cy, latentR * pulse);
            coreGrad.addColorStop(0, 'rgba(224, 122, 63, 0.45)');
            coreGrad.addColorStop(0.5, 'rgba(201, 163, 92, 0.22)');
            coreGrad.addColorStop(1, 'rgba(14, 12, 10, 0)');
            ctx.beginPath();
            ctx.arc(cx, cy, latentR * pulse, 0, Math.PI * 2);
            ctx.fillStyle = coreGrad;
            ctx.fill();

            // Core boundary ring
            ctx.beginPath();
            ctx.arc(cx, cy, latentR, 0, Math.PI * 2);
            ctx.strokeStyle = PALETTE.terracotta;
            ctx.lineWidth = 1.5;
            ctx.stroke();

            ctx.fillStyle = PALETTE.terracotta;
            ctx.font = 'bold 8.5px "JetBrains Mono", monospace';
            ctx.textAlign = 'center';
            ctx.fillText('192d LATENT KV CORE', cx, cy - 8);
            ctx.fillStyle = PALETTE.inkSoft;
            ctx.font = '7.5px "JetBrains Mono", monospace';
            ctx.fillText('7.1× KV REDUCTION (216d)', cx, cy + 6);
            ctx.fillText('W_D · h_t', cx, cy + 18);

            // 2. 12 Rotating Query Heads & Absorption Beams
            var numHeads = 12;
            for (var h = 0; h < numHeads; h++) {
                var baseAngle = (h / numHeads) * Math.PI * 2;
                var angle = baseAngle + t * (0.18 + (h % 3) * 0.04);
                var hx = cx + Math.cos(angle) * headR;
                var hy = cy + Math.sin(angle) * headR;

                // Matrix Absorption Beam from Head to Latent Core
                var beamAlpha = 0.30 + 0.45 * Math.sin(t * 3.0 + h) + (flashEffect * 0.3);
                ctx.beginPath();
                ctx.moveTo(hx, hy);
                ctx.lineTo(cx, cy);
                ctx.strokeStyle = 'rgba(224, 122, 63, ' + Math.min(1, beamAlpha).toFixed(2) + ')';
                ctx.lineWidth = (h % 2 === 0) ? 1.5 : 1.0;
                ctx.stroke();

                // Absorption annotation node on the beam
                var midX = cx + (hx - cx) * 0.55;
                var midY = cy + (hy - cy) * 0.55;
                ctx.beginPath();
                ctx.arc(midX, midY, 2.5, 0, Math.PI * 2);
                ctx.fillStyle = PALETTE.goldGlow;
                ctx.fill();

                // Decoupled RoPE node along 24d orbit
                var ropeAngle = baseAngle - t * 0.35;
                var rx = cx + Math.cos(ropeAngle) * ropeR;
                var ry = cy + Math.sin(ropeAngle) * ropeR;
                ctx.beginPath();
                ctx.arc(rx, ry, 2.8, 0, Math.PI * 2);
                ctx.fillStyle = PALETTE.oliveGlow;
                ctx.fill();

                // RoPE tether line
                ctx.beginPath();
                ctx.moveTo(hx, hy);
                ctx.lineTo(rx, ry);
                ctx.strokeStyle = 'rgba(154, 148, 64, 0.4)';
                ctx.lineWidth = 1;
                ctx.setLineDash([2, 4]);
                ctx.stroke();
                ctx.setLineDash([]);

                // Head Node
                ctx.beginPath();
                ctx.arc(hx, hy, 4.5, 0, Math.PI * 2);
                ctx.fillStyle = PALETTE.terracotta;
                ctx.fill();
                ctx.strokeStyle = '#fff';
                ctx.lineWidth = 1;
                ctx.stroke();

                // Head Label
                ctx.fillStyle = PALETTE.ink;
                ctx.font = 'bold 8px "JetBrains Mono", monospace';
                ctx.textAlign = 'center';
                ctx.fillText('H' + (h + 1), hx + Math.cos(angle) * 12, hy + Math.sin(angle) * 12 + 3);

                // Hover check
                var dHead = Math.hypot(mouseX - hx, mouseY - hy);
                if (dHead < 16 && !probeNode.found) {
                    probeNode.found = true;
                    probeNode.type = 'head';
                    probeNode.x = hx;
                    probeNode.y = hy;
                    probeNode.id = h + 1;
                    probeNode.angle = angle;
                }
            }

            // Inflowing Latent Particles
            particles.forEach(function (p) {
                if (!reduced) {
                    p.radiusFrac -= 0.12 * simSpeed * 0.016;
                    if (p.radiusFrac < 0.38) p.radiusFrac = 0.95;
                    p.theta += p.speed * 0.016 * simSpeed;
                }
                var pr = radius * p.radiusFrac;
                var px = cx + Math.cos(p.theta) * pr;
                var py = cy + Math.sin(p.theta) * pr;

                ctx.beginPath();
                ctx.arc(px, py, p.size, 0, Math.PI * 2);
                ctx.fillStyle = (p.radiusFrac > 0.65 ? PALETTE.oliveGlow : PALETTE.terracottaGlow);
                ctx.fill();
            });

            ctx.restore();
        }

        function drawMoERoutingField(t, probeNode) {
            ctx.save();
            var numExperts = 20;
            var expertR = radius * 0.90;
            var activeExperts = [];

            // Compute active top-4 experts deterministically based on time
            var stepCycle = Math.floor(t * 1.5);
            for (var k = 0; k < 4; k++) {
                activeExperts.push((stepCycle * 3 + k * 5) % numExperts);
            }

            // Central Shared Expert
            var sharedR = radius * 0.24;
            var sharedGrad = ctx.createRadialGradient(cx, cy, 2, cx, cy, sharedR);
            sharedGrad.addColorStop(0, 'rgba(201, 163, 92, 0.45)');
            sharedGrad.addColorStop(1, 'rgba(14, 12, 10, 0)');
            ctx.beginPath();
            ctx.arc(cx, cy, sharedR, 0, Math.PI * 2);
            ctx.fillStyle = sharedGrad;
            ctx.fill();

            ctx.beginPath();
            ctx.arc(cx, cy, sharedR * 0.75, 0, Math.PI * 2);
            ctx.strokeStyle = PALETTE.gold;
            ctx.lineWidth = 2;
            ctx.stroke();

            ctx.fillStyle = PALETTE.gold;
            ctx.font = 'bold 9px "JetBrains Mono", monospace';
            ctx.textAlign = 'center';
            ctx.fillText('SHARED EXP', cx, cy - 4);
            ctx.fillStyle = PALETTE.inkSoft;
            ctx.font = '7.5px "JetBrains Mono", monospace';
            ctx.fillText('ALWAYS ON', cx, cy + 8);

            // Draw 20 expert stations
            for (var e = 0; e < numExperts; e++) {
                var angle = (e / numExperts) * Math.PI * 2 - Math.PI / 2;
                var ex = cx + Math.cos(angle) * expertR;
                var ey = cy + Math.sin(angle) * expertR;

                var isActive = (activeExperts.indexOf(e) !== -1);

                // Dispatch Conduit from Center
                if (isActive) {
                    var beamAlpha = 0.5 + 0.4 * Math.sin(t * 6.0 + e) + (flashEffect * 0.3);
                    ctx.beginPath();
                    ctx.moveTo(cx, cy);
                    ctx.lineTo(ex, ey);
                    ctx.strokeStyle = 'rgba(224, 122, 63, ' + Math.min(1, beamAlpha).toFixed(2) + ')';
                    ctx.lineWidth = 2.0;
                    ctx.stroke();

                    // Active pulse ring
                    ctx.beginPath();
                    ctx.arc(ex, ey, 10 + 4 * Math.sin(t * 8.0 + e), 0, Math.PI * 2);
                    ctx.strokeStyle = PALETTE.terracottaGlow;
                    ctx.lineWidth = 1;
                    ctx.stroke();
                }

                // Expert Station Node
                ctx.beginPath();
                ctx.arc(ex, ey, isActive ? 6.5 : 4.5, 0, Math.PI * 2);
                ctx.fillStyle = isActive ? PALETTE.terracotta : PALETTE.paperBg;
                ctx.fill();
                ctx.strokeStyle = isActive ? PALETTE.gold : PALETTE.ruleStrong;
                ctx.lineWidth = isActive ? 2 : 1;
                ctx.stroke();

                // Dynamic Bias Bar Arc
                var biasVal = (Math.sin(e * 1.3 + t * 0.8) * 0.4 + 0.5);
                ctx.beginPath();
                ctx.arc(ex, ey, 8.5, angle - Math.PI * 0.4, angle - Math.PI * 0.4 + biasVal * Math.PI * 0.8);
                ctx.strokeStyle = isActive ? PALETTE.gold : 'rgba(154, 148, 64, 0.5)';
                ctx.lineWidth = 1.5;
                ctx.stroke();

                // Label
                ctx.fillStyle = isActive ? '#ffffff' : PALETTE.inkFaint;
                ctx.font = (isActive ? 'bold 8.5px' : '7.5px') + ' "JetBrains Mono", monospace';
                ctx.textAlign = 'center';
                var labelR = expertR + (isActive ? 16 : 14);
                ctx.fillText((e < 10 ? '0' : '') + e, cx + Math.cos(angle) * labelR, cy + Math.sin(angle) * labelR + 3);

                // Hover check
                var dExp = Math.hypot(mouseX - ex, mouseY - ey);
                if (dExp < 16 && !probeNode.found) {
                    probeNode.found = true;
                    probeNode.type = 'expert';
                    probeNode.x = ex;
                    probeNode.y = ey;
                    probeNode.id = e;
                    probeNode.active = isActive;
                    probeNode.bias = biasVal;
                }
            }

            ctx.restore();
        }

        function drawMTPTreeField(t, probeNode) {
            ctx.save();
            // Main Model Core & MTP Speculative Branch
            var mainX = cx - radius * 0.45;
            var mainY = cy;
            var mtpX = cx + radius * 0.45;
            var mtpY = cy;

            // Connecting Speculative Synapse Bridge
            var synAlpha = 0.5 + 0.4 * Math.sin(t * 5.0) + (flashEffect * 0.3);
            ctx.beginPath();
            ctx.moveTo(mainX, mainY);
            ctx.lineTo(mtpX, mtpY);
            ctx.strokeStyle = 'rgba(154, 148, 64, ' + Math.min(1, synAlpha).toFixed(2) + ')';
            ctx.lineWidth = 3.0;
            ctx.stroke();

            // Main Model Node
            ctx.beginPath();
            ctx.arc(mainX, mainY, 28, 0, Math.PI * 2);
            ctx.fillStyle = 'rgba(224, 122, 63, 0.25)';
            ctx.fill();
            ctx.strokeStyle = PALETTE.terracotta;
            ctx.lineWidth = 2;
            ctx.stroke();

            ctx.fillStyle = PALETTE.terracotta;
            ctx.font = 'bold 9px "JetBrains Mono", monospace';
            ctx.textAlign = 'center';
            ctx.fillText('MAIN MODEL', mainX, mainY - 6);
            ctx.fillStyle = PALETTE.inkSoft;
            ctx.font = '7.5px "JetBrains Mono", monospace';
            ctx.fillText('Draft tok_t', mainX, mainY + 8);

            // MTP Speculative Node
            ctx.beginPath();
            ctx.arc(mtpX, mtpY, 28, 0, Math.PI * 2);
            ctx.fillStyle = 'rgba(154, 148, 64, 0.25)';
            ctx.fill();
            ctx.strokeStyle = PALETTE.olive;
            ctx.lineWidth = 2;
            ctx.stroke();

            ctx.fillStyle = PALETTE.olive;
            ctx.font = 'bold 9px "JetBrains Mono", monospace';
            ctx.textAlign = 'center';
            ctx.fillText('MTP HEAD (D1)', mtpX, mtpY - 6);
            ctx.fillStyle = PALETTE.inkSoft;
            ctx.font = '7.5px "JetBrains Mono", monospace';
            ctx.fillText('Draft tok_{t+1}', mtpX, mtpY + 8);

            // Orbiting Speculative Verification Nodes
            var numTokens = 8;
            for (var k = 0; k < numTokens; k++) {
                var tokAngle = (k / numTokens) * Math.PI * 2 + t * 0.8;
                var tx = mtpX + Math.cos(tokAngle) * 56;
                var ty = mtpY + Math.sin(tokAngle) * 44;

                ctx.beginPath();
                ctx.arc(tx, ty, 3.5, 0, Math.PI * 2);
                ctx.fillStyle = (k % 2 === 0 ? PALETTE.oliveGlow : PALETTE.terracottaGlow);
                ctx.fill();

                ctx.beginPath();
                ctx.moveTo(mtpX, mtpY);
                ctx.lineTo(tx, ty);
                ctx.strokeStyle = 'rgba(154, 148, 64, 0.3)';
                ctx.lineWidth = 1;
                ctx.stroke();
            }

            ctx.restore();
        }

        function drawDualPipeSchedule(t, probeNode) {
            ctx.save();
            var numStages = 8;
            var rx = radius * 0.85;
            var ry = radius * 0.52;

            // Racetrack Path
            ctx.beginPath();
            ctx.ellipse(cx, cy, rx, ry, 0, 0, Math.PI * 2);
            ctx.strokeStyle = 'rgba(58, 50, 38, 0.4)';
            ctx.lineWidth = 2;
            ctx.stroke();

            // Center overlap badge
            ctx.fillStyle = 'rgba(22, 19, 16, 0.9)';
            ctx.fillRect(cx - 75, cy - 24, 150, 48);
            ctx.strokeStyle = PALETTE.gold;
            ctx.strokeRect(cx - 75, cy - 24, 150, 48);

            ctx.fillStyle = PALETTE.gold;
            ctx.font = 'bold 9px "JetBrains Mono", monospace';
            ctx.textAlign = 'center';
            ctx.fillText('100% OVERLAP CORE', cx, cy - 6);
            ctx.fillStyle = PALETTE.olive;
            ctx.font = '7.5px "JetBrains Mono", monospace';
            ctx.fillText('1F1B FWD ⇄ BWD STREAM', cx, cy + 8);

            // Draw PP Stage nodes
            for (var s = 0; s < numStages; s++) {
                var angle = (s / numStages) * Math.PI * 2;
                var sx = cx + Math.cos(angle) * rx;
                var sy = cy + Math.sin(angle) * ry;

                ctx.beginPath();
                ctx.arc(sx, sy, 5, 0, Math.PI * 2);
                ctx.fillStyle = PALETTE.paperBg;
                ctx.fill();
                ctx.strokeStyle = PALETTE.ruleStrong;
                ctx.lineWidth = 1.5;
                ctx.stroke();

                ctx.fillStyle = PALETTE.ink;
                ctx.font = 'bold 8px "JetBrains Mono", monospace';
                ctx.textAlign = 'center';
                ctx.fillText('PP' + s, sx + Math.cos(angle) * 14, sy + Math.sin(angle) * 14 + 3);
            }

            // Forward microbatches (Terracotta, clockwise)
            for (var f = 0; f < 16; f++) {
                var fAngle = (f / 16) * Math.PI * 2 + t * 0.45;
                var fx = cx + Math.cos(fAngle) * rx;
                var fy = cy + Math.sin(fAngle) * ry;

                ctx.beginPath();
                ctx.arc(fx, fy, 2.5, 0, Math.PI * 2);
                ctx.fillStyle = PALETTE.terracottaGlow;
                ctx.fill();
            }

            // Backward microbatches (Olive, counter-clockwise)
            for (var b = 0; b < 16; b++) {
                var bAngle = (b / 16) * Math.PI * 2 - t * 0.45;
                var bx = cx + Math.cos(bAngle) * (rx * 0.78);
                var by = cy + Math.sin(bAngle) * (ry * 0.78);

                ctx.beginPath();
                ctx.arc(bx, by, 2.5, 0, Math.PI * 2);
                ctx.fillStyle = PALETTE.oliveGlow;
                ctx.fill();
            }

            ctx.restore();
        }

        function drawShockwaves(dt) {
            ctx.save();
            for (var i = shockwaves.length - 1; i >= 0; i--) {
                var sw = shockwaves[i];
                sw.radius += (sw.maxRadius - sw.radius) * 4.0 * dt;
                sw.life -= dt * 1.8;
                if (sw.life <= 0) {
                    shockwaves.splice(i, 1);
                    continue;
                }
                ctx.beginPath();
                ctx.arc(sw.x, sw.y, sw.radius, 0, Math.PI * 2);
                ctx.strokeStyle = sw.color;
                ctx.globalAlpha = sw.life * 0.6;
                ctx.lineWidth = 2 * sw.life;
                ctx.stroke();
            }
            ctx.restore();
        }

        function updateProbeHUD(probeNode) {
            if (!probeEl) return;
            if (probeNode.found) {
                probeEl.style.opacity = '1';
                var targetX = probeNode.x;
                var targetY = probeNode.y;

                if (probeNode.type === 'head') {
                    probeTag.textContent = 'QUERY HEAD #' + (probeNode.id < 10 ? '0' : '') + probeNode.id + ' / 12';
                    probeCoords.textContent = 'q\'_t = W_q h_t · W_U [192d latent absorption]';
                    probeDecay.textContent = 'Decoupled RoPE k_t [24d] · θ = 10K';
                } else if (probeNode.type === 'expert') {
                    probeTag.textContent = 'EXPERT STATION #' + (probeNode.id < 10 ? '0' : '') + probeNode.id + ' / 20';
                    probeCoords.textContent = (probeNode.active ? 'ACTIVE (TOP-4 DISPATCH)' : 'STANDBY EXPERT');
                    probeDecay.textContent = 'Dynamic Load Bias b_i = +' + probeNode.bias.toFixed(3);
                }

                var containerRect = container.getBoundingClientRect();
                var cardW = 280;
                var cardX = targetX + 16;
                var cardY = targetY - 24;
                if (cardX + cardW > containerRect.width - 12) cardX = targetX - cardW - 16;
                if (cardX < 12) cardX = 12;
                if (cardY < 10) cardY = 10;
                probeEl.style.left = Math.round(cardX) + 'px';
                probeEl.style.top = Math.round(cardY) + 'px';
            } else {
                probeEl.style.opacity = '0';
            }
        }

        function renderFrame(dt, force) {
            if (!force && isPaused) return;

            simTime += dt * simSpeed;
            if (flashEffect > 0) flashEffect = Math.max(0, flashEffect - dt * 2.0);

            ctx.clearRect(0, 0, width, height);
            ctx.fillStyle = PALETTE.paperBg;
            ctx.fillRect(0, 0, width, height);

            drawDialAndGrid(simTime);

            var probeNode = { found: false };

            if (currentMode === 'mla') {
                drawMLAManifold(simTime, probeNode);
            } else if (currentMode === 'moe') {
                drawMoERoutingField(simTime, probeNode);
            } else if (currentMode === 'mtp') {
                drawMTPTreeField(simTime, probeNode);
            } else if (currentMode === 'dualpipe') {
                drawDualPipeSchedule(simTime, probeNode);
            }

            drawShockwaves(dt);
            updateProbeHUD(probeNode);
        }

        function loop(timestamp) {
            if (!lastTime) lastTime = timestamp;
            var dt = Math.min((timestamp - lastTime) / 1000, 0.1);
            lastTime = timestamp;

            renderFrame(dt, false);

            if (!isPaused && !reduced) {
                animId = requestAnimationFrame(loop);
            }
        }

        if (reduced) {
            renderFrame(0, true);
        } else {
            lastTime = performance.now();
            animId = requestAnimationFrame(loop);
            document.addEventListener('visibilitychange', function () {
                if (document.hidden) {
                    if (animId) { cancelAnimationFrame(animId); animId = null; }
                } else if (!isPaused && !animId) {
                    lastTime = performance.now();
                    animId = requestAnimationFrame(loop);
                }
            });
        }
    }

    // ------------------------------------------------------------------
    // 2. Interactive Mechanism Exploder Suite (3 Labs)
    // ------------------------------------------------------------------
    function initMechanismExploders() {
        // Lab 1: MLA VRAM Calculator
        var mlaSlider = document.getElementById('mlaContextSlider');
        var mlaLabel = document.getElementById('mlaContextLabel');
        var statMha = document.getElementById('statMhaVram');
        var statMla = document.getElementById('statMlaVram');
        var statSaved = document.getElementById('statMlaSaved');
        var absorbBtn = document.getElementById('mlaAbsorbAnimBtn');

        if (mlaSlider && mlaLabel) {
            function updateMlaCalc() {
                var L = parseInt(mlaSlider.value, 10);
                mlaLabel.textContent = L.toLocaleString() + ' tokens';

                // Standard MHA: 2 * 18 layers * L * 12 heads * 128 head_dim * 2 bytes
                var mhaBytes = 2 * 18 * L * 12 * 128 * 2;
                // DeepSeek MLA: 18 layers * L * (192 latent + 24 rope) * 2 bytes
                var mlaBytes = 18 * L * (192 + 24) * 2;

                var mhaGB = (mhaBytes / (1024 * 1024 * 1024)).toFixed(2);
                var mlaGB = (mlaBytes / (1024 * 1024 * 1024)).toFixed(2);
                var savedGB = ((mhaBytes - mlaBytes) / (1024 * 1024 * 1024)).toFixed(2);

                if (statMha) statMha.textContent = mhaGB + ' GB';
                if (statMla) statMla.textContent = mlaGB + ' GB';
                if (statSaved) statSaved.textContent = savedGB + ' GB (7.1× cut)';
            }
            mlaSlider.addEventListener('input', updateMlaCalc);
            mlaSlider.addEventListener('change', updateMlaCalc);
            updateMlaCalc();
        }

        if (absorbBtn) {
            absorbBtn.addEventListener('click', function () {
                if (heroController.triggerAbsorptionFlash) {
                    heroController.triggerAbsorptionFlash();
                }
                var origText = absorbBtn.textContent;
                absorbBtn.textContent = '✓ Absorbing: W_U · W_q → q\'';
                setTimeout(function () {
                    absorbBtn.textContent = origText;
                }, 1600);
            });
        }

        // Lab 2: DeepSeekMoE Router Playground
        var miniGrid = document.getElementById('moeMiniGrid');
        var activeListEl = document.getElementById('moeActiveList');
        var routeBatchBtn = document.getElementById('moeRouteBatchBtn');

        function updateMoEActiveListFromDOM() {
            if (!miniGrid || !activeListEl) return;
            var chosen = [];
            miniGrid.querySelectorAll('.moe-mini-cell.active:not(.shared)').forEach(function (c) {
                var expNum = parseInt(c.getAttribute('data-exp'), 10);
                if (!isNaN(expNum)) chosen.push(expNum);
            });
            chosen.sort(function (a, b) { return a - b; });
            activeListEl.textContent = (chosen.length ? chosen.map(function (idx) {
                return '#' + (idx < 10 ? '0' : '') + idx;
            }).join(', ') : 'None') + ' + shared';
        }

        if (miniGrid) {
            var gridHtml = [];
            for (var i = 0; i < 20; i++) {
                var pad = (i < 10 ? '0' : '') + i;
                gridHtml.push('<div class="moe-mini-cell' + (i === 2 || i === 7 || i === 13 || i === 18 ? ' active' : '') + '" data-exp="' + i + '" style="cursor: pointer;" title="Toggle Expert #' + pad + '">E' + pad + '</div>');
            }
            gridHtml.push('<div class="moe-mini-cell shared">Shared Expert (Always Active)</div>');
            miniGrid.innerHTML = gridHtml.join('');

            // Click listener for individual cells
            miniGrid.querySelectorAll('.moe-mini-cell:not(.shared)').forEach(function (cell) {
                cell.addEventListener('click', function () {
                    cell.classList.toggle('active');
                    updateMoEActiveListFromDOM();
                    if (heroController.triggerMoEPulse) heroController.triggerMoEPulse();
                });
            });
        }

        if (routeBatchBtn && miniGrid) {
            routeBatchBtn.addEventListener('click', function () {
                var chosen = [];
                while (chosen.length < 4) {
                    var r = Math.floor(Math.random() * 20);
                    if (chosen.indexOf(r) === -1) chosen.push(r);
                }
                chosen.sort(function (a, b) { return a - b; });

                var cells = miniGrid.querySelectorAll('.moe-mini-cell:not(.shared)');
                cells.forEach(function (c, idx) {
                    c.classList.toggle('active', chosen.indexOf(idx) !== -1);
                });

                updateMoEActiveListFromDOM();

                if (heroController.triggerMoEPulse) {
                    heroController.triggerMoEPulse();
                }
            });
        }

        // Lab 3: MTP Speculative Verification Sandbox
        var verifyBtn = document.getElementById('mtpVerifyStepBtn');
        var treeDisplay = document.getElementById('mtpTreeDisplay');
        var mtpStatusText = document.getElementById('mtpStatusText');
        var mtpSpeedText = document.getElementById('mtpSpeedText');

        var SAMPLE_PAIRS = [
            ["multi-head", "latent"],
            ["matrix", "absorption"],
            ["dynamic", "biasing"],
            ["auxiliary-loss", "free"],
            ["speculative", "verification"],
            ["dualpipe", "concurrency"],
            ["grouped", "gemm"],
            ["chinchilla", "optimal"]
        ];
        var pairIdx = 0;

        if (verifyBtn && treeDisplay) {
            verifyBtn.addEventListener('click', function () {
                pairIdx = (pairIdx + 1) % SAMPLE_PAIRS.length;
                var pair = SAMPLE_PAIRS[pairIdx];
                var pVal = (0.74 + Math.random() * 0.24).toFixed(2);
                var isAcc = parseFloat(pVal) >= 0.80;

                var draftBadge = isAcc ? 'MTP ACCEPTED ✓' : 'MTP REJECTED ✗';
                var badgeClass = isAcc ? 'acc' : 'draft';

                treeDisplay.innerHTML = [
                    '<div class="mtp-branch"><span class="mtp-badge draft">MAIN HEAD</span><span>Token t &rarr; "' + pair[0] + '" (p = 0.99)</span></div>',
                    '<div class="mtp-branch"><span class="mtp-badge ' + badgeClass + '">' + draftBadge + '</span><span>Token t+1 &rarr; "' + pair[1] + '" (p = ' + pVal + ')</span></div>'
                ].join('');

                if (mtpStatusText) {
                    mtpStatusText.textContent = isAcc ? 'ACCEPTED (2 Tokens / Step)' : 'REJECTED (Fall Back to 1 Token)';
                }

                if (mtpSpeedText) {
                    mtpSpeedText.textContent = isAcc ? '2.00× effective rate' : '1.00× fallback rate';
                }

                if (heroController.triggerMTPPulse) {
                    heroController.triggerMTPPulse();
                }
            });
        }
    }

    // ------------------------------------------------------------------
    // 3. Living Pipeline Pass Telemetry (FIG · A1)
    // ------------------------------------------------------------------
    function initPassDiagram() {
        var canvas = document.getElementById('passDiagramCanvas');
        if (!canvas) return;

        var ctx = canvas.getContext('2d');
        if (!ctx) return;

        var container = canvas.parentElement;
        var tooltipEl = document.getElementById('passStageTooltip');
        var stTag = document.getElementById('stTag');
        var stOp = document.getElementById('stOp');
        var stShape = document.getElementById('stShape');
        var stDesc = document.getElementById('stDesc');
        var phaseEl = document.getElementById('phCurrentPhase');
        var tickerText = document.getElementById('passTickerText');
        var tickerBeacon = document.getElementById('passTickerBeacon');
        var pauseBtn = document.getElementById('passPauseBtn');
        var phaseBtns = document.querySelectorAll('.pass-controls .pass-btn[data-phase]');

        var currentPhaseMode = 'cycle';
        var isPaused = false;
        var isHovered = false;
        var mouseX = -9999, mouseY = -9999;
        var hoveredStation = null;
        var animId = null;
        var lastTime = 0;
        var simTime = 0;

        var PALETTE = {
            paperBg: '#0e0c0a',
            paperStation: '#161310',
            paperStationHover: '#1c1813',
            rule: '#2c261c',
            ruleStrong: '#3a3226',
            terracotta: '#e07a3f',
            terracottaGlow: 'rgba(224, 122, 63, 0.75)',
            terracottaTint: 'rgba(224, 122, 63, 0.16)',
            olive: '#9a9440',
            oliveGlow: 'rgba(154, 148, 64, 0.75)',
            oliveTint: 'rgba(154, 148, 64, 0.16)',
            gold: '#c9a35c',
            goldGlow: 'rgba(201, 163, 92, 0.85)',
            goldTint: 'rgba(201, 163, 92, 0.20)',
            ink: '#d8ccb4',
            inkSoft: '#b3a68c',
            inkFaint: '#7a7160'
        };

        var STAGES = [
            {
                id: 1,
                tag: 'STAGE 01 · EMBEDDING',
                badge: '01 · EMB',
                title: 'Embedding',
                sub: 'x_t → 768',
                chip: 'Tied Weights',
                op: 'h_0 = Embedding(x_t) · Weight-Tied with LM Head',
                shape: 'Input: [B, 4096] uint32 → Output: [B, 4096, 768] bf16',
                desc: 'DeepSeek-Coder-V2 vocab (100,018) · Tied embedding/head parameter matrix',
                isParam: true
            },
            {
                id: 2,
                tag: 'STAGE 02 · MLA ATTENTION',
                badge: '02 · MLA',
                title: 'MLA Attention',
                sub: '192d c_t + 24d R',
                chip: '7.1× KV Cut',
                op: 'c_t = W_D · h_t [192d] · q\'_t = W_q h_t · W_U',
                shape: 'Latent KV: [B, 4096, 192] bf16 + RoPE [B, 4096, 24]',
                desc: 'Multi-Head Latent Attention with matrix absorption (7.1× KV reduction)',
                isParam: true
            },
            {
                id: 3,
                tag: 'STAGE 03 · DENSE WARMUP',
                badge: '03 · DENSE',
                title: 'Dense Warmup',
                sub: 'Layers 00–01',
                chip: 'SwiGLU Dense',
                op: 'h_l = h_{l-1} + SwiGLU(RMSNorm(h_{l-1}))',
                shape: 'Dense intermediate: 1536d bf16 · 2 Initial Blocks',
                desc: '2 initial dense representation warmup layers before MoE expert routing',
                isParam: true
            },
            {
                id: 4,
                tag: 'STAGE 04 · MoE ROUTER',
                badge: '04 · MOE',
                title: 'MoE Router',
                sub: 'Top-4 / 20 Exp',
                chip: 'Aux-Loss-Free',
                op: 's = sigmoid(W_g x) + b, top-4 routed + 1 shared',
                shape: 'Gating: [B, 4096, 20] probs + dynamic bias Δb',
                desc: 'Auxiliary-loss-free load balancing with out-of-band dynamic bias updates',
                isParam: true
            },
            {
                id: 5,
                tag: 'STAGE 05 · TRITON GROUPED-GEMM',
                badge: '05 · TRITON',
                title: 'Triton GEMM',
                sub: 'Fused 20-Exp',
                chip: '1920d Active',
                op: 'y = FusedGroupedGEMM(x, expert_weights) + Shared(x)',
                shape: 'SwiGLU: 5 active experts × 384d = 1920d bf16',
                desc: 'Custom Triton grouped-GEMM (BLOCK_T=16) fuses all 20 experts in 1 launch',
                isParam: true
            },
            {
                id: 6,
                tag: 'STAGE 06 · MTP HEAD & LOSS',
                badge: '06 · LOSS',
                title: 'MTP & Loss',
                sub: 'Chunked CE',
                chip: 'μP FP32 Master',
                op: 'ℓ = CE(logits, y) + λ·CE(MTP, y_{+1}); θ ← θ − η·AdamW(∇θ)',
                shape: 'Logits: [B, 4096, 100018] in chunks of 4096 → Loss ℓ',
                desc: 'MTP speculative auxiliary depth-1 head + chunked CE + μP AdamW master',
                isParam: true
            }
        ];

        var forwardParticles = [];
        for (var f = 0; f < 24; f++) {
            forwardParticles.push({
                xFrac: f / 24,
                speed: 0.18 + 0.08 * Math.random(),
                size: 1.6 + 1.0 * Math.random(),
                lane: (f % 3) - 1
            });
        }

        var backwardParticles = [];
        for (var b = 0; b < 24; b++) {
            backwardParticles.push({
                xFrac: b / 24,
                speed: 0.20 + 0.08 * Math.random(),
                size: 1.6 + 1.0 * Math.random(),
                lane: (b % 3) - 1
            });
        }

        var adamParticles = [];
        var width = 0, height = 0;
        var stations = [];

        function layoutStations() {
            var rect = container.getBoundingClientRect();
            var dpr = Math.min(window.devicePixelRatio || 1, 2);
            width = rect.width;
            height = rect.height || 260;
            canvas.width = Math.round(width * dpr);
            canvas.height = Math.round(height * dpr);
            ctx.setTransform(1, 0, 0, 1, 0, 0);
            ctx.scale(dpr, dpr);

            stations = [];
            var numStages = STAGES.length;
            var marginX = 16;
            var availW = width - (marginX * 2);
            var gap = Math.max(8, Math.min(16, (availW - (numStages * 82)) / (numStages - 1)));
            var stationW = Math.max(76, (availW - (gap * (numStages - 1))) / numStages);
            var stationH = Math.min(124, height * 0.52);
            var stationY = (height - stationH) / 2 + 2;

            for (var i = 0; i < numStages; i++) {
                var sx = marginX + i * (stationW + gap);
                stations.push({
                    stage: STAGES[i],
                    x: sx,
                    y: stationY,
                    w: stationW,
                    h: stationH,
                    cx: sx + stationW / 2,
                    cy: stationY + stationH / 2,
                    glowForward: 0,
                    glowBackward: 0,
                    glowAdam: 0
                });
            }
        }

        if (window.ResizeObserver) {
            var ro = new ResizeObserver(function () {
                layoutStations();
                if (reduced || isPaused) render(0, true);
            });
            ro.observe(container);
        } else {
            window.addEventListener('resize', layoutStations);
        }
        layoutStations();

        // Mouse Listeners
        container.addEventListener('mousemove', function (e) {
            var rect = canvas.getBoundingClientRect();
            mouseX = e.clientX - rect.left;
            mouseY = e.clientY - rect.top;
            isHovered = true;
        });

        container.addEventListener('mouseleave', function () {
            isHovered = false;
            mouseX = -9999;
            mouseY = -9999;
            hoveredStation = null;
            if (tooltipEl) tooltipEl.style.opacity = '0';
        });

        container.addEventListener('click', function (e) {
            var rect = canvas.getBoundingClientRect();
            var clickX = e.clientX - rect.left;
            var clickY = e.clientY - rect.top;

            stations.forEach(function (st) {
                if (clickX >= st.x && clickX <= st.x + st.w && clickY >= st.y && clickY <= st.y + st.h) {
                    st.glowForward = 1.0;
                    st.glowBackward = 1.0;
                    for (var k = 0; k < 12; k++) {
                        adamParticles.push({
                            x: st.cx,
                            y: st.cy,
                            vx: (Math.random() - 0.5) * 160,
                            vy: (Math.random() - 0.5) * 160,
                            life: 1.0,
                            color: PALETTE.gold
                        });
                    }
                }
            });
        });

        if (pauseBtn) {
            pauseBtn.addEventListener('click', function (e) {
                e.stopPropagation();
                isPaused = !isPaused;
                pauseBtn.innerHTML = isPaused ? '&#9658;' : '&#10074;&#10074;';
                pauseBtn.setAttribute('aria-label', isPaused ? 'Resume pipeline' : 'Pause pipeline');
                if (!isPaused && !animId) {
                    lastTime = performance.now();
                    loop(lastTime);
                }
            });
        }

        phaseBtns.forEach(function (btn) {
            btn.addEventListener('click', function (e) {
                e.stopPropagation();
                phaseBtns.forEach(function (b) { b.classList.remove('active'); });
                btn.classList.add('active');
                currentPhaseMode = btn.getAttribute('data-phase') || 'cycle';
                if (reduced || isPaused) render(0, true);
            });
        });

        var CYCLE_DURATION = 8.6;

        function getCycleState(t) {
            if (currentPhaseMode === 'forward') {
                return { phase: 'forward', progress: (t * 0.4) % 1.0, subText: 'FORWARD ACTIVATIONS STREAMING (1F1B DUALPIPE)' };
            }
            if (currentPhaseMode === 'backward') {
                return { phase: 'backward', progress: (t * 0.4) % 1.0, subText: 'AUTOGRAD GRADIENT PROPAGATION & CHECKPOINTING' };
            }

            var cycleT = t % CYCLE_DURATION;
            if (cycleT < 3.2) {
                return { phase: 'forward', progress: cycleT / 3.2, subText: 'FORWARD · Activations x_t → 18 Layers (MLA & MoE) → MTP Auxiliary Head' };
            } else if (cycleT < 4.2) {
                return { phase: 'loss', progress: (cycleT - 3.2) / 1.0, subText: 'LOSS COMPUTATION · Chunked Cross-Entropy + MTP Speculative Loss' };
            } else if (cycleT < 7.0) {
                return { phase: 'backward', progress: (cycleT - 4.2) / 2.8, subText: 'AUTOGRAD BACKWARD · Checkpointing Recomputes Activations (FA2 Pattern)' };
            } else if (cycleT < 8.0) {
                return { phase: 'adam', progress: (cycleT - 7.0) / 1.0, subText: 'ADAMW STEP · μP FP32 Master Updates θ ← θ - η·∇ℓ & Dynamic Bias Δb' };
            } else {
                return { phase: 'rest', progress: (cycleT - 8.0) / 0.6, subText: 'STEP COMMITTED · Next Mini-Batch Ingest' };
            }
        }

        function drawBackground() {
            ctx.fillStyle = PALETTE.paperBg;
            ctx.fillRect(0, 0, width, height);

            ctx.save();
            // Subtle technical grid
            ctx.strokeStyle = 'rgba(58, 50, 38, 0.22)';
            ctx.lineWidth = 1;
            ctx.setLineDash([2, 8]);
            for (var x = 20; x < width; x += 40) {
                ctx.beginPath();
                ctx.moveTo(x, 0);
                ctx.lineTo(x, height);
                ctx.stroke();
            }
            ctx.setLineDash([]);

            // Upper Forward Conduit
            var fwdY = height * 0.18;
            ctx.beginPath();
            ctx.moveTo(14, fwdY);
            ctx.lineTo(width - 14, fwdY);
            ctx.strokeStyle = 'rgba(224, 122, 63, 0.28)';
            ctx.lineWidth = 1.5;
            ctx.setLineDash([4, 6]);
            ctx.stroke();

            // Lower Backward Conduit
            var bwdY = height * 0.82;
            ctx.beginPath();
            ctx.moveTo(14, bwdY);
            ctx.lineTo(width - 14, bwdY);
            ctx.strokeStyle = 'rgba(154, 148, 64, 0.28)';
            ctx.lineWidth = 1.5;
            ctx.setLineDash([4, 6]);
            ctx.stroke();
            ctx.setLineDash([]);

            ctx.fillStyle = PALETTE.terracotta;
            ctx.font = 'bold 8.5px "JetBrains Mono", monospace';
            ctx.textAlign = 'left';
            ctx.fillText('FORWARD CONDUIT (ACTIVATIONS) →', 20, fwdY - 6);

            ctx.fillStyle = PALETTE.olive;
            ctx.font = 'bold 8.5px "JetBrains Mono", monospace';
            ctx.textAlign = 'right';
            ctx.fillText('← BACKWARD CONDUIT (GRADIENTS)', width - 20, bwdY + 14);
            ctx.restore();
        }

        function drawParticles(state, dt) {
            ctx.save();
            var fwdY = height * 0.18;
            var bwdY = height * 0.82;

            // Forward streaming particles
            if (state.phase === 'forward' || state.phase === 'loss' || currentPhaseMode === 'forward') {
                forwardParticles.forEach(function (p) {
                    if (!reduced) p.xFrac += p.speed * dt;
                    if (p.xFrac > 1.0) p.xFrac = 0.0;

                    var px = 16 + p.xFrac * (width - 32);
                    var py = fwdY + p.lane * 3.0;

                    ctx.beginPath();
                    ctx.arc(px, py, p.size, 0, Math.PI * 2);
                    ctx.fillStyle = PALETTE.terracottaGlow;
                    ctx.fill();

                    // Forward motion trail
                    ctx.beginPath();
                    ctx.moveTo(px, py);
                    ctx.lineTo(px - 14 * p.speed, py);
                    ctx.strokeStyle = 'rgba(224, 122, 63, 0.35)';
                    ctx.lineWidth = p.size * 0.7;
                    ctx.stroke();
                });
            }

            // Backward streaming particles
            if (state.phase === 'backward' || currentPhaseMode === 'backward') {
                backwardParticles.forEach(function (p) {
                    if (!reduced) p.xFrac += p.speed * dt;
                    if (p.xFrac > 1.0) p.xFrac = 0.0;

                    var px = width - 16 - p.xFrac * (width - 32);
                    var py = bwdY + p.lane * 3.0;

                    ctx.beginPath();
                    ctx.arc(px, py, p.size, 0, Math.PI * 2);
                    ctx.fillStyle = PALETTE.oliveGlow;
                    ctx.fill();

                    // Backward motion trail
                    ctx.beginPath();
                    ctx.moveTo(px, py);
                    ctx.lineTo(px + 14 * p.speed, py);
                    ctx.strokeStyle = 'rgba(154, 148, 64, 0.35)';
                    ctx.lineWidth = p.size * 0.7;
                    ctx.stroke();
                });
            }

            // AdamW Burst Particles
            for (var k = adamParticles.length - 1; k >= 0; k--) {
                var ap = adamParticles[k];
                ap.x += ap.vx * dt;
                ap.y += ap.vy * dt;
                ap.life -= dt * 1.6;

                if (ap.life <= 0) {
                    adamParticles.splice(k, 1);
                    continue;
                }

                ctx.beginPath();
                ctx.arc(ap.x, ap.y, 2.0 * ap.life, 0, Math.PI * 2);
                ctx.fillStyle = 'rgba(201, 163, 92, ' + ap.life.toFixed(2) + ')';
                ctx.fill();
            }

            ctx.restore();
        }

        function drawStations(state, dt) {
            ctx.save();
            var closest = null;
            var numStations = stations.length;

            stations.forEach(function (st, idx) {
                var frac = idx / (numStations - 1);

                // Compute activation level
                var isFwdActive = false;
                var isBwdActive = false;
                var isAdamActive = (state.phase === 'adam');

                if (state.phase === 'forward') {
                    isFwdActive = state.progress >= (frac - 0.12) && state.progress <= (frac + 0.18);
                } else if (state.phase === 'loss') {
                    isFwdActive = (idx === numStations - 1);
                } else if (state.phase === 'backward') {
                    var bwdFrac = 1.0 - frac;
                    isBwdActive = state.progress >= (bwdFrac - 0.12) && state.progress <= (bwdFrac + 0.18);
                }

                if (isFwdActive) st.glowForward = 1.0;
                else st.glowForward = Math.max(0, st.glowForward - dt * 2.2);

                if (isBwdActive) st.glowBackward = 1.0;
                else st.glowBackward = Math.max(0, st.glowBackward - dt * 2.2);

                if (isAdamActive) st.glowAdam = 1.0;
                else st.glowAdam = Math.max(0, st.glowAdam - dt * 2.0);

                // Check Hover
                var isHover = isHovered && (mouseX >= st.x && mouseX <= st.x + st.w && mouseY >= st.y && mouseY <= st.y + st.h);
                if (isHover) closest = st;

                // Station Box Drawing
                var cardBg = isHover ? PALETTE.paperStationHover : PALETTE.paperStation;
                var borderColor = PALETTE.ruleStrong;

                if (st.glowAdam > 0.1) {
                    borderColor = PALETTE.gold;
                    cardBg = 'rgba(201, 163, 92, ' + (0.15 * st.glowAdam).toFixed(2) + ')';
                } else if (st.glowForward > 0.1) {
                    borderColor = PALETTE.terracotta;
                    cardBg = 'rgba(224, 122, 63, ' + (0.14 * st.glowForward).toFixed(2) + ')';
                } else if (st.glowBackward > 0.1) {
                    borderColor = PALETTE.olive;
                    cardBg = 'rgba(154, 148, 64, ' + (0.14 * st.glowBackward).toFixed(2) + ')';
                }

                // Station Background & Border
                ctx.fillStyle = cardBg;
                ctx.strokeStyle = isHover ? PALETTE.gold : borderColor;
                ctx.lineWidth = (isHover || st.glowForward > 0.3 || st.glowBackward > 0.3) ? 1.5 : 1.0;

                ctx.beginPath();
                ctx.rect(st.x, st.y, st.w, st.h);
                ctx.fill();
                ctx.stroke();

                // Corner Technical Brackets
                var crLen = 4;
                ctx.strokeStyle = isHover ? PALETTE.gold : PALETTE.ruleStrong;
                ctx.lineWidth = 1;
                ctx.beginPath();
                // Top-Left
                ctx.moveTo(st.x, st.y + crLen); ctx.lineTo(st.x, st.y); ctx.lineTo(st.x + crLen, st.y);
                // Top-Right
                ctx.moveTo(st.x + st.w - crLen, st.y); ctx.lineTo(st.x + st.w, st.y); ctx.lineTo(st.x + st.w, st.y + crLen);
                // Bottom-Left
                ctx.moveTo(st.x, st.y + st.h - crLen); ctx.lineTo(st.x, st.y + st.h); ctx.lineTo(st.x + crLen, st.y + st.h);
                // Bottom-Right
                ctx.moveTo(st.x + st.w - crLen, st.y + st.h); ctx.lineTo(st.x + st.w, st.y + st.h); ctx.lineTo(st.x + st.w, st.y + st.h - crLen);
                ctx.stroke();

                // Status Beacon LED Dot
                var ledColor = PALETTE.inkFaint;
                if (st.glowAdam > 0.2) ledColor = PALETTE.gold;
                else if (st.glowForward > 0.2) ledColor = PALETTE.terracotta;
                else if (st.glowBackward > 0.2) ledColor = PALETTE.olive;

                ctx.beginPath();
                ctx.arc(st.x + 8, st.y + 10, 2.5, 0, Math.PI * 2);
                ctx.fillStyle = ledColor;
                ctx.fill();

                // Responsive title & badge for compact screens
                var isCompact = st.w < 70;
                var displayBadge = isCompact ? '0' + st.stage.id : st.stage.badge;
                var displayTitle = isCompact ? (st.stage.id === 1 ? 'Embed' : st.stage.id === 2 ? 'MLA' : st.stage.id === 3 ? 'Dense' : st.stage.id === 4 ? 'MoE' : st.stage.id === 5 ? 'Triton' : 'MTP') : st.stage.title;

                // Stage Number Badge
                ctx.font = isCompact ? 'bold 7.5px "JetBrains Mono", monospace' : 'bold 8.5px "JetBrains Mono", monospace';
                ctx.fillStyle = (st.glowForward > 0.2 ? PALETTE.terracotta : (st.glowBackward > 0.2 ? PALETTE.olive : PALETTE.inkSoft));
                ctx.textAlign = 'left';
                ctx.fillText(displayBadge, st.x + (isCompact ? 13 : 14), st.y + 13);

                // Stage Title
                ctx.font = isCompact ? 'bold 9px "JetBrains Mono", monospace' : 'bold 10px "JetBrains Mono", monospace';
                ctx.fillStyle = isHover ? '#ffffff' : PALETTE.ink;
                ctx.textAlign = 'center';
                ctx.fillText(displayTitle, st.cx, st.y + st.h * 0.40);

                // Subtitle
                if (st.h > 80) {
                    ctx.font = isCompact ? '7.5px "JetBrains Mono", monospace' : '8.5px "JetBrains Mono", monospace';
                    ctx.fillStyle = PALETTE.gold;
                    ctx.fillText(st.stage.sub, st.cx, st.y + st.h * 0.60);
                }

                // Chip Tag
                if (st.h > 100 && !isCompact) {
                    ctx.font = '8px "JetBrains Mono", monospace';
                    ctx.fillStyle = PALETTE.inkSoft;
                    ctx.fillText(st.stage.chip, st.cx, st.y + st.h * 0.78);
                }

                // Re-compute / Checkpoint Tag during Backward
                if (st.glowBackward > 0.3 && (idx >= 1 && idx <= 4)) {
                    ctx.fillStyle = PALETTE.olive;
                    ctx.font = 'bold 7.5px "JetBrains Mono", monospace';
                    ctx.fillText('RE-COMPUTE', st.cx, st.y + st.h - 6);
                } else if (st.glowAdam > 0.3) {
                    ctx.fillStyle = PALETTE.gold;
                    ctx.font = 'bold 7.5px "JetBrains Mono", monospace';
                    ctx.fillText('θ UPDATE', st.cx, st.y + st.h - 6);
                }

                // Inter-stage dataflow bridge connector
                if (idx < numStations - 1) {
                    var nextSt = stations[idx + 1];
                    var gapStartX = st.x + st.w;
                    var gapEndX = nextSt.x;
                    var midY = st.cy;

                    ctx.save();
                    ctx.beginPath();
                    ctx.moveTo(gapStartX, midY);
                    ctx.lineTo(gapEndX, midY);
                    ctx.strokeStyle = (st.glowForward > 0.2 || nextSt.glowForward > 0.2) ? 'rgba(224, 122, 63, 0.7)' :
                                      (st.glowBackward > 0.2 || nextSt.glowBackward > 0.2) ? 'rgba(154, 148, 64, 0.7)' :
                                      PALETTE.ruleStrong;
                    ctx.lineWidth = (st.glowForward > 0.2 || nextSt.glowForward > 0.2 || st.glowBackward > 0.2 || nextSt.glowBackward > 0.2) ? 1.5 : 1.0;
                    ctx.stroke();

                    // Small directional chevron arrow
                    var arrowX = (gapStartX + gapEndX) / 2;
                    ctx.beginPath();
                    if (state.phase === 'backward') {
                        ctx.moveTo(arrowX + 2.5, midY - 2.5);
                        ctx.lineTo(arrowX - 2.5, midY);
                        ctx.lineTo(arrowX + 2.5, midY + 2.5);
                        ctx.strokeStyle = PALETTE.olive;
                    } else {
                        ctx.moveTo(arrowX - 2.5, midY - 2.5);
                        ctx.lineTo(arrowX + 2.5, midY);
                        ctx.lineTo(arrowX - 2.5, midY + 2.5);
                        ctx.strokeStyle = (st.glowForward > 0.2) ? PALETTE.terracotta : PALETTE.rule;
                    }
                    ctx.lineWidth = 1;
                    ctx.stroke();
                    ctx.restore();
                }
            });

            hoveredStation = closest;
            ctx.restore();
        }

        function updateHUD(state) {
            if (phaseEl) {
                var phaseName = 'FORWARD ACTIVATIONS (1F1B DUALPIPE)';
                if (state.phase === 'loss') phaseName = 'CHUNKED CE LOSS & MTP (ℓ)';
                else if (state.phase === 'backward') phaseName = 'AUTOGRAD BACKWARD (RECOMPUTE)';
                else if (state.phase === 'adam') phaseName = 'μP ADAMW WEIGHT UPDATE (FP32)';
                else if (state.phase === 'rest') phaseName = 'STANDBY / NEXT MINI-BATCH';
                phaseEl.textContent = phaseName;
            }

            if (tickerText) {
                tickerText.textContent = state.subText;
            }

            if (tickerBeacon) {
                tickerBeacon.style.color = (state.phase === 'forward' ? PALETTE.terracotta : (state.phase === 'backward' ? PALETTE.olive : PALETTE.gold));
            }

            if (hoveredStation && tooltipEl) {
                tooltipEl.style.opacity = '1';
                var st = hoveredStation.stage;
                var targetX = hoveredStation.x;
                var targetY = hoveredStation.y;

                var cardW = 320;
                var cardH = 80;

                var leftPx = targetX + hoveredStation.w / 2 - cardW / 2;
                var topPx = targetY - cardH - 12;
                if (topPx < 8) topPx = targetY + hoveredStation.h + 12;

                leftPx = Math.max(12, Math.min(width - cardW - 12, leftPx));

                tooltipEl.style.left = Math.round(leftPx) + 'px';
                tooltipEl.style.top = Math.round(topPx) + 'px';

                if (stTag) stTag.textContent = st.tag;
                if (stOp) stOp.textContent = st.op;
                if (stShape) stShape.textContent = st.shape;
                if (stDesc) stDesc.textContent = st.desc;
            } else if (tooltipEl) {
                tooltipEl.style.opacity = '0';
            }
        }

        function render(dt, force) {
            if (!force && isPaused) return;

            simTime += dt;
            ctx.clearRect(0, 0, width, height);

            var state = getCycleState(simTime);

            drawBackground();
            drawParticles(state, dt);
            drawStations(state, dt);
            updateHUD(state);
        }

        function loop(timestamp) {
            if (!lastTime) lastTime = timestamp;
            var dt = Math.min((timestamp - lastTime) / 1000, 0.1);
            lastTime = timestamp;

            render(dt, false);

            if (!isPaused && !reduced) {
                animId = requestAnimationFrame(loop);
            }
        }

        if (reduced) {
            render(0, true);
        } else {
            lastTime = performance.now();
            animId = requestAnimationFrame(loop);

            document.addEventListener('visibilitychange', function () {
                if (document.hidden) {
                    if (animId) {
                        cancelAnimationFrame(animId);
                        animId = null;
                    }
                } else if (!isPaused && !animId) {
                    lastTime = performance.now();
                    animId = requestAnimationFrame(loop);
                }
            });
        }
    }

    // ------------------------------------------------------------------
    // 4. 18-Layer Stack Interactive Inspector (FIG · 01)
    // ------------------------------------------------------------------
    function initLayerStack() {
        var box = document.getElementById('widget-layer-stack');
        if (!box) return;
        var stackPre = box.querySelector('pre.ascii-stack');
        var panelPre = box.querySelector('pre.ascii-panel');
        if (!stackPre || !panelPre) return;

        var DETAILS = {
            dense: [
                "STAGE       : Layers 00 - 01 (2 Dense warmup blocks)",
                "ATTENTION   : MLA (kv_lora_rank=192, qk_rope=24)",
                "FFN         : SwiGLU Dense (1,536 intermediate dim)",
                "PURPOSE     : Representation warmup before expert routing",
                "SOURCE      : models/transformer.py · configs/pretrain_a100_422m.yaml"
            ].join('\n'),
            moe: [
                "STAGE       : Layers 02 - 17 (16 MoE blocks)",
                "ATTENTION   : MLA (kv_lora_rank=192, qk_rope=24)",
                "FFN ROUTING : DeepSeekMoE (20 routed top-4 + 1 shared)",
                "INTER_DIM   : 384 / expert (active: 5 x 384 = 1,920)",
                "ROUTING BIAS: Aux-loss-free gating with dynamic delta_b",
                "SOURCE      : models/moe.py · configs/pretrain_a100_422m.yaml"
            ].join('\n'),
            total: [
                "STAGE       : Full 18-Layer DeepSeek-v3-Lite Stack",
                "DIM / HEADS : d_model=768, n_heads=12 (head_dim=128)",
                "PARAMS      : 411.6M total / 247M active per token",
                "SCALING     : μP (Maximal Update Parameterization)",
                "SOURCE      : models/transformer.py · training/pretrain.py"
            ].join('\n')
        };

        // Wrap lines of stack in interactive spans if not already wrapped
        var text = stackPre.innerText || stackPre.textContent;
        var lines = text.split('\n');
        var newLines = [];
        lines.forEach(function (line) {
            if (line.indexOf('L02-L17') !== -1) {
                newLines.push('<span class="stack-row active" data-stage="moe" tabindex="0">' + line + '</span>');
            } else if (line.indexOf('L00-L01') !== -1) {
                newLines.push('<span class="stack-row" data-stage="dense" tabindex="0">' + line + '</span>');
            } else if (line.indexOf('TOTAL 18 L') !== -1) {
                newLines.push('<span class="stack-row" data-stage="total" tabindex="0">' + line + '</span>');
            } else {
                newLines.push(line);
            }
        });
        stackPre.innerHTML = newLines.join('\n');

        function selectStage(span) {
            var stage = span.getAttribute('data-stage');
            if (!stage || !DETAILS[stage]) return;
            stackPre.querySelectorAll('.stack-row').forEach(function (r) { r.classList.remove('active'); });
            span.classList.add('active');
            panelPre.textContent = DETAILS[stage];
        }

        stackPre.querySelectorAll('.stack-row').forEach(function (span) {
            span.addEventListener('mouseenter', function () { selectStage(span); });
            span.addEventListener('click', function () { selectStage(span); });
            span.addEventListener('keydown', function (e) {
                if (e.key === 'Enter' || e.key === ' ') {
                    e.preventDefault();
                    selectStage(span);
                }
            });
        });
    }

    // ------------------------------------------------------------------
    // 5. MoE Routing Playground (FIG / GATE) for Concepts Page
    // ------------------------------------------------------------------
    function initMoePlayground() {
        var box = document.getElementById('widget-moe-routing');
        if (!box) return;
        var pre = box.querySelector('pre.moe-grid');
        if (!pre) return;

        var controls = box.querySelector('.moe-controls');
        if (!controls) {
            controls = document.createElement('div');
            controls.className = 'moe-controls';
            controls.innerHTML = '<button class="moe-fire" type="button">▸ Route Token</button><div class="moe-status">Active experts: 02, 07, 13, 18 + shared</div>';
            box.appendChild(controls);
        }

        var btn = controls.querySelector('.moe-fire');
        var status = controls.querySelector('.moe-status');

        if (btn && !btn._hasMoeListener) {
            btn._hasMoeListener = true;
            btn.addEventListener('click', function () {
                var indices = [];
                while (indices.length < 4) {
                    var r = Math.floor(Math.random() * 20);
                    if (indices.indexOf(r) === -1) indices.push(r);
                }
                indices.sort(function (a, b) { return a - b; });

                var lines = ["gate(x) ▸ top-4 of 20 routed:"];
                for (var row = 0; row < 4; row++) {
                    var rline = [];
                    for (var col = 0; col < 5; col++) {
                        var idx = row * 5 + col;
                        var pad = (idx < 10 ? '0' : '') + idx;
                        if (indices.indexOf(idx) !== -1) {
                            rline.push('▓▓' + pad);
                        } else {
                            rline.push('░░' + pad);
                        }
                    }
                    lines.push(rline.join(' '));
                }
                lines.push('shared  ▓▓sh (always on)');
                pre.textContent = lines.join('\n');
                if (status) {
                    status.textContent = 'Active experts: ' + indices.map(function(i){ return (i<10?'0':'')+i; }).join(', ') + ' + shared';
                }
            });
        }
    }

    // ------------------------------------------------------------------
    // 6. MLA Latent Absorption Toggle (FIG / MLA) for Concepts Page
    // ------------------------------------------------------------------
    function initMlaToggle() {
        var box = document.getElementById('widget-mla-absorb');
        if (!box) return;
        var pre = box.querySelector('pre.mla-figure');
        if (!pre) return;

        var controls = box.querySelector('.mla-toggle-bar');
        if (!controls) {
            controls = document.createElement('div');
            controls.className = 'mla-toggle-bar';
            controls.innerHTML = [
                '<button type="button" data-mode="absorbed" aria-pressed="true">Absorbed (Inference)</button>',
                '<button type="button" data-mode="standard" aria-pressed="false">Standard (Training SDPA)</button>'
            ].join(' ');
            box.appendChild(controls);
        }

        var VIEWS = {
            absorbed: [
                "absorbed attention (inference mode: latent-compressed)",
                "",
                " h[768] ─ W_D ─► c [192] (+ rope 24) ─► cached in KV",
                " c ─► W_U · W_q absorbed into q′ ─► attn(q′, c) ─► out",
                "",
                " KV cache / token : 192 + 24 = 216 dims",
                " Compression      : 1,536 ► 216 (≈ 7.1× memory reduction)"
            ].join('\n'),
            standard: [
                "standard attention (training mode: materialised KV)",
                "",
                " h[768] ─► W_q · h [12×128 = 1,536] (Q_nope + Q_rope)",
                " h[768] ─► W_kv_b · c [12×128 = 1,536] (K_nope + V)",
                "",
                " Memory / token   : 12 × 128 = 1,536 dims uncompressed",
                " Advantage        : standard SDPA flash-attention support"
            ].join('\n')
        };

        controls.querySelectorAll('button').forEach(function (btn) {
            if (btn._hasMlaListener) return;
            btn._hasMlaListener = true;
            btn.addEventListener('click', function () {
                var mode = btn.getAttribute('data-mode');
                if (VIEWS[mode]) pre.textContent = VIEWS[mode];
                controls.querySelectorAll('button').forEach(function (b) {
                    b.setAttribute('aria-pressed', b === btn ? 'true' : 'false');
                });
            });
        });
    }

    // ------------------------------------------------------------------
    // 7. DualPipe Bidirectional Pipeline Step-through (FIG / DUALPIPE)
    // ------------------------------------------------------------------
    function initDualPipeWidget() {
        var box = document.getElementById('widget-dualpipe');
        if (!box) return;
        var pre = box.querySelector('pre.dualpipe-figure');
        if (!pre) return;

        var controls = box.querySelector('.dualpipe-controls');
        if (!controls) {
            controls = document.createElement('div');
            controls.className = 'dualpipe-controls';
            controls.innerHTML = '<button class="dualpipe-step" type="button">▸ Step Microbatch</button><div class="dualpipe-status">Microbatch: F0 / B3 active on Stage 0</div>';
            box.appendChild(controls);
        }

        var btn = controls.querySelector('.dualpipe-step');
        var status = controls.querySelector('.dualpipe-status');
        var mb = 0;

        if (btn && !btn._hasDpListener) {
            btn._hasDpListener = true;
            btn.addEventListener('click', function () {
                mb = (mb + 1) % 4;
                var fwd = ["[ F0 ]", "[ F1 ]", "[ F2 ]", "[ F3 ]"];
                var bwd = ["[ B3 ]", "[ B2 ]", "[ B1 ]", "[ B0 ]"];
                fwd[mb] = "▓ F" + mb + " ▓";
                bwd[mb] = "▓ B" + (3 - mb) + " ▓";

                var lines = [
                    "DualPipe Bidirectional Pipeline Schedule (8 PP Stages)",
                    "",
                    " Forward Chunk (1F1B)  : " + fwd.join(" ──► "),
                    " Backward Chunk (1F1B) : " + bwd.join(" ◄── "),
                    " Comm Overlap Ratio    : ~100% dispatch/gather overlap with computation"
                ];
                pre.textContent = lines.join('\n');
                if (status) {
                    status.textContent = 'Microbatch: F' + mb + ' / B' + (3 - mb) + ' active in pipeline stage';
                }
            });
        }
    }

    // ------------------------------------------------------------------
    // 8. MTP Speculative Tree Verification (FIG / MTP) for Inference Page
    // ------------------------------------------------------------------
    function initMtpWidget() {
        var box = document.getElementById('widget-mtp-tree');
        if (!box) return;
        var pre = box.querySelector('pre.mtp-figure');
        if (!pre) return;

        var controls = box.querySelector('.mtp-controls');
        if (!controls) {
            controls = document.createElement('div');
            controls.className = 'mtp-controls';
            controls.innerHTML = '<button class="mtp-step" type="button">▸ Verify Speculative Step</button><div class="mtp-status">Status: Token t verified (1.0), Token t+1 accepted (0.84)</div>';
            box.appendChild(controls);
        }

        var btn = controls.querySelector('.mtp-step');
        var status = controls.querySelector('.mtp-status');

        if (btn && !btn._hasMtpListener) {
            btn._hasMtpListener = true;
            btn.addEventListener('click', function () {
                var acc = (0.78 + Math.random() * 0.16).toFixed(2);
                var isAcc = parseFloat(acc) >= 0.80;
                var mark = isAcc ? "ACCEPTED ✓" : "REJECTED (fall back to draft)";

                var lines = [
                    "MTP Multi-Token Speculative Prediction (Depth = 1)",
                    "",
                    " Main Model (Draft) : [ tok_t ] ────► p(x_{t} | x_{<t})       (accepted: 1.00)",
                    " MTP Head (Draft+1) : [ tok_{t+1} ] ──► p(x_{t+1} | x_{<t+1})   (" + mark + ": " + acc + ")",
                    "",
                    " Generation Speedup : " + (isAcc ? "2.0× tokens/step" : "1.0× tokens/step (fallback)")
                ];
                pre.textContent = lines.join('\n');
                if (status) {
                    status.textContent = 'Status: Token t+1 ' + mark + ' (p = ' + acc + ')';
                }
            });
        }
    }

    // ------------------------------------------------------------------
    // 9. Expandable Code Blocks (>14 lines) & Copy-to-Clipboard
    // ------------------------------------------------------------------
    function fallbackCopy(text) {
        var ta = document.createElement('textarea');
        ta.value = text;
        ta.style.position = 'fixed';
        ta.style.top = '-9999px';
        ta.style.left = '-9999px';
        document.body.appendChild(ta);
        ta.select();
        try { document.execCommand('copy'); } catch (e) {}
        document.body.removeChild(ta);
    }

    window.toggleCode = function (btn) {
        var wrapper = btn.closest('.code-wrapper');
        if (!wrapper) return;
        var collapsed = wrapper.classList.toggle('collapsed');
        var nLines = wrapper.getAttribute('data-lines') || '';
        if (!nLines) {
            var codeEl = wrapper.querySelector('code');
            if (codeEl) {
                var lines = codeEl.innerText.split('\n').length;
                nLines = String(lines);
            }
        }
        btn.textContent = collapsed ? ('expand \u25be \u00b7 ' + nLines + ' lines') : 'collapse \u25b4';
    };

    window.copyCode = function (btn) {
        var wrapper = btn.closest('.code-wrapper');
        if (!wrapper) return;
        var code = wrapper.querySelector('pre code') || wrapper.querySelector('code');
        if (!code) return;
        var text = code.innerText || code.textContent;

        function flash() {
            var orig = btn.textContent;
            btn.textContent = 'Copied!';
            btn.classList.add('copied');
            setTimeout(function () {
                btn.textContent = orig;
                btn.classList.remove('copied');
            }, 1800);
        }

        if (navigator.clipboard && navigator.clipboard.writeText) {
            navigator.clipboard.writeText(text).then(flash).catch(function () {
                fallbackCopy(text);
                flash();
            });
        } else {
            fallbackCopy(text);
            flash();
        }
    };

    // ------------------------------------------------------------------
    // 10. Navigation Filtering & Sidebar Mobile Toggle
    // ------------------------------------------------------------------
    window.filterNav = function () {
        var input = document.getElementById('navSearch');
        if (!input) return;
        var q = (input.value || '').toLowerCase().trim();
        var links = document.querySelectorAll('.nav-link');
        var groupHeaders = document.querySelectorAll('.nav-group');

        links.forEach(function (l) {
            var txt = l.textContent.toLowerCase();
            var item = l.closest('.nav-item');
            if (!item) return;
            item.style.display = (!q || txt.indexOf(q) !== -1) ? '' : 'none';
        });

        groupHeaders.forEach(function (grp) {
            var visibleItems = grp.querySelectorAll('.nav-item:not([style*="display: none"])');
            grp.style.display = (!q || visibleItems.length > 0) ? '' : 'none';
        });
    };

    window.toggleSidebar = function () {
        var sb = document.getElementById('sidebar');
        if (sb) sb.classList.toggle('open');
    };

    // ------------------------------------------------------------------
    // 11. Table of Contents Scrollspy
    // ------------------------------------------------------------------
    function initScrollspy() {
        var tocSidebar = document.querySelector('.toc-sidebar');
        var tocLinks = Array.from(document.querySelectorAll('.toc-link'));
        if (!tocLinks.length) return;

        var headings = [];
        tocLinks.forEach(function (link) {
            var href = link.getAttribute('href');
            if (href && href.startsWith('#')) {
                var el = document.getElementById(href.substring(1));
                if (el) headings.push({ el: el, link: link });
            }
        });

        if (!headings.length) return;

        var ticking = false;
        function updateSpy() {
            var scrollPos = window.scrollY + 140;
            var current = null;

            for (var i = 0; i < headings.length; i++) {
                if (headings[i].el.offsetTop <= scrollPos) {
                    current = headings[i];
                } else {
                    break;
                }
            }

            tocLinks.forEach(function (l) { l.classList.remove('active'); });
            if (current) {
                current.link.classList.add('active');
                if (tocSidebar && window.innerWidth >= 1280) {
                    var linkTop = current.link.offsetTop;
                    var sideScroll = tocSidebar.scrollTop;
                    var sideHeight = tocSidebar.clientHeight;
                    if (linkTop < sideScroll + 40 || linkTop > sideScroll + sideHeight - 60) {
                        tocSidebar.scrollTo({ top: linkTop - 80, behavior: 'smooth' });
                    }
                }
            }
            ticking = false;
        }

        window.addEventListener('scroll', function () {
            if (!ticking) {
                window.requestAnimationFrame(updateSpy);
                ticking = true;
            }
        }, { passive: true });

        updateSpy();
    }

    // ------------------------------------------------------------------
    // Boot Initialization
    // ------------------------------------------------------------------
    document.addEventListener('DOMContentLoaded', function () {
        initMlaHero();
        initMechanismExploders();
        initPassDiagram();
        initLayerStack();
        initMoePlayground();
        initMlaToggle();
        initDualPipeWidget();
        initMtpWidget();
        initScrollspy();
    });
})();
