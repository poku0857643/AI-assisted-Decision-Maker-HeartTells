// HeartTells chat — streaming conversation with a living heartbeat.
(() => {
    const STORAGE_KEY = 'hearttells_chat_v1';
    const ENDPOINT = '/api/v1/chat/stream';
    const reducedMotion = window.matchMedia('(prefers-reduced-motion: reduce)').matches;

    const els = {
        body: document.body,
        messages: document.getElementById('messages'),
        suggestions: document.getElementById('suggestions'),
        form: document.getElementById('composer'),
        input: document.getElementById('input'),
        send: document.getElementById('send-btn'),
        clear: document.getElementById('clear-btn'),
        statusText: document.getElementById('status-text'),
        bpm: document.getElementById('bpm-value'),
        heart: document.querySelector('.heart'),
        ripple: document.querySelector('.heart-ripple'),
        ecg: document.getElementById('ecg-trace'),
    };

    let history = loadHistory();
    let busy = false;

    // ------------------------------------------------------------------
    // Heartbeat engine: BPM drifts with what's happening in the chat
    // ------------------------------------------------------------------
    const heart = {
        bpm: 72,
        base: 70,
        activity: 0,      // added while typing / waiting / streaming
        moodOffset: 0,    // added from detected mood
        lastBeat: performance.now(),

        target() {
            const breath = Math.sin(performance.now() / 4000) * 2; // respiratory sinus arrhythmia
            return this.base + this.activity + this.moodOffset + breath;
        },
        setActivity(v) { this.activity = v; },
        setMood(mood) {
            this.moodOffset = { anxious: 16, excited: 12, sad: -4, calm: -7 }[mood] || 0;
            els.body.dataset.mood = mood;
        },
    };

    function beat() {
        if (reducedMotion) return;
        for (const el of [els.heart, els.ripple]) {
            el.classList.remove('beat');
            void el.offsetWidth; // restart animation
            el.classList.add('beat');
        }
        ecg.spike();
    }

    function heartLoop(now) {
        heart.bpm += (heart.target() - heart.bpm) * 0.02;
        const interval = 60000 / heart.bpm;
        if (now - heart.lastBeat >= interval) {
            heart.lastBeat = now;
            beat();
            els.bpm.textContent = Math.round(heart.bpm);
            document.documentElement.style.setProperty('--beat', `${(interval / 1000).toFixed(3)}s`);
        }
        ecg.draw();
        requestAnimationFrame(heartLoop);
    }

    // Tiny scrolling ECG trace in the header
    const ecg = (() => {
        const ctx = els.ecg.getContext('2d');
        const dpr = window.devicePixelRatio || 1;
        const w = 160, h = 40;
        els.ecg.width = w * dpr;
        els.ecg.height = h * dpr;
        ctx.scale(dpr, dpr);
        const samples = new Array(w).fill(0);
        // P, Q, R, S, T wave shape
        const shape = [0, 0.08, 0.12, 0.08, 0, 0, -0.1, 0.95, -0.35, 0, 0, 0.05, 0.14, 0.18, 0.14, 0.05, 0];
        let queue = [];
        let acc = 0;
        let last = performance.now();

        return {
            spike() { queue = shape.slice(); },
            draw() {
                const now = performance.now();
                acc += (now - last) / 16;
                last = now;
                while (acc >= 1) {
                    acc -= 1;
                    samples.shift();
                    samples.push((queue.length ? queue.shift() : 0) + (Math.random() - 0.5) * 0.02);
                }
                ctx.clearRect(0, 0, w, h);
                const color = getComputedStyle(els.heart).fill || '#ef4f6b';
                const grad = ctx.createLinearGradient(0, 0, w, 0);
                grad.addColorStop(0, 'transparent');
                grad.addColorStop(0.7, color);
                grad.addColorStop(1, color);
                ctx.strokeStyle = grad;
                ctx.lineWidth = 1.6;
                ctx.lineJoin = 'round';
                ctx.beginPath();
                samples.forEach((v, x) => {
                    const y = h * 0.62 - v * h * 0.55;
                    x === 0 ? ctx.moveTo(x, y) : ctx.lineTo(x, y);
                });
                ctx.stroke();
                // glowing head of the trace
                const headY = h * 0.62 - samples[w - 1] * h * 0.55;
                ctx.fillStyle = color;
                ctx.beginPath();
                ctx.arc(w - 2, headY, 2, 0, Math.PI * 2);
                ctx.fill();
            },
        };
    })();

    // ------------------------------------------------------------------
    // Status line
    // ------------------------------------------------------------------
    let statusTimer;
    function setStatus(text) {
        if (els.statusText.textContent === text) return;
        clearTimeout(statusTimer);
        els.statusText.classList.add('fading');
        statusTimer = setTimeout(() => {
            els.statusText.textContent = text;
            els.statusText.classList.remove('fading');
        }, 200);
    }

    const idleLines = ['Here with you', 'Listening for your heart', 'Take your time', 'No rush'];
    let idleIndex = 0;
    setInterval(() => {
        if (busy || document.activeElement === els.input && els.input.value) return;
        idleIndex = (idleIndex + 1) % idleLines.length;
        setStatus(idleLines[idleIndex]);
    }, 9000);

    // ------------------------------------------------------------------
    // Rendering
    // ------------------------------------------------------------------
    function relativeTime(ts) {
        const s = Math.round((Date.now() - ts) / 1000);
        if (s < 45) return 'just now';
        const m = Math.round(s / 60);
        if (m < 60) return `${m} min ago`;
        const h = Math.round(m / 60);
        if (h < 24) return `${h} hr ago`;
        return new Date(ts).toLocaleDateString();
    }

    function refreshTimes() {
        els.messages.querySelectorAll('.meta[data-ts]').forEach((el) => {
            el.textContent = relativeTime(Number(el.dataset.ts));
        });
    }
    setInterval(refreshTimes, 30000);

    function nearBottom() {
        const m = els.messages;
        return m.scrollHeight - m.scrollTop - m.clientHeight < 120;
    }

    function scrollToBottom(force = false) {
        if (force || nearBottom()) els.messages.scrollTop = els.messages.scrollHeight;
    }

    function renderWelcome() {
        const div = document.createElement('div');
        div.className = 'welcome';
        div.innerHTML = `
            <div class="big-heart">💗</div>
            <h2>What's your heart telling you?</h2>
            <p>Tell me about a choice you're weighing — like "should I take the new job or stay?" — and we'll listen to both your mind and your heart.</p>`;
        els.messages.appendChild(div);
    }

    function addMessage(entry, { animate = true } = {}) {
        els.messages.querySelector('.welcome')?.remove();
        const prev = els.messages.lastElementChild;
        const msg = document.createElement('div');
        msg.className = `msg ${entry.role === 'user' ? 'user' : 'bot'}`;
        if (prev?.classList.contains(entry.role === 'user' ? 'user' : 'bot')) msg.classList.add('same-sender');
        if (!animate) msg.style.animation = 'none';

        const bubble = document.createElement('div');
        bubble.className = 'bubble';
        bubble.textContent = entry.content;
        msg.appendChild(bubble);

        const meta = document.createElement('div');
        meta.className = 'meta';
        meta.dataset.ts = entry.ts;
        meta.textContent = relativeTime(entry.ts);
        msg.appendChild(meta);

        if (entry.role === 'assistant') msg.appendChild(buildReactions(entry));

        els.messages.appendChild(msg);
        scrollToBottom(true);
        return { msg, bubble };
    }

    function buildReactions(entry) {
        const wrap = document.createElement('div');
        wrap.className = 'reactions';
        ['💗', '🤔', '🙏'].forEach((emoji) => {
            const btn = document.createElement('button');
            btn.type = 'button';
            btn.className = 'react-btn';
            btn.textContent = emoji;
            btn.setAttribute('aria-label', `React ${emoji}`);
            if (entry.reaction === emoji) { btn.classList.add('picked'); wrap.classList.add('has-picked'); }
            btn.addEventListener('click', () => {
                const picked = entry.reaction === emoji ? null : emoji;
                entry.reaction = picked;
                wrap.querySelectorAll('.react-btn').forEach((b) => b.classList.remove('picked'));
                wrap.classList.toggle('has-picked', !!picked);
                if (picked) {
                    btn.classList.add('picked');
                    if (picked === '💗') beat();
                }
                saveHistory();
            });
            wrap.appendChild(btn);
        });
        return wrap;
    }

    function addTyping() {
        const msg = document.createElement('div');
        msg.className = 'msg bot typing';
        msg.innerHTML = '<div class="bubble"><span class="dot"></span><span class="dot"></span><span class="dot"></span></div>';
        els.messages.appendChild(msg);
        scrollToBottom(true);
        return msg;
    }

    function addChoiceCard([a, b]) {
        const card = document.createElement('div');
        card.className = 'choice-card';
        const make = (label, text) => {
            const btn = document.createElement('button');
            btn.type = 'button';
            btn.className = 'choice';
            btn.innerHTML = `<small>${label}</small><span></span>`;
            btn.querySelector('span').textContent = text;
            btn.addEventListener('click', () => send(`Imagine I chose ${text}`));
            return btn;
        };
        const vs = document.createElement('div');
        vs.className = 'vs';
        vs.textContent = 'or';
        card.append(make('Path A', a), vs, make('Path B', b));
        els.messages.appendChild(card);
        scrollToBottom(true);
    }

    function showSuggestions(list) {
        clearSuggestions(true);
        list.forEach((text, i) => {
            const chip = document.createElement('button');
            chip.type = 'button';
            chip.className = 'chip';
            chip.textContent = text;
            chip.style.animationDelay = `${i * 70}ms`;
            chip.addEventListener('click', () => send(text));
            els.suggestions.appendChild(chip);
        });
    }

    function clearSuggestions(instant = false) {
        const chips = [...els.suggestions.children];
        if (instant) { chips.forEach((c) => c.remove()); return; }
        chips.forEach((c) => { c.classList.add('leaving'); setTimeout(() => c.remove(), 200); });
    }

    // ------------------------------------------------------------------
    // Sending + streaming
    // ------------------------------------------------------------------
    async function send(text) {
        text = text.trim();
        if (!text || busy) return;
        busy = true;
        updateSendState();
        clearSuggestions();

        const userEntry = { role: 'user', content: text, ts: Date.now() };
        const priorHistory = history.map(({ role, content }) => ({ role, content }));
        history.push(userEntry);
        addMessage(userEntry);
        saveHistory();

        els.input.value = '';
        autoGrow();
        els.send.classList.remove('sent'); void els.send.offsetWidth; els.send.classList.add('sent');

        setStatus('Feeling into it…');
        heart.setActivity(10);
        const typing = addTyping();

        const botEntry = { role: 'assistant', content: '', ts: Date.now() };
        let view = null;

        try {
            const res = await fetch(ENDPOINT, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ message: text, history: priorHistory.slice(-20) }),
            });
            if (!res.ok || !res.body) throw new Error(`HTTP ${res.status}`);

            const reader = res.body.getReader();
            const decoder = new TextDecoder();
            let buffer = '';

            while (true) {
                const { value, done } = await reader.read();
                if (done) break;
                buffer += decoder.decode(value, { stream: true });
                let idx;
                while ((idx = buffer.indexOf('\n\n')) !== -1) {
                    const raw = buffer.slice(0, idx);
                    buffer = buffer.slice(idx + 2);
                    const line = raw.split('\n').find((l) => l.startsWith('data:'));
                    if (!line) continue;
                    const event = JSON.parse(line.slice(5));

                    if (event.type === 'mood') {
                        heart.setMood(event.mood);
                    } else if (event.type === 'choices') {
                        typing.remove();
                        addChoiceCard(event.choices);
                        els.messages.appendChild(typing);
                    } else if (event.type === 'token') {
                        if (!view) {
                            typing.remove();
                            view = addMessage(botEntry);
                            view.bubble.textContent = '';
                            view.msg.classList.add('streaming');
                            setStatus('Speaking…');
                            heart.setActivity(5);
                        }
                        botEntry.content += event.text;
                        const span = document.createElement('span');
                        span.className = 'word';
                        span.textContent = event.text;
                        view.bubble.appendChild(span);
                        scrollToBottom();
                    } else if (event.type === 'suggestions') {
                        showSuggestions(event.suggestions);
                    }
                }
            }
        } catch (err) {
            console.error('Chat error:', err);
            typing.remove();
            if (!view) {
                botEntry.content = "I lost the thread for a second — my connection hiccuped. Could you say that again?";
                view = addMessage(botEntry);
            }
        } finally {
            typing.remove();
            if (view) {
                view.msg.classList.remove('streaming');
                // collapse word spans back to plain text once finished
                view.bubble.textContent = botEntry.content;
                history.push(botEntry);
                saveHistory();
            }
            busy = false;
            heart.setActivity(0);
            setStatus('Here with you');
            updateSendState();
            els.input.focus();
        }
    }

    // ------------------------------------------------------------------
    // Composer
    // ------------------------------------------------------------------
    function autoGrow() {
        els.input.style.height = 'auto';
        els.input.style.height = `${Math.min(els.input.scrollHeight, 160)}px`;
    }

    function updateSendState() {
        els.send.disabled = busy || !els.input.value.trim();
    }

    let typingIdle;
    els.input.addEventListener('input', () => {
        autoGrow();
        updateSendState();
        if (busy) return;
        // your heart quickens a little while you're putting it into words
        heart.setActivity(6);
        setStatus('Listening…');
        clearTimeout(typingIdle);
        typingIdle = setTimeout(() => {
            if (!busy) { heart.setActivity(0); setStatus('Here with you'); }
        }, 1800);
    });

    els.input.addEventListener('keydown', (e) => {
        if (e.key === 'Enter' && !e.shiftKey && !e.isComposing) {
            e.preventDefault();
            send(els.input.value);
        }
    });

    els.form.addEventListener('submit', (e) => {
        e.preventDefault();
        send(els.input.value);
    });

    els.clear.addEventListener('click', () => {
        if (busy) return;
        history = [];
        saveHistory();
        els.messages.innerHTML = '';
        heart.setMood('neutral');
        renderWelcome();
        showSuggestions(['I have two options', "I'm not sure how I feel", 'Check my heart']);
        els.input.focus();
    });

    // ------------------------------------------------------------------
    // Persistence (per-browser convenience only)
    // ------------------------------------------------------------------
    function loadHistory() {
        try { return JSON.parse(localStorage.getItem(STORAGE_KEY)) || []; } catch { return []; }
    }
    function saveHistory() {
        try { localStorage.setItem(STORAGE_KEY, JSON.stringify(history.slice(-100))); } catch { /* ignore */ }
    }

    // ------------------------------------------------------------------
    // Boot
    // ------------------------------------------------------------------
    if (history.length) {
        history.forEach((entry) => addMessage(entry, { animate: false }));
    } else {
        renderWelcome();
        showSuggestions(['I have two options', "I'm not sure how I feel", 'Check my heart']);
    }
    requestAnimationFrame(heartLoop);
    els.input.focus();
})();
