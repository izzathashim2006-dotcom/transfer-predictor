/* Shared animations: scroll-reveal, count-up, stagger. One-shot only;
   respects prefers-reduced-motion; no infinite loops. */
(function () {
  const reduced = matchMedia('(prefers-reduced-motion: reduce)').matches;
  if (reduced) {
    document.documentElement.classList.add('no-anim');
    return;
  }

  /* ---- count-up: animates the numeric part of text like "0.653", "EUR 3.8M", "56,204" ---- */
  function countUp(el, dur) {
    const txt = el.textContent;
    const m = txt.match(/-?\d[\d,]*(\.\d+)?/);
    if (!m) return;
    const numStr = m[0].replace(/,/g, '');
    const val = parseFloat(numStr);
    if (isNaN(val) || (val === 0)) return;
    const dec = (numStr.split('.')[1] || '').length;
    const prefix = txt.slice(0, m.index);
    const suffix = txt.slice(m.index + m[0].length);
    const steps = 24, step = Math.max(16, (dur || 900) / steps);
    let i = 0;
    const tick = () => {
      i++;
      const k = 1 - Math.pow(1 - i / steps, 3);
      const v = val * k;
      el.textContent = prefix + v.toLocaleString(undefined,
        { minimumFractionDigits: dec, maximumFractionDigits: dec }) + suffix;
      if (i < steps) setTimeout(tick, step);
      else el.textContent = txt;
    };
    if (document.hidden) { return; }  // hidden tab: keep final value, skip animation
    tick();
  }
  window.__animateCount = countUp;

  /* ---- scroll-reveal ---- */
  const io = new IntersectionObserver(entries => {
    entries.forEach(e => {
      if (e.isIntersecting) {
        const el = e.target;
        const sibs = el.parentElement ? [...el.parentElement.children].filter(c => c.classList.contains('reveal')) : [];
        el.style.transitionDelay = Math.min(sibs.indexOf(el) * 70, 350) + 'ms';
        el.classList.add('in');
        io.unobserve(el);
      }
    });
  }, { threshold: 0.1 });

  function observeAll() {
    document.querySelectorAll('.card,.metric,.bf-item,.radar-row,.lt-row,.stat,.rbar,.crumbs').forEach(el => {
      if (!el.classList.contains('reveal')) {
        el.classList.add('reveal');
        io.observe(el);
      }
    });
  }

  document.addEventListener('DOMContentLoaded', observeAll);

  /* ---- reveal dynamically injected rows (radar, prediction cards) ---- */
  const mo = new MutationObserver(muts => {
    muts.forEach(m => {
      [...m.addedNodes].forEach(n => {
        if (n.nodeType !== 1) return;
        const targets = n.matches('.radar-row,.bf-item,.lt-row,.stat') ? [n]
          : (n.querySelectorAll ? [...n.querySelectorAll('.radar-row,.bf-item,.lt-row,.stat')] : []);
        targets.forEach(t => {
          if (!t.classList.contains('reveal')) {
            t.classList.add('reveal');
            io.observe(t);
          }
        });
      });
    });
  });
  document.addEventListener('DOMContentLoaded', () =>
    mo.observe(document.body, { childList: true, subtree: true }));
})();
