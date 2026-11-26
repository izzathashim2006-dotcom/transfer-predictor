/* Transfer window countdown — real dates, computed from today.
   Windows: summer (opens ~Jun 10, closes Sep 1) and winter (opens Jan 1, closes Feb 2). */
(function () {
  function nextBoundary(now) {
    const y = now.getFullYear();
    const pts = [];
    for (const yy of [y - 1, y, y + 1]) {
      pts.push({t: new Date(yy, 0, 1), type: 'open', label: 'winter window opens'});
      pts.push({t: new Date(yy, 1, 2), type: 'close', label: 'winter window closes'});
      pts.push({t: new Date(yy, 5, 10), type: 'open', label: 'summer window opens'});
      pts.push({t: new Date(yy, 8, 1), type: 'close', label: 'summer window closes'});
    }
    pts.sort((a, b) => a.t - b.t);
    let state = 'closed', next = null, nextLabel = '';
    for (const p of pts) {
      if (p.t > now) { next = p.t; nextLabel = p.label; break; }
      state = p.type === 'open' ? 'open' : 'closed';
    }
    return {state, next, nextLabel};
  }

  function fmt(ms) {
    const d = Math.floor(ms / 864e5), h = Math.floor(ms % 864e5 / 36e5), m = Math.floor(ms % 36e5 / 6e4);
    return (d ? d + 'd ' : '') + h + 'h ' + m + 'm';
  }

  function tick() {
    const el = document.getElementById('windowCount');
    if (!el) return;
    const now = new Date();
    const {state, next, nextLabel} = nextBoundary(now);
    if (state === 'open') {
      el.innerHTML = '🟢 <b style="color:#2dd4a8">Transfer window open</b> — closes ' +
        next.toLocaleDateString(undefined, {month: 'short', day: 'numeric'}) + ' (in ' + fmt(next - now) + ')';
    } else {
      el.innerHTML = '🔴 Transfer window closed — ' + nextLabel +
        ' ' + next.toLocaleDateString(undefined, {month: 'short', day: 'numeric'}) + ' (in ' + fmt(next - now) + ')';
    }
  }
  document.addEventListener('DOMContentLoaded', () => { tick(); setInterval(tick, 30000); });
})();
