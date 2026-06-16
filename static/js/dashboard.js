// Dashboard JS — calls Flask API endpoints

async function predict() {
  const text = document.getElementById('review-text').value.trim();
  if (!text) return alert('Please enter a review.');
  const res = await fetch('/api/predict', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ text })
  });
  const data = await res.json();
  const box = document.getElementById('prediction-result');
  box.className = `result-box ${data.sentiment}`;
  box.innerHTML = `
    <div class="result-sentiment">${data.sentiment.toUpperCase()}</div>
    <div class="result-conf">Confidence: ${data.confidence_pct} &nbsp;·&nbsp; Tokens: ${data.token_count}</div>
    <div style="margin-top:8px;font-size:12px;color:inherit;opacity:.75">
      Positive signals: ${data.scores.positive} &nbsp;|&nbsp;
      Negative signals: ${data.scores.negative} &nbsp;|&nbsp;
      Neutral tokens: ${data.scores.neutral}
    </div>`;
}

async function batchAnalyze() {
  const raw = document.getElementById('batch-input').value.trim();
  if (!raw) return;
  const reviews = raw.split('\n').map(r => r.trim()).filter(Boolean);
  const res = await fetch('/api/batch', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ reviews })
  });
  const data = await res.json();
  const container = document.getElementById('batch-results');
  container.innerHTML = data.results.map(r => `
    <div class="batch-item">
      <span class="badge ${r.sentiment}">${r.sentiment}</span>
      <span>${r.original_text.slice(0, 100)}${r.original_text.length > 100 ? '...' : ''}</span>
      <span style="margin-left:auto;font-size:11px;color:#888;white-space:nowrap">${r.confidence_pct}</span>
    </div>`).join('');

  updateMetrics(data.results);
  drawChart(data.results);
}

function updateMetrics(results) {
  const pos = results.filter(r => r.sentiment === 'positive').length;
  const neg = results.filter(r => r.sentiment === 'negative').length;
  document.getElementById('total').textContent = results.length;
  document.getElementById('pos-count').textContent = pos;
  document.getElementById('neg-count').textContent = neg;
  document.getElementById('csat').textContent = Math.round(pos / results.length * 100) + '%';
}

let chart;
function drawChart(results) {
  const counts = { positive: 0, negative: 0, neutral: 0 };
  results.forEach(r => counts[r.sentiment]++);
  const ctx = document.getElementById('dist-chart').getContext('2d');
  if (chart) chart.destroy();
  chart = new Chart(ctx, {
    type: 'doughnut',
    data: {
      labels: ['Positive', 'Negative', 'Neutral'],
      datasets: [{ data: [counts.positive, counts.negative, counts.neutral],
        backgroundColor: ['#1D9E75', '#D85A30', '#378ADD'], borderWidth: 2, borderColor: '#fff' }]
    },
    options: { plugins: { legend: { position: 'bottom' } }, cutout: '62%' }
  });
}
