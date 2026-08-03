const root = document.documentElement;
const themeToggle = document.querySelector('#themeToggle');
const themeLabel = document.querySelector('#themeLabel');
const storedTheme = localStorage.getItem('edgeflow-theme');
const preferredDark = window.matchMedia?.('(prefers-color-scheme: dark)').matches;

function setTheme(theme) {
  root.dataset.theme = theme;
  localStorage.setItem('edgeflow-theme', theme);
  themeLabel.textContent = theme === 'dark' ? 'Light mode' : 'Dark mode';
  themeToggle.setAttribute('aria-label', theme === 'dark' ? '切換淺色模式' : '切換深色模式');
}
setTheme(storedTheme || (preferredDark ? 'dark' : 'light'));
themeToggle.addEventListener('click', () => setTheme(root.dataset.theme === 'dark' ? 'light' : 'dark'));

const profiles = {
  chat: {
    orb: '1×', name: 'Interactive chat', meta: 'P50 1,024 prompt · 128 output · concurrency 1',
    logo: 'TC', plan: 'PyTorch · reduce-overhead', detail: 'BF16 · static bucket · CUDA Graph', confidence: 'High',
    session: '−14.8%', startup: '8.4 s', vram: '8.7 GB', quality: 'Within gate',
    scope: 'Prompt ≤ 1536 · C=1 · 20+ requests', fallback: 'PyTorch eager BF16'
  },
  document: {
    orb: '4K', name: 'Document analysis', meta: 'P50 4,096 prompt · 256 output · concurrency 1',
    logo: 'LC', plan: 'llama.cpp · Q6_K', detail: 'GGUF · flash attention · full GPU offload', confidence: 'Medium',
    session: '−11.2%', startup: '3.1 s', vram: '4.9 GB', quality: 'Within gate',
    scope: 'Prompt 2048–6144 · C=1 · short sessions', fallback: 'llama.cpp Q8_0'
  },
  batch: {
    orb: '4×', name: 'Batch evaluation', meta: 'P50 512 prompt · 32 output · concurrency 4',
    logo: 'VL', plan: 'vLLM · continuous batching', detail: 'BF16 · token budget 4096 · max sequences 8', confidence: 'High',
    session: '−19.6%', startup: '11.7 s', vram: '10.8 GB', quality: 'Exact parity',
    scope: 'Concurrency 2–8 · 100+ requests', fallback: 'PyTorch eager BF16'
  }
};

const ids = ['orbValue','workloadName','workloadMeta','runtimeLogo','planName','planDetail','confidenceValue','sessionCost','startupCost','vramValue','qualityValue','scopeText','fallbackText'];
const map = ['orb','name','meta','logo','plan','detail','confidence','session','startup','vram','quality','scope','fallback'];
document.querySelectorAll('.segment').forEach((button) => {
  button.addEventListener('click', () => {
    document.querySelectorAll('.segment').forEach((b) => b.classList.toggle('active', b === button));
    const profile = profiles[button.dataset.profile];
    ids.forEach((id, index) => { document.getElementById(id).textContent = profile[map[index]]; });
  });
});

document.querySelectorAll('.evidence-node').forEach((node) => {
  node.addEventListener('click', () => {
    document.querySelector('#evidenceDetail p').textContent = node.dataset.detail;
    document.querySelectorAll('.evidence-node').forEach((n) => n.setAttribute('aria-pressed', n === node ? 'true' : 'false'));
  });
});

const search = document.querySelector('#runSearch');
const statusFilter = document.querySelector('#statusFilter');
const rows = [...document.querySelectorAll('#runTable tr')];
function filterRows() {
  const query = search.value.trim().toLowerCase();
  const status = statusFilter.value;
  let visible = 0;
  rows.forEach((row) => {
    const matchText = !query || row.textContent.toLowerCase().includes(query);
    const matchStatus = status === 'all' || row.dataset.status === status;
    row.hidden = !(matchText && matchStatus);
    if (!row.hidden) visible += 1;
  });
  document.querySelector('#rowCount').textContent = `${visible} demo record${visible === 1 ? '' : 's'}`;
}
search.addEventListener('input', filterRows);
statusFilter.addEventListener('change', filterRows);

document.querySelector('.notice-dismiss').addEventListener('click', (event) => event.currentTarget.closest('.notice').remove());

const dialog = document.querySelector('#tuneDialog');
document.querySelector('#newTune').addEventListener('click', () => dialog.showModal());

const toast = document.querySelector('#toast');
function showToast(message) {
  toast.textContent = message;
  toast.classList.add('show');
  window.setTimeout(() => toast.classList.remove('show'), 1700);
}

document.querySelector('#copyCommand').addEventListener('click', async () => {
  const command = 'edgeflow experiment run --matrix configs/screening.yaml';
  try {
    await navigator.clipboard.writeText(command);
    showToast('Reproduce command copied');
  } catch {
    showToast(command);
  }
});

dialog.addEventListener('close', () => {
  if (dialog.returnValue === 'default') showToast('Demo WorkloadSpec created');
});
