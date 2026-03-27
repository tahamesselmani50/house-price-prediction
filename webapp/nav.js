/* nav.js — Shared navigation component */

const PAGES = [
  { href: 'index.html',      label: 'Accueil',      icon: '🏠' },
  { href: 'eda.html',        label: 'EDA',          icon: '📊' },
  { href: 'models.html',     label: 'Modèles',      icon: '🤖' },
  { href: 'predict.html',    label: 'Prédiction',   icon: '⚡' },
  { href: 'about.html',      label: 'À Propos',     icon: 'ℹ️' },
];

function renderNav() {
  const current = window.location.pathname.split('/').pop() || 'index.html';
  document.getElementById('navbar').innerHTML = `
    <nav class="navbar">
      <a href="index.html" class="nav-brand">
        <div class="nav-logo">
          <svg viewBox="0 0 24 24"><path d="M3 9.5L12 3l9 6.5V20a1 1 0 01-1 1H4a1 1 0 01-1-1V9.5z"/></svg>
        </div>
        <div>
          <span class="nav-brand-text">HousePriceML</span>
          <span class="nav-brand-sub">Ames Housing · Kaggle</span>
        </div>
      </a>
      <div class="nav-links">
        ${PAGES.map(p => `
          <a href="${p.href}" class="nav-link ${current===p.href?'active':''}">
            ${p.label}
          </a>`).join('')}
      </div>
      <div class="nav-right">
        <span class="nav-badge">
          <svg viewBox="0 0 24 24" fill="currentColor"><path d="M18.825 23.859c-.022.092-.117.141-.281.141h-3.139c-.187 0-.351-.082-.492-.248l-5.178-6.589-1.448 1.374v5.111c0 .235-.117.352-.351.352H5.505c-.236 0-.354-.117-.354-.352V.353c0-.233.118-.353.354-.353h2.431c.234 0 .351.12.351.353v14.343l6.203-6.272c.165-.165.33-.246.495-.246h3.239c.144 0 .236.06.285.18.046.149.034.255-.036.315l-6.555 6.344 6.836 8.507c.095.104.117.208.07.331"/></svg>
          R² = 0.905
        </span>
      </div>
    </nav>
  `;
}

function animBars() {
  document.querySelectorAll('[data-w]').forEach(el => {
    setTimeout(() => { el.style.width = el.getAttribute('data-w'); }, 100);
  });
}

document.addEventListener('DOMContentLoaded', () => {
  renderNav();
  animBars();
});
