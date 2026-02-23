import { loadTaxonomy } from './taxonomy.js';
import { updateSize, camera, controls } from './scene.js';
import { buildLegend } from './legend.js';
import { initViewToggle } from './views.js';
import { loadData, initLoadButton, fetchStats } from './loader.js';
import { initTooltipListeners } from './tooltip.js';
import { initSearchListeners } from './search.js';
import { startAnimationLoop } from './animate.js?v=5';
import { state } from './state.js';

// Init listeners (sync)
initViewToggle();
initTooltipListeners();
initSearchListeners();
initLoadButton();

// Start animation loop immediately (renders loading state)
startAnimationLoop();

// Fetch stats (fire-and-forget)
fetchStats();

// Load taxonomy, then build legend and load data
loadTaxonomy().then(() => {
  buildLegend();
  loadData();
});

// Resize handler
window.addEventListener('resize', updateSize);

// Demo automation bridge
window.meshDemo = { camera, controls, state };
