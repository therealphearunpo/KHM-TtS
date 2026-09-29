/**
 * KHM-TtS Web Studio Application Logic
 * Integrates:
 * - Real-time Khmer tokenization and grapheme inspection
 * - Speech synthesis requests to /api/synthesize
 * - Dataset Explorer with pagination, search, and instant playback
 * - Model statistics & system status monitoring
 */

document.addEventListener('DOMContentLoaded', () => {
  // State
  let currentPage = 1;
  const pageSize = 20;
  let totalSamples = 0;
  let activeAudio = null;
  let tokenizeTimer = null;

  // DOM Elements
  const tabButtons = document.querySelectorAll('.nav-tab');
  const tabContents = document.querySelectorAll('.tab-content');
  const textInput = document.getElementById('khmer-text-input');
  const charCounter = document.getElementById('char-counter');
  const speedSlider = document.getElementById('speed-slider');
  const speedVal = document.getElementById('speed-val');
  const btnSynthesize = document.getElementById('btn-synthesize');
  const audioElement = document.getElementById('tts-audio-element');
  const btnDownloadWav = document.getElementById('btn-download-wav');
  const durationBadge = document.getElementById('audio-duration-badge');
  const normalizedDisplay = document.getElementById('normalized-text-display');
  const tokenChipsContainer = document.getElementById('token-chips-container');
  const tokenCountBadge = document.getElementById('token-count-badge');
  const statusBadge = document.getElementById('system-status');
  const statusLabel = document.getElementById('status-label');

  // Dataset elements
  const searchInput = document.getElementById('dataset-search-input');
  const btnPagePrev = document.getElementById('btn-page-prev');
  const btnPageNext = document.getElementById('btn-page-next');
  const pageIndicator = document.getElementById('page-indicator');
  const tableBody = document.getElementById('dataset-table-body');
  const statTotalWavs = document.getElementById('stat-total-wavs');
  const statVocabSize = document.getElementById('stat-vocab-size');
  const statLexiconWords = document.getElementById('stat-lexicon-words');

  // =========================================================================
  // 1. Navigation & Tabs
  // =========================================================================
  tabButtons.forEach(btn => {
    btn.addEventListener('click', () => {
      const targetTab = btn.getAttribute('data-tab');
      tabButtons.forEach(b => {
        b.classList.remove('active');
        b.setAttribute('aria-selected', 'false');
      });
      tabContents.forEach(c => c.classList.remove('active'));

      btn.classList.add('active');
      btn.setAttribute('aria-selected', 'true');
      const activeContent = document.getElementById(`tab-${targetTab}`);
      if (activeContent) activeContent.classList.add('active');

      if (targetTab === 'dataset' && tableBody.children.length <= 1) {
        loadDatasetSamples();
      }
    });
  });

  // =========================================================================
  // 2. Synthesizer Inputs & Token Inspector
  // =========================================================================
  function updateCharCount() {
    const len = textInput.value.length;
    charCounter.textContent = `${len} តួអក្សរ`;
  }

  speedSlider.addEventListener('input', () => {
    speedVal.textContent = `${parseFloat(speedSlider.value).toFixed(2)}x`;
  });

  textInput.addEventListener('input', () => {
    updateCharCount();
    clearTimeout(tokenizeTimer);
    tokenizeTimer = setTimeout(runTokenizePreview, 350);
  });

  // Quick Preset Chips
  document.querySelectorAll('.preset-chip').forEach(chip => {
    chip.addEventListener('click', () => {
      textInput.value = chip.getAttribute('data-text');
      updateCharCount();
      runTokenizePreview();
      textInput.focus();
    });
  });

  async function runTokenizePreview() {
    const text = textInput.value.trim();
    if (!text) {
      normalizedDisplay.textContent = '—';
      tokenChipsContainer.innerHTML = '<span class="token-chip-placeholder">វាយអត្ថបទដើម្បីពិនិត្យ Token breakdown...</span>';
      tokenCountBadge.textContent = '0 tokens';
      return;
    }

    try {
      const res = await fetch('/api/tokenize', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ text })
      });
      if (!res.ok) throw new Error('Tokenize API error');
      const data = await res.json();

      normalizedDisplay.textContent = data.normalized;
      tokenCountBadge.textContent = `${data.tokens.length} tokens`;

      tokenChipsContainer.innerHTML = '';
      data.tokens.forEach((t, i) => {
        const chip = document.createElement('span');
        chip.className = 'token-chip';
        if (t === '<space>') {
          chip.classList.add('space-chip');
        } else if ([',', '.', '?', '!', ':'].includes(t)) {
          chip.classList.add('pause-chip');
        }
        const tid = data.token_ids[i] !== undefined ? data.token_ids[i] : '';
        chip.innerHTML = `${escapeHtml(t)} <span class="token-id">#${tid}</span>`;
        tokenChipsContainer.appendChild(chip);
      });
    } catch (err) {
      console.warn('Fallback client tokenizer preview:', err);
      normalizedDisplay.textContent = text;
    }
  }

  // =========================================================================
  // 3. Speech Synthesis Execution
  // =========================================================================
  btnSynthesize.addEventListener('click', async () => {
    const text = textInput.value.trim();
    if (!text) {
      alert('សូមបញ្ចូលអត្ថបទខ្មែរជាមុនសិន។');
      textInput.focus();
      return;
    }

    const speed = parseFloat(speedSlider.value);
    btnSynthesize.disabled = true;
    const originalBtnHtml = btnSynthesize.innerHTML;
    btnSynthesize.innerHTML = `
      <span class="btn-icon">
        <svg class="spin-animation" width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5">
          <circle cx="12" cy="12" r="10" stroke-opacity="0.25"></circle>
          <path d="M12 2a10 10 0 0 1 10 10"></path>
        </svg>
      </span>
      <span class="btn-text">កំពុងសំយោគសំឡេង... (Synthesizing)</span>
    `;

    try {
      const res = await fetch('/api/synthesize', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ text, speed })
      });

      if (!res.ok) {
        const errJson = await res.json().catch(() => ({}));
        throw new Error(errJson.error || `Synthesis failed (HTTP ${res.status})`);
      }

      const blob = await res.blob();
      const audioUrl = URL.createObjectURL(blob);

      audioElement.src = audioUrl;
      audioElement.load();
      audioElement.play().catch(() => {});

      btnDownloadWav.href = audioUrl;

      audioElement.onloadedmetadata = () => {
        const dur = audioElement.duration;
        if (!isNaN(dur)) {
          durationBadge.textContent = `${dur.toFixed(2)}s`;
        }
      };
    } catch (err) {
      alert(`Synthesis Error: ${err.message}\n\nNote: If model checkpoints are not yet trained, run training first.`);
    } finally {
      btnSynthesize.disabled = false;
      btnSynthesize.innerHTML = originalBtnHtml;
    }
  });

  // =========================================================================
  // 4. Dataset Explorer & Samples API
  // =========================================================================
  let searchDebounceTimer = null;
  searchInput.addEventListener('input', () => {
    clearTimeout(searchDebounceTimer);
    searchDebounceTimer = setTimeout(() => {
      currentPage = 1;
      loadDatasetSamples();
    }, 300);
  });

  btnPagePrev.addEventListener('click', () => {
    if (currentPage > 1) {
      currentPage--;
      loadDatasetSamples();
    }
  });

  btnPageNext.addEventListener('click', () => {
    const totalPages = Math.ceil(totalSamples / pageSize);
    if (currentPage < totalPages) {
      currentPage++;
      loadDatasetSamples();
    }
  });

  async function loadDatasetSamples() {
    const query = searchInput.value.trim();
    tableBody.innerHTML = `<tr><td colspan="4" class="table-loading">កំពុងផ្ទុកទិន្នន័យ (Loading samples)...</td></tr>`;

    try {
      const url = `/api/samples?q=${encodeURIComponent(query)}&page=${currentPage}&limit=${pageSize}`;
      const res = await fetch(url);
      if (!res.ok) throw new Error('Failed to load dataset samples');
      const data = await res.json();

      totalSamples = data.total;
      const totalPages = Math.max(1, Math.ceil(totalSamples / pageSize));
      pageIndicator.textContent = `ទំព័រ ${currentPage} / ${totalPages} (${totalSamples.toLocaleString()} Utterances)`;

      btnPagePrev.disabled = currentPage <= 1;
      btnPageNext.disabled = currentPage >= totalPages;

      if (!data.samples || data.samples.length === 0) {
        tableBody.innerHTML = `<tr><td colspan="4" class="table-loading">រកមិនឃើញទិន្នន័យត្រូវគ្នានឹង "${escapeHtml(query)}" ឡើយ។</td></tr>`;
        return;
      }

      tableBody.innerHTML = '';
      data.samples.forEach(s => {
        const tr = document.createElement('tr');

        tr.innerHTML = `
          <td><span class="utt-id">${escapeHtml(s.id)}</span></td>
          <td>${escapeHtml(s.text)}</td>
          <td style="text-align: center;">
            <button class="utt-audio-btn" data-id="${escapeHtml(s.id)}">
              <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5">
                <polygon points="5 3 19 12 5 21 5 3"></polygon>
              </svg>
              Play
            </button>
          </td>
          <td style="text-align: right;">
            <button class="utt-use-btn" data-text="${escapeHtml(s.text)}">Use Text</button>
          </td>
        `;
        tableBody.appendChild(tr);
      });

      // Bind Play buttons
      tableBody.querySelectorAll('.utt-audio-btn').forEach(btn => {
        btn.addEventListener('click', () => {
          const uttId = btn.getAttribute('data-id');
          playDatasetAudio(uttId, btn);
        });
      });

      // Bind Use Text buttons
      tableBody.querySelectorAll('.utt-use-btn').forEach(btn => {
        btn.addEventListener('click', () => {
          const sampleText = btn.getAttribute('data-text');
          textInput.value = sampleText;
          updateCharCount();
          runTokenizePreview();

          // Switch to synthesizer tab
          const synthTabBtn = document.getElementById('tab-btn-synthesizer');
          if (synthTabBtn) synthTabBtn.click();
          textInput.scrollIntoView({ behavior: 'smooth' });
        });
      });
    } catch (err) {
      tableBody.innerHTML = `<tr><td colspan="4" class="table-loading" style="color: var(--danger);">Error: ${err.message}</td></tr>`;
    }
  }

  function playDatasetAudio(uttId, btn) {
    if (activeAudio) {
      activeAudio.pause();
      document.querySelectorAll('.utt-audio-btn').forEach(b => {
        b.innerHTML = `
          <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5">
            <polygon points="5 3 19 12 5 21 5 3"></polygon>
          </svg> Play
        `;
      });
    }

    const audioUrl = `/audio/${encodeURIComponent(uttId)}.wav`;
    const audio = new Audio(audioUrl);
    activeAudio = audio;

    btn.innerHTML = `
      <svg class="spin-animation" width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5">
        <circle cx="12" cy="12" r="10" stroke-opacity="0.25"></circle>
        <path d="M12 2a10 10 0 0 1 10 10"></path>
      </svg> Playing...
    `;

    audio.play().catch(err => {
      alert(`Could not play audio for ${uttId}: ${err.message}`);
      btn.innerHTML = 'Play';
    });

    audio.onended = () => {
      btn.innerHTML = `
        <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5">
          <polygon points="5 3 19 12 5 21 5 3"></polygon>
        </svg> Play
      `;
    };
  }

  // =========================================================================
  // 5. System Stats Check
  // =========================================================================
  async function fetchSystemStats() {
    try {
      const res = await fetch('/api/stats');
      if (!res.ok) throw new Error('Stats API offline');
      const stats = await res.json();

      statusBadge.classList.add('online');
      statusLabel.textContent = `Server Online (${stats.total_wavs.toLocaleString()} Utterances)`;

      if (statTotalWavs) statTotalWavs.textContent = stats.total_wavs.toLocaleString();
      if (statVocabSize && stats.vocab_size) statVocabSize.textContent = stats.vocab_size;
      if (statLexiconWords && stats.lexicon_words) statLexiconWords.textContent = stats.lexicon_words.toLocaleString();
    } catch (err) {
      statusBadge.classList.remove('online');
      statusLabel.textContent = 'Server Disconnected';
    }
  }

  function escapeHtml(str) {
    if (!str) return '';
    return str
      .replace(/&/g, '&amp;')
      .replace(/</g, '&lt;')
      .replace(/>/g, '&gt;')
      .replace(/"/g, '&quot;')
      .replace(/'/g, '&#039;');
  }

  // Initial Boot
  updateCharCount();
  runTokenizePreview();
  fetchSystemStats();
});
