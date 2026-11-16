/**
 * Instant Client-Side Full-Text Search for "Data Science & Machine Learning for Physics"
 */

(function () {
  let searchIndex = null;
  let activeIndex = -1;

  async function loadSearchIndex() {
    if (searchIndex) return searchIndex;
    try {
      // Determine root-relative path for search_index.json
      const basePath = document.body.getAttribute('data-basepath') || './';
      const res = await fetch(`${basePath}search_index.json`);
      if (!res.ok) throw new Error('Failed to load search_index.json');
      searchIndex = await res.json();
      return searchIndex;
    } catch (err) {
      console.warn('Search index load error:', err);
      return [];
    }
  }

  function initSearchUI() {
    const backdrop = document.getElementById('search-modal-backdrop');
    const input = document.getElementById('search-modal-input');
    const resultsContainer = document.getElementById('search-modal-results');
    const openBtns = document.querySelectorAll('.trigger-search-btn');

    if (!backdrop || !input || !resultsContainer) return;

    function openModal() {
      backdrop.classList.add('open');
      input.value = '';
      resultsContainer.innerHTML = '<div style="padding:16px; color:var(--text-muted); font-size:0.85rem; text-align:center;">Type to search equations, physical principles, or project names...</div>';
      input.focus();
      loadSearchIndex();
    }

    function closeModal() {
      backdrop.classList.remove('open');
      activeIndex = -1;
    }

    openBtns.forEach(btn => btn.addEventListener('click', openModal));

    backdrop.addEventListener('click', (e) => {
      if (e.target === backdrop) closeModal();
    });

    document.addEventListener('keydown', (e) => {
      // Open with "/" or "Cmd+K" / "Ctrl+K" when not focused on an input
      if ((e.key === '/' && document.activeElement.tagName !== 'INPUT' && document.activeElement.tagName !== 'TEXTAREA') ||
          ((e.metaKey || e.ctrlKey) && e.key.toLowerCase() === 'k')) {
        e.preventDefault();
        openModal();
      } else if (e.key === 'Escape' && backdrop.classList.contains('open')) {
        closeModal();
      }
    });

    input.addEventListener('input', async (e) => {
      const q = e.target.value.trim().toLowerCase();
      if (!q) {
        resultsContainer.innerHTML = '<div style="padding:16px; color:var(--text-muted); font-size:0.85rem; text-align:center;">Type to search equations, physical principles, or project names...</div>';
        return;
      }

      const index = await loadSearchIndex();
      const tokens = q.split(/\s+/).filter(Boolean);

      // Score matches
      const matches = [];
      for (const item of index) {
        let score = 0;
        const titleLower = item.title.toLowerCase();
        const partLower = item.part.toLowerCase();
        const contentLower = item.content.toLowerCase();
        const keywordsLower = (item.keywords || []).join(' ').toLowerCase();

        let allTokensMatch = true;
        for (const token of tokens) {
          let tokenFound = false;
          if (titleLower.includes(token)) {
            score += 15;
            tokenFound = true;
          }
          if (keywordsLower.includes(token)) {
            score += 8;
            tokenFound = true;
          }
          if (partLower.includes(token)) {
            score += 4;
            tokenFound = true;
          }
          if (contentLower.includes(token)) {
            score += 2;
            tokenFound = true;
          }
          if (!tokenFound) {
            allTokensMatch = false;
            break;
          }
        }

        if (allTokensMatch && score > 0) {
          // Extract snippet
          let snippet = '';
          const firstTokenPos = contentLower.indexOf(tokens[0]);
          if (firstTokenPos !== -1) {
            const start = Math.max(0, firstTokenPos - 45);
            const end = Math.min(contentLower.length, firstTokenPos + 95);
            snippet = (start > 0 ? '...' : '') + item.content.substring(start, end) + (end < item.content.length ? '...' : '');
          } else {
            snippet = item.content.substring(0, 110) + '...';
          }

          matches.push({
            item,
            score,
            snippet
          });
        }
      }

      matches.sort((a, b) => b.score - a.score);
      renderResults(matches.slice(0, 12), tokens);
    });

    input.addEventListener('keydown', (e) => {
      const items = resultsContainer.querySelectorAll('.search-result-item');
      if (!items.length) return;

      if (e.key === 'ArrowDown') {
        e.preventDefault();
        activeIndex = (activeIndex + 1) % items.length;
        updateSelected(items);
      } else if (e.key === 'ArrowUp') {
        e.preventDefault();
        activeIndex = (activeIndex - 1 + items.length) % items.length;
        updateSelected(items);
      } else if (e.key === 'Enter') {
        e.preventDefault();
        if (activeIndex >= 0 && items[activeIndex]) {
          items[activeIndex].click();
        }
      }
    });

    function updateSelected(items) {
      items.forEach((it, idx) => {
        if (idx === activeIndex) {
          it.classList.add('selected');
          it.scrollIntoView({ block: 'nearest' });
        } else {
          it.classList.remove('selected');
        }
      });
    }

    function renderResults(results, tokens) {
      activeIndex = -1;
      if (!results.length) {
        resultsContainer.innerHTML = '<div style="padding:16px; color:var(--text-muted); font-size:0.85rem; text-align:center;">No matching physics topics or chapters found.</div>';
        return;
      }

      const basePath = document.body.getAttribute('data-basepath') || './';
      let html = '';
      results.forEach((res, i) => {
        const item = res.item;
        let highlightedTitle = escapeHtml(item.title);
        let highlightedSnippet = escapeHtml(res.snippet);

        tokens.forEach(tok => {
          const regex = new RegExp(`(${escapeRegExp(tok)})`, 'gi');
          highlightedTitle = highlightedTitle.replace(regex, '<span class="search-highlight">$1</span>');
          highlightedSnippet = highlightedSnippet.replace(regex, '<span class="search-highlight">$1</span>');
        });

        html += `
          <a href="${basePath}${item.url}" class="search-result-item" data-index="${i}">
            <div class="search-result-part">${escapeHtml(item.part)}</div>
            <div class="search-result-title">${highlightedTitle}</div>
            <div class="search-result-snippet">${highlightedSnippet}</div>
          </a>
        `;
      });

      resultsContainer.innerHTML = html;
    }

    function escapeHtml(str) {
      return str.replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;').replace(/"/g, '&quot;');
    }

    function escapeRegExp(string) {
      return string.replace(/[.*+?^${}()|[\]\\]/g, '\\$&');
    }
  }

  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', initSearchUI);
  } else {
    initSearchUI();
  }
})();
