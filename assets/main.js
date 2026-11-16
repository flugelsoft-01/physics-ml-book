/**
 * Main Interactive Logic: Theme Switcher, Code Copy, Progress Tracker, Mobile Nav
 * Author: Kalyanjit Hatibaruah | Publisher: Flugelsoft Labs (2026)
 */

(function () {
  // 1. Theme Switcher
  const savedTheme = localStorage.getItem('physics_book_theme') || 'light';
  document.documentElement.setAttribute('data-theme', savedTheme);

  function setupThemeToggle() {
    const toggleBtns = document.querySelectorAll('.theme-toggle-btn');
    toggleBtns.forEach(btn => {
      btn.addEventListener('click', () => {
        const currentTheme = document.documentElement.getAttribute('data-theme') || 'light';
        const newTheme = currentTheme === 'dark' ? 'light' : 'dark';
        document.documentElement.setAttribute('data-theme', newTheme);
        localStorage.setItem('physics_book_theme', newTheme);
        updateToggleButtons(newTheme);
      });
    });
    updateToggleButtons(savedTheme);
  }

  function updateToggleButtons(theme) {
    const textEls = document.querySelectorAll('.theme-mode-text');
    textEls.forEach(el => {
      el.textContent = theme === 'dark' ? '🌙 Dark' : '☀️ Light';
    });
  }

  // 2. Mobile Drawer Navigation
  function setupMobileDrawer() {
    const toggleBtn = document.querySelector('.mobile-menu-btn');
    const sidebar = document.querySelector('.sidebar');
    if (!toggleBtn || !sidebar) return;

    toggleBtn.addEventListener('click', (e) => {
      e.stopPropagation();
      sidebar.classList.toggle('open');
    });

    document.addEventListener('click', (e) => {
      if (sidebar.classList.contains('open') && !sidebar.contains(e.target) && e.target !== toggleBtn) {
        sidebar.classList.remove('open');
      }
    });
  }

  // 3. Code Block Copy to Clipboard & Download .py
  function setupCodeBlocks() {
    const codeBlocks = document.querySelectorAll('pre');
    codeBlocks.forEach((pre, index) => {
      const parent = pre.parentElement;
      if (parent && parent.classList.contains('code-block-wrapper')) return;

      const wrapper = document.createElement('div');
      wrapper.className = 'code-block-wrapper';

      const header = document.createElement('div');
      header.className = 'code-header';
      
      const lang = pre.querySelector('code')?.className || 'python';
      const cleanLang = lang.replace('language-', '').replace('hljs', '').trim() || 'python';
      
      header.innerHTML = `
        <span>${cleanLang.toUpperCase()}</span>
        <div style="display: flex; align-items: center; gap: 6px;">
          <button class="download-py-btn" title="Download as .py script">💾 .py</button>
          <button class="copy-btn">Copy</button>
        </div>
      `;
      
      pre.parentNode.insertBefore(wrapper, pre);
      wrapper.appendChild(header);
      wrapper.appendChild(pre);

      // Copy functionality
      const copyBtn = header.querySelector('.copy-btn');
      copyBtn.addEventListener('click', async () => {
        const codeText = pre.innerText;
        try {
          await navigator.clipboard.writeText(codeText);
          copyBtn.textContent = 'Copied!';
          copyBtn.style.background = '#059669';
          setTimeout(() => {
            copyBtn.textContent = 'Copy';
            copyBtn.style.background = '';
          }, 2000);
        } catch (err) {
          copyBtn.textContent = 'Error';
        }
      });

      // Download .py functionality
      const downloadBtn = header.querySelector('.download-py-btn');
      downloadBtn.addEventListener('click', () => {
        const codeText = pre.innerText;
        const blob = new Blob([codeText], { type: 'text/x-python' });
        const url = URL.createObjectURL(blob);
        const a = document.createElement('a');
        const filename = `physics_ml_script_${index + 1}.py`;
        a.href = url;
        a.download = filename;
        document.body.appendChild(a);
        a.click();
        document.body.removeChild(a);
        URL.revokeObjectURL(url);
      });
    });
  }

  // 4. Reading Progress Tracker with LocalStorage
  const PROGRESS_KEY = 'physics_ml_completed_chapters';
  const TOTAL_MODULES = 36; // 30 Capstones + 6 Reference Modules

  function getCompletedList() {
    try {
      return JSON.parse(localStorage.getItem(PROGRESS_KEY) || '[]');
    } catch {
      return [];
    }
  }

  function saveCompletedList(list) {
    localStorage.setItem(PROGRESS_KEY, JSON.stringify(list));
  }

  function updateProgressUI() {
    const completed = getCompletedList();
    const count = completed.length;
    const pct = Math.round((count / TOTAL_MODULES) * 100);

    // Update Progress Bar
    const fillEl = document.getElementById('sidebar-progress-fill');
    const textEl = document.getElementById('sidebar-progress-text');
    if (fillEl) fillEl.style.width = `${pct}%`;
    if (textEl) textEl.textContent = `${count}/${TOTAL_MODULES} (${pct}%)`;

    // Update TOC item checks
    document.querySelectorAll('.toc-item').forEach(link => {
      const href = link.getAttribute('href') || '';
      const match = href.match(/([a-zA-Z0-9_-]+)\.html$/);
      if (match) {
        const slug = match[1];
        let checkEl = link.querySelector('.toc-check');
        if (completed.includes(slug)) {
          if (!checkEl) {
            checkEl = document.createElement('span');
            checkEl.className = 'toc-check';
            checkEl.textContent = '✓';
            link.appendChild(checkEl);
          }
        } else if (checkEl) {
          checkEl.remove();
        }
      }
    });

    // Update current page button state
    const markBtn = document.getElementById('mark-complete-btn');
    if (markBtn) {
      const currentSlug = markBtn.dataset.slug;
      const isDone = completed.includes(currentSlug);
      if (isDone) {
        markBtn.classList.add('completed');
        markBtn.querySelector('.complete-checkbox-icon').textContent = '☑';
        markBtn.querySelector('.complete-checkbox-text').textContent = 'Completed';
      } else {
        markBtn.classList.remove('completed');
        markBtn.querySelector('.complete-checkbox-icon').textContent = '☐';
        markBtn.querySelector('.complete-checkbox-text').textContent = 'Mark Completed';
      }
    }
  }

  function setupProgressTracker() {
    const markBtn = document.getElementById('mark-complete-btn');
    if (markBtn) {
      markBtn.addEventListener('click', () => {
        const slug = markBtn.dataset.slug;
        if (!slug) return;
        let list = getCompletedList();
        if (list.includes(slug)) {
          list = list.filter(s => s !== slug);
        } else {
          list.push(slug);
        }
        saveCompletedList(list);
        updateProgressUI();
      });
    }
    updateProgressUI();
  }

  // 5. Scroll Spy for "On This Page" TOC
  function setupScrollSpy() {
    const links = document.querySelectorAll('.right-toc-link');
    const sections = Array.from(document.querySelectorAll('.content-container h2[id]'));
    if (!links.length || !sections.length) return;

    const observer = new IntersectionObserver((entries) => {
      entries.forEach(entry => {
        if (entry.isIntersecting) {
          const id = entry.target.getAttribute('id');
          links.forEach(link => {
            if (link.getAttribute('href') === `#${id}`) {
              link.classList.add('active');
            } else {
              link.classList.remove('active');
            }
          });
        }
      });
    }, {
      rootMargin: '0px 0px -70% 0px',
      threshold: 0
    });

    sections.forEach(sec => observer.observe(sec));
  }

  document.addEventListener('DOMContentLoaded', () => {
    setupThemeToggle();
    setupMobileDrawer();
    setupCodeBlocks();
    setupProgressTracker();
    setupScrollSpy();
  });
})();
