/**
 * Main Interactive Logic: Theme Switcher, Code Copy, Active Anchor Spy, Mobile Nav
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

  // 3. Code Block Copy to Clipboard
  function setupCodeCopy() {
    const codeBlocks = document.querySelectorAll('pre');
    codeBlocks.forEach(pre => {
      const parent = pre.parentElement;
      // If already wrapped, skip
      if (parent && parent.classList.contains('code-block-wrapper')) return;

      const wrapper = document.createElement('div');
      wrapper.className = 'code-block-wrapper';

      const header = document.createElement('div');
      header.className = 'code-header';
      
      const lang = pre.querySelector('code')?.className || 'python';
      const cleanLang = lang.replace('language-', '').replace('hljs', '').trim() || 'python';
      
      header.innerHTML = `<span>${cleanLang.toUpperCase()}</span><button class="copy-btn">Copy</button>`;
      
      pre.parentNode.insertBefore(wrapper, pre);
      wrapper.appendChild(header);
      wrapper.appendChild(pre);

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
    });
  }

  // 4. Scroll Spy for "On This Page" TOC
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
    setupCodeCopy();
    setupScrollSpy();
  });
})();
