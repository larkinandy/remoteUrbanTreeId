// Handle chapter links locally: native fragment navigation can be blocked or
// resolve against the host URL when the storyboard is rendered inside srcdoc.
(() => {
  const root = document.getElementById('urban-forest-storyboard');
  root.querySelectorAll('.sb-nav a[href^="#"]').forEach(link => {
    link.addEventListener('click', event => {
      if (event.ctrlKey || event.metaKey || event.shiftKey || event.altKey) return;
      const target = document.getElementById(link.getAttribute('href').slice(1));
      if (!target) return;
      event.preventDefault();
      target.scrollIntoView({behavior: 'instant', block: 'start'});
    });
  });
})();
