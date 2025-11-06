// static/js/script.js
document.addEventListener('DOMContentLoaded', () => {
    // Dark mode
    const toggle = document.querySelector('.dark-mode-toggle');
    toggle?.addEventListener('click', () => {
        const isDark = document.documentElement.getAttribute('data-theme') === 'dark';
        document.documentElement.setAttribute('data-theme', isDark ? 'light' : 'dark');
    });

    // Submit loading
    const submit = document.getElementById('submitBtn');
    submit?.addEventListener('click', () => submit.classList.add('loading'));
});