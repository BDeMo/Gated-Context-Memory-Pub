// Set this only for deployment to a custom domain instead of GitHub Pages.
const repositoryOverride = '';
const owner = location.hostname.endsWith('.github.io')
  ? location.hostname.slice(0, -'.github.io'.length) : '';
const repository = location.pathname.split('/').filter(Boolean)[0] || '';
const repositoryUrl = repositoryOverride || (owner && repository
  ? `https://github.com/${encodeURIComponent(owner)}/${encodeURIComponent(repository)}`
  : 'https://github.com/nokia-applied-research/Gated-Context-Memory-Pub');
document.querySelectorAll('.repo-link').forEach(link => { link.href = repositoryUrl; });
