const byId = id => document.getElementById(id);
let busy = false;
function message(label, kind = '') {
  byId('welcome').hidden = true;
  const article = document.createElement('article');
  article.className = `message ${kind}`;
  const heading = document.createElement('div');
  heading.className = 'message-label'; heading.textContent = label;
  const body = document.createElement('div'); body.className = 'message-body';
  article.append(heading, body); byId('messages').append(article);
  return body;
}
function scrollDown() { byId('messages').scrollTop = byId('messages').scrollHeight; }
async function request(url, options) {
  const response = await fetch(url, options);
  let data;
  try { data = await response.json(); } catch { throw new Error(`The server returned an unreadable response (${response.status}). Check the server logs.`); }
  if (!response.ok) throw new Error([data.error || `Request failed (${response.status})`, data.detail].filter(Boolean).join(' — '));
  return data;
}
byId('chat-form').addEventListener('submit', async event => {
  event.preventDefault(); const query = byId('query').value.trim(); if (!query || busy) return;
  busy = true;
  ['send', 'clear'].forEach(id => byId(id).disabled = true);
  message('YOU', 'user').textContent = query; byId('query').value = '';
  byId('request-status').textContent = 'Searching your knowledge base and catalogues, then preparing an answer…';
  scrollDown(); const start = performance.now();
  try {
    const data = await request('/generate', {method: 'POST', headers: {'Content-Type': 'application/json'}, body: JSON.stringify({query})});
    const body = message(`ASSISTANT · ${((performance.now() - start) / 1000).toFixed(1)}s`);
    body.textContent = typeof data.response === 'string' ? data.response : JSON.stringify(data.response, null, 2);
    byId('request-status').textContent = '';
  } catch (error) {
    message('REQUEST FAILED', 'error').textContent = error.message;
    byId('request-status').textContent = 'Your question has been restored so you can retry.';
    if (!byId('query').value) byId('query').value = query;
  } finally {
    busy = false; ['send', 'clear'].forEach(id => byId(id).disabled = false); scrollDown(); byId('query').focus();
  }
});
byId('query').addEventListener('keydown', event => {
  if (event.key === 'Enter' && !event.shiftKey && !event.isComposing) { event.preventDefault(); byId('chat-form').requestSubmit(); }
});
document.querySelectorAll('.suggestions button').forEach(button => button.addEventListener('click', () => { byId('query').value = button.textContent; byId('query').focus(); }));
byId('clear').addEventListener('click', () => {
  document.querySelectorAll('.message').forEach(element => element.remove());
  byId('welcome').hidden = false; byId('request-status').textContent = ''; byId('query').value = ''; byId('query').focus();
});
byId('upload-form').addEventListener('submit', async event => {
  event.preventDefault(); const file = byId('pdf').files[0]; if (!file) return;
  if (file.size >= 25 * 1024 * 1024) { byId('upload-status').textContent = 'Choose a PDF smaller than 25 MiB.'; return; }
  const form = new FormData(); form.append('file', file);
  byId('upload').disabled = true; byId('pdf').disabled = true;
  byId('upload-status').textContent = 'Extracting and indexing… The first upload may also download the local embedding model.';
  try {
    const data = await request('/catalog/documents', {method: 'POST', body: form});
    const warnings = data.warnings || [];
    byId('upload-status').textContent = `${data.duplicate ? 'Already indexed' : 'Ready'}: ${file.name} · ${data.chunks} chunks. Ask the assistant about this brochure.${warnings.length ? ` ${warnings.length} extraction/review warnings: ${warnings.slice(0, 3).join('; ')}` : ''}`;
    byId('upload-form').reset();
  } catch (error) { byId('upload-status').textContent = error.message; }
  finally { byId('upload').disabled = false; byId('pdf').disabled = false; }
});
