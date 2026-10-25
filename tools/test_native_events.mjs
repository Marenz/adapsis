// Offline regression over the actual browser broadcast handler; no live server.
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import vm from 'node:vm';

const html = readFileSync(new URL('../web/adapsis.html', import.meta.url), 'utf8');
const start = html.indexOf('(function setupEventBroadcast()');
assert(start >= 0);
const handler = html.slice(start, html.indexOf('})();', start) + 5);
const messages = [];
const results = [];
let stream;
function node() {
  return { textContent: '', children: [], appendChild(child) { this.children.push(child); } };
}
vm.runInNewContext(handler, {
  EventSource: class { constructor() { stream = this; } },
  sending: false,
  document: { createElement: node },
  chatEl: { scrollTop: 0, scrollHeight: 0 },
  addMessage(role, text) {
    const body = node();
    const message = { role, text, body, querySelector() { return body; } };
    messages.push(message);
    return message;
  },
  addResult(message, text, kind) { results.push({ message, text, kind }); },
  refreshProgram() {},
});
for (const event of [
  { type: 'start', message: 'fixture' },
  { type: 'iteration', n: 1 },
  { type: 'text', text: 'Native reply' },
  { type: 'result', message: 'Applied', success: true },
  { type: 'error', message: 'No finish' },
  { type: 'done', completed: false, has_errors: true },
]) stream.onmessage({ data: JSON.stringify(event) });
assert(messages.some(m => m.body.children.some(child => child.textContent === 'Native reply')));
assert(results.some(result => result.text === 'OK: Applied'));
assert(results.some(result => result.kind === 'error' && result.text === 'ERROR: No finish'));
assert(messages.some(message => message.text.startsWith('Incomplete')));
assert(!messages.some(message => message.text === 'Done'));
stream.onmessage({ data: JSON.stringify({ type: 'done', completed: true, has_errors: true }) });
assert(messages.at(-1).text.includes('reported errors'));
console.log('Native browser broadcast regression passed');
