// Exercise the shipped script with controllable camera permissions and sockets.
const fs = require('node:fs');
const vm = require('node:vm');
const assert = require('node:assert/strict');
const html = fs.readFileSync('Content/Website/index.html', 'utf8');
const script = html.match(/<script>([\s\S]*?)<\/script>/)[1];
const elements = {};
function element(id) {
  return elements[id] ||= {value: '', options: [], handlers: {}, hidden: false,
    addEventListener(type, fn) { this.handlers[type] = fn; },
    append(option) { this.options.push(option); }};
}
for (const id of ['video', 'canvas', 'notice', 'status', 'camera', 'name', 'fps', 'quality', 'start', 'stop', 'empty']) element(id);
elements.video.play = async () => {};
elements.video.videoWidth = 1920;
elements.video.videoHeight = 1080;
elements.canvas.getContext = () => ({drawImage() {}});
elements.canvas.toBlob = fn => fn({size: 100});
elements.fps.value = '15'; elements.quality.value = '0.7';
let permissionResolve;
const sockets = [];
class Socket {
  static OPEN = 1; static CONNECTING = 0;
  constructor(url) { this.url = url; this.readyState = 0; this.bufferedAmount = 0; sockets.push(this); }
  close() { this.readyState = 3; this.onclose?.(); }
  send(blob) { this.lastSent = blob; }
}
let timerId = 0;
const timers = new Map();
const context = {
  document: {getElementById: element, createElement: () => ({})},
  window: {isSecureContext: true, addEventListener() {}},
  location: {hash: '#token=session-secret', search: '', protocol: 'https:', host: '192.168.1.25:9443'},
  navigator: {mediaDevices: {
    enumerateDevices: async () => [{kind: 'videoinput', deviceId: 'rear', label: 'Rear lens'}],
    getUserMedia: () => new Promise(resolve => { permissionResolve = resolve; })}},
  localStorage: {getItem() { throw Error('storage disabled'); }},
  crypto: {randomUUID: () => 'test-phone'}, URLSearchParams, WebSocket: Socket,
  setInterval: fn => { timers.set(++timerId, fn); return timerId; },
  setTimeout: fn => { timers.set(++timerId, fn); return timerId; },
  clearInterval: id => timers.delete(id), clearTimeout: id => timers.delete(id),
};
vm.runInNewContext(script, context);
function camera() {
  const track = {label: 'Rear lens', stopped: false, stop() { this.stopped = true; }, addEventListener() {}};
  return {track, getTracks: () => [track], getVideoTracks: () => [track]};
}
(async () => {
  const cancelled = elements.start.handlers.click();
  elements.stop.handlers.click();
  const first = camera(); permissionResolve(first); await cancelled;
  assert.equal(first.track.stopped, true);
  assert.equal(sockets.length, 0);
  const started = elements.start.handlers.click();
  const second = camera(); permissionResolve(second); await started;
  assert.equal(sockets.length, 1);
  assert.match(sockets[0].url, /^wss:\/\//);
  assert.match(sockets[0].url, /token=session-secret/);
  sockets[0].readyState = 1; sockets[0].onopen();
  assert.equal(elements.status.textContent, 'CONNECTED');
  assert.equal(elements.empty.hidden, true);
  for (const fn of [...timers.values()]) fn();
  assert.ok(sockets[0].lastSent);
  assert.equal(elements.canvas.width, 1920);
  assert.equal(elements.canvas.height, 1080);
  elements.stop.handlers.click();
  assert.equal(second.track.stopped, true);
  assert.equal(elements.status.textContent, 'OFFLINE');
  assert.equal(elements.start.disabled, false);
  assert.equal(timers.size, 0);
  console.log('PASS: Private browsing, cancel during permission, fragment token, WSS streaming and Stop cleanup');
})().catch(error => { console.error(error); process.exitCode = 1; });
