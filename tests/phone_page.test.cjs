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
elements.canvas.toBlob = fn => fn(new Blob(['jpeg']));
elements.fps.value = '15'; elements.quality.value = '0.7';
let permissionResolve;
const sockets = [];
const reportedCameraErrors = [];
class Socket {
  static OPEN = 1; static CONNECTING = 0;
  constructor(url) { this.url = url; this.readyState = 0; this.bufferedAmount = 0; sockets.push(this); }
  close() { this.readyState = 3; this.onclose?.(); }
  send(blob) { this.lastSent = blob; }
}
let timerId = 0;
let clockMs = 1000;
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
  fetch: async (url, options) => { reportedCameraErrors.push({url, body: JSON.parse(options.body)}); return {ok: true}; },
  performance: {now: () => clockMs}, Blob,
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
  const clockRequest = JSON.parse(sockets[0].lastSent);
  assert.equal(clockRequest.type, 'clock');
  for (const fn of [...timers.values()]) fn();
  assert.equal(typeof sockets[0].lastSent, 'string', 'No untimed JPEG before clock synchronization');
  clockMs = 1010;
  sockets[0].onmessage({data: JSON.stringify({type: 'clock', client: clockRequest.client, server: 100.005})});
  for (const fn of [...timers.values()]) fn();
  assert.ok(sockets[0].lastSent instanceof Blob);
  const packet = new DataView(await sockets[0].lastSent.arrayBuffer());
  assert.equal(packet.getUint32(0), 0x56465431);
  assert.ok(Math.abs(packet.getFloat64(4) - 100.01) < 1e-6);
  assert.ok(Math.abs(packet.getFloat64(12) - 0.005) < 1e-6);
  assert.equal(elements.canvas.width, 1920);
  assert.equal(elements.canvas.height, 1080);
  elements.stop.handlers.click();
  assert.equal(second.track.stopped, true);
  assert.equal(elements.status.textContent, 'OFFLINE');
  assert.equal(elements.start.disabled, false);
  assert.equal(timers.size, 0);
  // The real video callback must stamp the captured frame, not JPEG completion.
  const videoCallbacks = new Map();
  let frameId = 0, finishEncoding;
  elements.video.requestVideoFrameCallback = fn => { videoCallbacks.set(++frameId, fn); return frameId; };
  elements.video.cancelVideoFrameCallback = id => videoCallbacks.delete(id);
  elements.canvas.toBlob = fn => { finishEncoding = fn; };
  clockMs = 2000;
  const thirdStart = elements.start.handlers.click();
  const third = camera(); permissionResolve(third); await thirdStart;
  const timedSocket = sockets[1];
  timedSocket.readyState = 1; timedSocket.onopen();
  const request = JSON.parse(timedSocket.lastSent);
  clockMs = 2010;
  timedSocket.onmessage({data: JSON.stringify({type: 'clock', client: request.client, server: 200.005})});
  clockMs = 2070;
  const [callbackId, callback] = [...videoCallbacks][0]; videoCallbacks.delete(callbackId);
  callback(clockMs, {captureTime: 2030, presentationTime: 2060});
  clockMs = 2200;
  finishEncoding(new Blob(['late-encoded-jpeg']));
  const timedPacket = new DataView(await timedSocket.lastSent.arrayBuffer());
  assert.ok(Math.abs(timedPacket.getFloat64(4) - 200.030) < 1e-6);
  elements.stop.handlers.click();
  assert.equal(videoCallbacks.size, 0);
  assert.equal(timers.size, 0);
  context.navigator.mediaDevices.getUserMedia = async () => {
    const error = new Error('Camera blocked by browser'); error.name = 'SecurityError'; throw error;
  };
  await elements.start.handlers.click();
  assert.equal(reportedCameraErrors.length, 1);
  assert.equal(reportedCameraErrors[0].body.code, 'SecurityError');
  assert.equal(reportedCameraErrors[0].body.device_id, 'test-phone');
  assert.match(reportedCameraErrors[0].url, /token=session-secret/);
  assert.match(elements.notice.textContent, /SecurityError/);
  console.log('PASS: Private browsing, cancel during permission, fragment token, WSS streaming and Stop cleanup');
})().catch(error => { console.error(error); process.exitCode = 1; });
