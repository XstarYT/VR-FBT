// Exercise the local-network HTTPS + direct WebRTC path in the shipped phone page.
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
for (const id of ['video', 'canvas', 'notice', 'status', 'camera', 'name', 'fps', 'quality', 'start', 'stop', 'empty', 'privacy']) element(id);
elements.video.play = async () => {};
elements.canvas.getContext = () => ({drawImage() {}});
elements.fps.value = '30'; elements.quality.value = '0.7';

const peers = [], requests = [], timers = new Map();
let timerId = 0;
class Peer {
  constructor(configuration) {
    this.configuration = configuration; this.iceGatheringState = 'complete'; this.connectionState = 'new';
    this.localDescription = null; this.handlers = {}; peers.push(this);
  }
  addTrack() {}
  addEventListener(type, fn) { this.handlers[type] = fn; }
  removeEventListener() {}
  async createOffer() { return {sdp: 'offer-sdp', type: 'offer'}; }
  async setLocalDescription(value) { this.localDescription = value; }
  async setRemoteDescription(value) { this.remoteDescription = value; this.connectionState = 'connected'; this.handlers.connectionstatechange?.(); }
  async getStats() { return new Map([['video', {type: 'outbound-rtp', kind: 'video', framesEncoded: 12}]]); }
  close() { this.connectionState = 'closed'; }
}
const track = {label: 'HONOR rear camera', stop() {}, addEventListener() {}};
const stream = {getTracks: () => [track], getVideoTracks: () => [track]};
const context = {
  document: {getElementById: element, createElement: () => ({})},
  window: {isSecureContext: true, addEventListener() {}},
  location: {hash: '#token=direct-secret', search: '', protocol: 'https:', host: `192.168.1.25:9443`, hostname: '192.168.1.25'},
  navigator: {mediaDevices: {enumerateDevices: async () => [], getUserMedia: async () => stream}},
  localStorage: {getItem: () => null, setItem() {}},
  crypto: {randomUUID: () => 'android-phone'}, URLSearchParams, RTCPeerConnection: Peer,
  AbortController,
  fetch: async (url, options) => { requests.push({url, options}); return {ok: true, json: async () => ({sdp: 'answer-sdp', type: 'answer'})}; },
  WebSocket: class { constructor() { throw new Error('WebSocket fallback must not run'); } },
  setInterval: fn => { timers.set(++timerId, fn); return timerId; },
  setTimeout: fn => { timers.set(++timerId, fn); return timerId; },
  clearInterval: id => timers.delete(id), clearTimeout: id => timers.delete(id),
};
vm.runInNewContext(script, context);

(async () => {
  await elements.start.handlers.click();
  assert.equal(peers.length, 1);
  assert.equal(peers[0].configuration.iceServers.length, 0);
  assert.equal(requests.length, 1);
  assert.match(requests[0].url, /^\/api\/webrtc\/offer\?/);
  assert.match(requests[0].url, /token=direct-secret/);
  assert.equal(JSON.parse(requests[0].options.body).type, 'offer');
  assert.equal(elements.status.textContent, 'DIRECT · LAN');
  assert.match(elements.privacy.textContent, /No cloud relay/);
  elements.stop.handlers.click();
  assert.equal(elements.status.textContent, 'OFFLINE');
  assert.equal(peers[0].connectionState, 'closed');
  console.log('PASS: LAN HTTPS secure context, no ICE servers, direct WebRTC offer and cleanup');
})().catch(error => { console.error(error); process.exitCode = 1; });
