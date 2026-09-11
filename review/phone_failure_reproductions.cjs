// Reuse the existing browser adapters while changing their failure behavior.
// Run from the repository root: node review/phone_failure_reproductions.cjs
const fs = require('node:fs');
const vm = require('node:vm');
const source = fs.readFileSync('tests/phone_webrtc_page.test.cjs', 'utf8');
const prefix = source.slice(0, source.indexOf('vm.runInNewContext(script, context);'));

const probe = `
let constraintChanges = 0;
const fallbackSockets = [];
context.WebSocket = class {
  static OPEN = 1; static CONNECTING = 0;
  constructor(url) { this.url = url; this.readyState = 0; fallbackSockets.push(this); }
  close() { this.readyState = 3; this.onclose?.(); }
};
track.applyConstraints = async () => { constraintChanges++; };
Peer.prototype.setRemoteDescription = async function(answer) {
  this.remoteDescription = answer;
  this.connectionState = 'connecting';
  this.handlers.connectionstatechange?.();
};
vm.runInNewContext(script, context);
(async () => {
  await elements.start.handlers.click();
  assert.equal(peers[0].connectionState, 'connecting');
  assert.equal(timers.size, 1);
  elements.fps.value = '10';
  await elements.fps.handlers.change();
  assert.equal(constraintChanges, 1);
  assert.equal(requestedConstraints.video.frameRate.max, 30);
  console.log('PASS F12 FPS control: selecting 10 FPS applies new constraints to the active direct-video track');
  const deadline = [...timers.values()][0];
  deadline();
  assert.equal(fallbackSockets.length, 1);
  assert.equal(peers[0].connectionState, 'closed');
  console.log('PASS F11 WebRTC timeout: a permanently connecting peer reaches compatibility fallback on the complete-attempt deadline');
  elements.stop.handlers.click();
})().catch(error => { console.error(error); process.exitCode = 1; });
`;
vm.runInNewContext(prefix + probe, {require, console, process, AbortController, URLSearchParams});
