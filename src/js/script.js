"use strict";

/* ===== CONFIG DEFAULTS (editable via livelyPropertyListener) ===== */
let color1 = "#A500FF";
let color2 = "#43FF00";
let tbColour = hexToRgb("#210F1E");
let tbDimming = 0.5;

let topOpacity = 1.0;
let amplitude = 737;
let topSamples = 128;

let frameSpread = 4;
let bottomHeight = 45;
let bufferLength = 40;

let speed = 2.0;            // new: pixels per frame (single speed control)
let bloomMultiplier = 1.0; // scales compute-pass color (can exceed 1.0 with float RT)
let bloomStrength = 0.1;   // controls whiteness/additive bloom in stretch pass
let ampPower = 0.01;
let verticalFadePower = 0.2;
let topEnabled = true;
let topType = 0;

let reverseHue = false;

let decayFactor = 0.97;   // CPU-side exponential decay per tick (0..1)

let sustainMix = 1.0;

/* top bar extra options */
let topBarSpacing = 1;      // px
let topBarSmoothing = 0.5;  // 0..1 (0 = no smoothing, 1 = no change)

/* ===== CANVASES & GL ===== */
const topCanvas = document.getElementById("topCanvas");
const bottomCanvas = document.getElementById("bottomCanvas");
const ctxTop = topCanvas.getContext("2d");

const gl = bottomCanvas.getContext("webgl2");
if (!gl) throw new Error("WebGL2 is required");

/* ===== STATE ===== */
let bottomResolution = 512; // per request
let writeIndex = 0;
let ringBuffer = null;     // Float32Array(bufferLength * bottomResolution), frames stored row-wise
let prevFrame = null;      // Float32Array(bottomResolution)
let huesRGB = null;        // Float32Array(bottomResolution*3) 0..255
let audioMapIndexes = null;
let lastAudioLen = 0;
let latestRawFrame = null;

/* top bar previous smoothing */
let topPrev = null;

/* ===== GL RESOURCES ===== */
let quadVBO = null, quadVAO = null;
let ampTexture = null;   // amplitude history: width = bottomResolution, height = bufferLength (R)
let hueTexture = null;   // hue texture: width = bottomResolution, height = 1 (RGB)
let eqTexture = null;    // eq 1D output: width = bottomResolution, height = 1 (RGB/A)
let computeProgram = null; // program that writes eqTexture (render-to-texture)
let stretchProgram = null; // program that stretches eqTexture to bottomCanvas

/* Compile-time bounds for shader loops (increase if you need bigger resolution/spread) */
const MAX_RES = 512;
const MAX_SPREAD = 32;

/* precomputed weights array (size 2*MAX_SPREAD+1) */
let uWeightsArray = new Float32Array(2 * MAX_SPREAD + 1);

/* detect float render target support (EXT_color_buffer_float) */
const extColorFloat = !!gl.getExtension("EXT_color_buffer_float");

/* ===== UTILITIES ===== */
function hexToRgb(hex) {
  hex = hex.replace("#", "").trim();
  return {
    r: parseInt(hex.slice(0, 2), 16),
    g: parseInt(hex.slice(2, 4), 16),
    b: parseInt(hex.slice(4, 6), 16)
  };
}
function rgbToHsv(r, g, b) {
  r /= 255; g /= 255; b /= 255;
  let max = Math.max(r, g, b), min = Math.min(r, g, b);
  let h = 0, s = 0, v = max;
  let d = max - min;
  s = max === 0 ? 0 : d / max;
  if (d !== 0) {
    if (max === r) h = ((g - b) / d) % 6;
    else if (max === g) h = (b - r) / d + 2;
    else h = (r - g) / d + 4;
    h = h * 60;
    if (h < 0) h += 360;
  }
  return { h, s, v };
}
function hsvToRgb(h, s, v) {
  let c = v * s, x = c * (1 - Math.abs((h / 60) % 2 - 1)), m = v - c;
  let rp = 0, gp = 0, bp = 0;
  if (0 <= h && h < 60) { rp = c; gp = x; } else if (60 <= h && h < 120) { rp = x; gp = c; } else if (120 <= h && h < 180) { gp = c; bp = x; } else if (180 <= h && h < 240) { gp = x; bp = c; } else if (240 <= h && h < 300) { rp = x; bp = c; } else { rp = c; bp = x; }
  return { r: Math.round((rp + m) * 255), g: Math.round((gp + m) * 255), b: Math.round((bp + m) * 255) };
}
function lerpHue(h0, h1, t) {
  // If reverse is false: always move forward (increasing hue) from h0 toward h1 around the circle.
  // If reverse is true: always move backward (decreasing hue) from h0 toward h1.
  if (!reverseHue) {
    const d = ( (h1 - h0 + 360) % 360 ); // 0..359 (forward delta)
    return (h0 + d * t) % 360;
  } else {
    const d = - ( (h0 - h1 + 360) % 360 ); // negative or 0 (backward delta)
    return (h0 + d * t + 360) % 360;
  }
}

/* ===== BUFFERS ALLOCATION ===== */
function allocateBuffers() {
  bottomResolution = Math.max(8, Math.round(bottomResolution));
  bufferLength = Math.max(2, Math.round(bufferLength));
  writeIndex = 0;
  ringBuffer = new Float32Array(bufferLength * bottomResolution);
  prevFrame = new Float32Array(bottomResolution);
  huesRGB = new Float32Array(bottomResolution * 3);
  audioMapIndexes = new Float32Array(bottomResolution);
  topPrev = new Float32Array(Math.max(1, topSamples)); // smoothing storage for top bars
  precomputeHueMap();
  computeAudioMap(lastAudioLen || 128);
  computeGaussianWeights(frameSpread);
}

/* ===== HUE MAP ===== */
function precomputeHueMap() {
  const a = rgbToHsv(...Object.values(hexToRgb(color1)));
  const b = rgbToHsv(...Object.values(hexToRgb(color2)));
  for (let i = 0; i < bottomResolution; i++) {
    const f = i / Math.max(1, bottomResolution - 1);
    const h = lerpHue(a.h, b.h, f);
    const rgb = hsvToRgb(h, (f*b.s)+(1-f)*a.s, (f*b.v)+(1-f)*a.v);
    huesRGB[i * 3 + 0] = rgb.r;
    huesRGB[i * 3 + 1] = rgb.g;
    huesRGB[i * 3 + 2] = rgb.b;
  }
}

/* ===== AUDIO MAPPING ===== */
function computeAudioMap(audioLen) {
  lastAudioLen = audioLen;
  for (let i = 0; i < bottomResolution; i++) {
    audioMapIndexes[i] = (i * (audioLen - 1)) / Math.max(1, bottomResolution - 1);
  }
}

/* ===== Gaussian weights (precompute on CPU) ===== */
function computeGaussianWeights(spread) {
  const sigma = Math.max(0.0001, spread / 2);
  let sum = 0;
  const N = 2 * MAX_SPREAD + 1;
  for (let i = 0; i < N; ++i) uWeightsArray[i] = 0;
  for (let off = -spread; off <= spread; ++off) {
    const w = Math.exp(-0.5 * (off / sigma) * (off / sigma));
    uWeightsArray[off + MAX_SPREAD] = w;
    sum += w;
  }
  if (sum > 0) {
    for (let i = 0; i < N; ++i) uWeightsArray[i] /= sum;
  }
}

/* ===== RING BUFFER HELPERS ===== */
function pushFrameFromAudio(audioArray) {
  const aLen = audioArray.length;
  if (aLen !== lastAudioLen) computeAudioMap(aLen);

  // Exponential decay on CPU (applied to whole ring buffer before writing new frame)
  const decay = Math.max(0.0, Math.min(0.9999, decayFactor));
  if (decay !== 1.0) {
    for (let i = 0; i < ringBuffer.length; ++i) ringBuffer[i] *= decay;
  }

  let frame = new Float32Array(bottomResolution);      // for delta (bottom visualizer)
  let rawFrame = new Float32Array(bottomResolution);   // for raw FFT (top bars)
  for (let i = 0; i < bottomResolution; i++) {
    let idxF = audioMapIndexes[i];
    let i0 = Math.floor(idxF);
    let frac = idxF - i0;
    let v0 = audioArray[Math.min(aLen - 1, i0)] || 0;
    let v1 = audioArray[Math.min(aLen - 1, i0 + 1)] || 0;
    let raw = v0 * (1 - frac) + v1 * frac;

    let delta = Math.max(0, raw - prevFrame[i]);
    frame[i] = delta;        // for bottom GPU visualizer
    rawFrame[i] = raw;       // for top bars
    prevFrame[i] = raw * sustainMix;
  }

  ringBuffer.set(frame, writeIndex * bottomResolution);
  writeIndex = (writeIndex + 1) % bufferLength;

  latestRawFrame = rawFrame;

  broadcast(Array.from(frame));
}

/* ===== TOP WAVE (with smoothing & spacing) ===== */
function drawTopWave(audioArray) {
  if (!topEnabled) return;

  const width = topCanvas.width;
  const height = topCanvas.height;
  const n = audioArray.length; // use all available audio points

  if (!topPrev || topPrev.length !== n) topPrev = new Float32Array(n);
  const points = new Array(n);

  // --- calculate smoothed Y positions for all FFT points ---
  for (let i = 0; i < n; i++) {
    const v = Math.min(1, audioArray[i] || 0);

    // exponential smoothing
    const alpha = 1 - Math.max(0, Math.min(1, topBarSmoothing));
    topPrev[i] = topPrev[i] * alpha + v * (1 - alpha);

    const vSm = topPrev[i];
    const barH = amplitude * vSm;
    const x = (i / (n - 1)) * width; // spread evenly across entire width
    const y = height - barH;
    points[i] = { x, y };
  }

  ctxTop.save();
  ctxTop.globalAlpha = topOpacity;

  // === Gradient left → right for both line and fill ===
  const gradient = ctxTop.createLinearGradient(0, 0, width, 0);
  for (let i = 0; i <= 1; i += 0.1) {
    gradient.addColorStop(i, interpolateHueColor(i));
  }

  // === Build waveform path ===
  ctxTop.beginPath();
  ctxTop.moveTo(points[0].x, points[0].y);
  for (let i = 1; i < n - 2; i++) {
    const xc = (points[i].x + points[i + 1].x) / 2;
    const yc = (points[i].y + points[i + 1].y) / 2;
    ctxTop.quadraticCurveTo(points[i].x, points[i].y, xc, yc);
  }
  ctxTop.quadraticCurveTo(points[n - 2].x, points[n - 2].y, points[n - 1].x, points[n - 1].y);

  // === Fill the solid area below the waveform ===
  ctxTop.lineTo(points[n - 1].x, height);
  ctxTop.lineTo(points[0].x, height);
  ctxTop.closePath();
  ctxTop.fillStyle = gradient;
  ctxTop.fill();

  // === Draw the outline line ===
  ctxTop.beginPath();
  ctxTop.moveTo(points[0].x, points[0].y);
  for (let i = 1; i < n - 2; i++) {
    const xc = (points[i].x + points[i + 1].x) / 2;
    const yc = (points[i].y + points[i + 1].y) / 2;
    ctxTop.quadraticCurveTo(points[i].x, points[i].y, xc, yc);
  }
  ctxTop.quadraticCurveTo(points[n - 2].x, points[n - 2].y, points[n - 1].x, points[n - 1].y);

  ctxTop.strokeStyle = gradient;
  ctxTop.lineWidth = 5; // visible even when silent
  ctxTop.lineJoin = "round";
  ctxTop.lineCap = "round";
  ctxTop.stroke();

  ctxTop.restore();
}

function drawTopBars(audioArray) {
  if (!topEnabled) return;

  const width = topCanvas.width;
  const height = topCanvas.height;
  const num = Math.max(8, Math.round(topSamples));
  const spacing = Math.round(topBarSpacing);

  ctxTop.imageSmoothingEnabled = false;

  if (!topPrev || topPrev.length < num) topPrev = new Float32Array(num);

  const totalSpacing = spacing * (num - 1);
  const usableWidth = width - totalSpacing;
  const idealBarWidth = usableWidth / num;

  let prevRight = 0;
  for (let i = 0; i < num; i++) {
    // Compute pixel-perfect x positions
    const idealLeft = i * (idealBarWidth + spacing);
    const idealRight = idealLeft + idealBarWidth;
    const x = Math.round(idealLeft);
    const nextRight = Math.round(idealRight);
    const w = nextRight - x; // ensures total adds to full width

    // audio smoothing
    const idx = Math.floor(i * (audioArray.length - 1) / Math.max(1, num - 1));
    const v = Math.min(1, audioArray[idx] || 0);
    const alpha = 1 - Math.max(0, Math.min(1, topBarSmoothing));
    topPrev[i] = topPrev[i] * alpha + v * (1 - alpha);
    const vSm = topPrev[i];

    const h = amplitude * vSm;
    const y = Math.round(height - h);

    ctxTop.fillStyle = interpolateHueColor(i / Math.max(1, num - 1));
    ctxTop.globalAlpha = topOpacity;
    ctxTop.fillRect(x, y, w, Math.round(h));

    prevRight = nextRight;
  }

  ctxTop.globalAlpha = 1.0;
}

function interpolateHueColor(t) {
  const f = t * (bottomResolution - 1);
  const i0 = Math.floor(f), i1 = Math.min(bottomResolution - 1, i0 + 1);
  const frac = f - i0;
  const r = Math.round(huesRGB[i0 * 3 + 0] * (1 - frac) + huesRGB[i1 * 3 + 0] * frac);
  const g = Math.round(huesRGB[i0 * 3 + 1] * (1 - frac) + huesRGB[i1 * 3 + 1] * frac);
  const b = Math.round(huesRGB[i0 * 3 + 2] * (1 - frac) + huesRGB[i1 * 3 + 2] * frac);
  return `rgb(${r},${g},${b})`;
}

/* ===== SHADERS ===== */
/* Vertex (same for both passes) */
const vertexSrc = `#version 300 es
in vec2 aPos;
out vec2 vUv;
void main(){ vUv = aPos * 0.5 + 0.5; gl_Position = vec4(aPos, 0.0, 1.0); }`;

/* Compute fragment:
   - rightward-only propagation: framesAgo = (x - s) / speed
   - full interpolation between consecutive frames (mix a0,a1 by fractional part)
   - accumulate over frameSpread using precomputed uWeights[]
   - outputs HDR color if float RT supported (we don't clamp here)
*/
const computeFragmentSrc = `#version 300 es
precision highp float;
precision highp int;
in vec2 vUv;
out vec4 FragColor;

uniform sampler2D uAmpTex; // amp history: width=resolution, height=bufferLength, R=amplitude
uniform sampler2D uHueTex; // hue texture: width=resolution, height=1 (RGB)
uniform int uResolution;
uniform int uBufferLength;
uniform float uSpeed; // effective pixels per frame
uniform int uFrameSpread;
uniform float uAmpPower;
uniform float uBloomMultiplier;
uniform int uNeighborLimit;
const int MAX_RES = ${MAX_RES};
const int MAX_SPREAD = ${MAX_SPREAD};
uniform float uWeights[${2 * MAX_SPREAD + 1}];

void main() {
  float xF = vUv.x * float(uResolution - 1);
  int xI = int(clamp(floor(xF + 0.5), 0.0, float(uResolution - 1)));

  float pp = max(0.0001, uSpeed);

  vec3 sumRGB = vec3(0.0);
  float sumW = 0.0;

  // iterate over source columns s (only from left side so pulses move right)
  for (int s = 0; s < MAX_RES; ++s) {
    if (s >= uResolution) break;
    if (s < xI - uNeighborLimit) continue;

    float d = xF - float(s);
    if (d < 0.0) continue; // only left-of-x contributes (drift right)

    float framesAgoF = d / pp;
    if (framesAgoF >= float(uBufferLength)) continue;

    // time interpolation between two adjacent frames
    float fFrac = fract(framesAgoF);
    int baseIdx = int(floor(framesAgoF));

    float ampSum = 0.0;
    float weightSum = 0.0;

    // accumulate contributions over frameSpread using precomputed weights
    for (int off = -MAX_SPREAD; off <= MAX_SPREAD; ++off) {
      if (abs(off) > uFrameSpread) continue;
      int fIdx = baseIdx + off;
      if (fIdx < 0 || fIdx >= uBufferLength) continue;
      float w = uWeights[off + MAX_SPREAD];

      // sample a0 and a1 for interpolation in time
      vec2 texA = vec2((float(s) + 0.5) / float(uResolution), (float(fIdx) + 0.5) / float(uBufferLength));
      float a0 = texture(uAmpTex, texA).r;
      float a1 = a0;
      if (fIdx + 1 < uBufferLength) {
        vec2 texB = vec2((float(s) + 0.5) / float(uResolution), (float(fIdx + 1) + 0.5) / float(uBufferLength));
        a1 = texture(uAmpTex, texB).r;
      }
      float ampT = mix(a0, a1, fFrac);
      ampSum += pow(ampT, uAmpPower) * w;
      weightSum += w;
    }

    float amp = (weightSum > 0.0) ? (ampSum / weightSum) : 0.0;
    // amp = pow(amp, uAmpPower); // non-linear scaling

    float contrib = amp;

    vec2 hueC = vec2((float(s) + 0.5) / float(uResolution), 0.5);
    vec3 hue = texture(uHueTex, hueC).rgb;

    sumRGB += hue * contrib;
    sumW += contrib;
  }

  vec3 col = vec3(0.0);
  if (sumW > 0.0) {
    // hue texture is 0..1; scale by bloom multiplier; DO NOT clamp here (keep HDR if supported)
    col = sumRGB * uBloomMultiplier;
  }

  FragColor = vec4(col, 1.0);
}
`;

/* Stretch fragment shader:
   - samples the 1px-high EQ texture,
   - applies vertical fade (starting bottom and fading up),
   - boosts fade for very bright values (so bright pulses bleed down and fade less),
   - then softly whiten based on overflow.
*/
const stretchFragmentSrc = `#version 300 es
precision highp float;
in vec2 vUv;
out vec4 FragColor;

uniform sampler2D uEq1D;         // 1D EQ texture
uniform float uVerticalFadePower; // how strong the vertical fade is
uniform float uBloomStrength;     // scales the overexposed white
uniform vec3 uTbColour;        // taskbar colour
uniform float uTbDimming;      // taskbar dimming (0..1)

void main() {
    // Sample the EQ 1D color
    vec3 color = texture(uEq1D, vec2(vUv.x, 0.5)).rgb;

    // Vertical fade: bottom = full color, top = tb colour
    float fade = pow(1.0 - vUv.y, uVerticalFadePower);
    vec3 fadedColor = mix(uTbColour, color, fade);

    // Compute overflow (bright over 1.0 triggers white additive)
    float maxChannel = max(max(color.r, color.g), color.b);
    float overflow = max(0.0, maxChannel - 1.0) * uBloomStrength;

    // White starts at bottom, fades with vertical fade
    vec3 whiteAdd = vec3(pow(1.0 - vUv.y, 1.0) * overflow);

    // Final color: faded color + white bloom + taskbar dimming
    FragColor = vec4(mix(fadedColor + whiteAdd, uTbColour, uTbDimming), 1.0);
}

`;

/* ===== GL UTILITIES ===== */
function compileGLShader(type, src) {
  const s = gl.createShader(type);
  gl.shaderSource(s, src);
  gl.compileShader(s);
  if (!gl.getShaderParameter(s, gl.COMPILE_STATUS)) {
    const log = gl.getShaderInfoLog(s);
    gl.deleteShader(s);
    throw new Error("Shader compile error: " + log);
  }
  return s;
}
function linkProgram(vs, fs) {
  const p = gl.createProgram();
  gl.attachShader(p, vs);
  gl.attachShader(p, fs);
  gl.bindAttribLocation(p, 0, "aPos");
  gl.linkProgram(p);
  if (!gl.getProgramParameter(p, gl.LINK_STATUS)) {
    const log = gl.getProgramInfoLog(p);
    gl.deleteProgram(p);
    throw new Error("Program link error: " + log);
  }
  return p;
}

/* ===== CREATE GL RESOURCES (call after allocateBuffers) ===== */
function createGLResources() {
  // fullscreen quad
  const verts = new Float32Array([
    -1, -1,
    1, -1,
    -1, 1,
    -1, 1,
    1, -1,
    1, 1
  ]);
  quadVBO = gl.createBuffer();
  gl.bindBuffer(gl.ARRAY_BUFFER, quadVBO);
  gl.bufferData(gl.ARRAY_BUFFER, verts, gl.STATIC_DRAW);

  quadVAO = gl.createVertexArray();
  gl.bindVertexArray(quadVAO);
  gl.enableVertexAttribArray(0);
  gl.bindBuffer(gl.ARRAY_BUFFER, quadVBO);
  gl.vertexAttribPointer(0, 2, gl.FLOAT, false, 0, 0);
  gl.bindVertexArray(null);

  // compile compute program (fragment writes 1px-high eqTexture)
  const vs = compileGLShader(gl.VERTEX_SHADER, vertexSrc);
  const fs = compileGLShader(gl.FRAGMENT_SHADER, computeFragmentSrc);
  computeProgram = linkProgram(vs, fs);

  computeProgram._locs = {
    uAmpTex: gl.getUniformLocation(computeProgram, "uAmpTex"),
    uHueTex: gl.getUniformLocation(computeProgram, "uHueTex"),
    uResolution: gl.getUniformLocation(computeProgram, "uResolution"),
    uBufferLength: gl.getUniformLocation(computeProgram, "uBufferLength"),
    uSpeed: gl.getUniformLocation(computeProgram, "uSpeed"),
    uFrameSpread: gl.getUniformLocation(computeProgram, "uFrameSpread"),
    uAmpPower: gl.getUniformLocation(computeProgram, "uAmpPower"),
    uBloomMultiplier: gl.getUniformLocation(computeProgram, "uBloomMultiplier"),
    uNeighborLimit: gl.getUniformLocation(computeProgram, "uNeighborLimit"),
    uWeights: gl.getUniformLocation(computeProgram, "uWeights")
  };

  // compile stretch program
  const vs2 = compileGLShader(gl.VERTEX_SHADER, vertexSrc);
  const fs2 = compileGLShader(gl.FRAGMENT_SHADER, stretchFragmentSrc);
  stretchProgram = linkProgram(vs2, fs2);
  stretchProgram._locs = {
    uEq1D: gl.getUniformLocation(stretchProgram, "uEq1D"),
    uVerticalFadePower: gl.getUniformLocation(stretchProgram, "uVerticalFadePower"),
    uBloomStrength: gl.getUniformLocation(stretchProgram, "uBloomStrength"),
    uTbColour: gl.getUniformLocation(stretchProgram, "uTbColour"),
    uTbDimming: gl.getUniformLocation(stretchProgram, "uTbDimming")
  };

  // amplitude texture (RGBA8)
  ampTexture = gl.createTexture();
  gl.bindTexture(gl.TEXTURE_2D, ampTexture);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.NEAREST);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.NEAREST);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
  gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA, bottomResolution, bufferLength, 0, gl.RGBA, gl.UNSIGNED_BYTE, null);

  // hue texture (RGB)
  hueTexture = gl.createTexture();
  gl.bindTexture(gl.TEXTURE_2D, hueTexture);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.LINEAR);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.LINEAR);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
  const hueData = new Uint8Array(bottomResolution * 3);
  for (let i = 0; i < bottomResolution; ++i) {
    hueData[i * 3 + 0] = Math.round(huesRGB[i * 3 + 0]);
    hueData[i * 3 + 1] = Math.round(huesRGB[i * 3 + 1]);
    hueData[i * 3 + 2] = Math.round(huesRGB[i * 3 + 2]);
  }
  gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGB, bottomResolution, 1, 0, gl.RGB, gl.UNSIGNED_BYTE, hueData);

  // eqTexture: 1px tall RGB(A) output of compute pass
  eqTexture = gl.createTexture();
  gl.bindTexture(gl.TEXTURE_2D, eqTexture);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.NEAREST);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.NEAREST);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);

  if (extColorFloat) {
    // prefer floating point render target so HDR values survive compute -> stretch
    gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA16F, bottomResolution, 1, 0, gl.RGBA, gl.FLOAT, null);
  } else {
    // fallback to 8-bit if float render targets aren't available
    gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGB, bottomResolution, 1, 0, gl.RGB, gl.UNSIGNED_BYTE, null);
  }

  // framebuffer for compute pass (render to eqTexture)
  const fb = gl.createFramebuffer();
  gl.bindFramebuffer(gl.FRAMEBUFFER, fb);
  gl.framebufferTexture2D(gl.FRAMEBUFFER, gl.COLOR_ATTACHMENT0, gl.TEXTURE_2D, eqTexture, 0);
  const status = gl.checkFramebufferStatus(gl.FRAMEBUFFER);
  if (status !== gl.FRAMEBUFFER_COMPLETE) {
    console.warn("Framebuffer incomplete status:", status);
  }
  gl.bindFramebuffer(gl.FRAMEBUFFER, null);
  computeProgram._framebuffer = fb;
}

/* ===== UPLOAD & DRAW PASSES ===== */
function uploadAmpTextureFromRingBuffer() {
  // Build Uint8Array RGBA where R = amplitude*255, others = 0, A = 255
  const tex = new Uint8Array(bottomResolution * bufferLength * 4);
  // rows: f=0 (newest) .. f=bufferLength-1 (oldest)
  for (let f = 0; f < bufferLength; ++f) {
    const bufIdx = (writeIndex - 1 - f + bufferLength) % bufferLength;
    for (let x = 0; x < bottomResolution; ++x) {
      let v = ringBuffer[bufIdx * bottomResolution + x] || 0;
      v = Math.max(0, Math.min(1, v));
      const outIdx = (f * bottomResolution + x) * 4;
      const c = Math.floor(v * 255);
      tex[outIdx + 0] = c;
      tex[outIdx + 1] = 0;
      tex[outIdx + 2] = 0;
      tex[outIdx + 3] = 255;
    }
  }
  gl.bindTexture(gl.TEXTURE_2D, ampTexture);
  gl.texSubImage2D(gl.TEXTURE_2D, 0, 0, 0, bottomResolution, bufferLength, gl.RGBA, gl.UNSIGNED_BYTE, tex);
}

function computePass_To1D() {
  // render to eqTexture (1px tall) using computeProgram
  gl.bindFramebuffer(gl.FRAMEBUFFER, computeProgram._framebuffer);
  gl.viewport(0, 0, bottomResolution, 1);
  gl.clearColor(0, 0, 0, 0);
  gl.clear(gl.COLOR_BUFFER_BIT);

  gl.useProgram(computeProgram);
  gl.bindVertexArray(quadVAO);

  gl.activeTexture(gl.TEXTURE0);
  gl.bindTexture(gl.TEXTURE_2D, ampTexture);
  gl.uniform1i(computeProgram._locs.uAmpTex, 0);

  gl.activeTexture(gl.TEXTURE1);
  gl.bindTexture(gl.TEXTURE_2D, hueTexture);
  gl.uniform1i(computeProgram._locs.uHueTex, 1);

  gl.uniform1i(computeProgram._locs.uResolution, bottomResolution);
  gl.uniform1i(computeProgram._locs.uBufferLength, bufferLength);

  // speed controls effective pixels-per-frame directly
  const effectiveSpeed = Math.max(0.0001, Number(speed) || 1.0);
  gl.uniform1f(computeProgram._locs.uSpeed, effectiveSpeed);

  gl.uniform1i(computeProgram._locs.uFrameSpread, Math.max(0, Math.round(frameSpread)));
  gl.uniform1f(computeProgram._locs.uAmpPower, ampPower);
  gl.uniform1f(computeProgram._locs.uBloomMultiplier, bloomMultiplier);

  // neighbor limit (keeps shader loops bounded)
  const neighborLimit = Math.min(bottomResolution - 1, Math.ceil(effectiveSpeed * (bufferLength + Math.max(1, frameSpread))));
  gl.uniform1i(computeProgram._locs.uNeighborLimit, neighborLimit);

  // weights (full array)
  gl.uniform1fv(computeProgram._locs.uWeights, uWeightsArray);

  gl.drawArrays(gl.TRIANGLES, 0, 6);
  gl.bindVertexArray(null);
  gl.bindFramebuffer(gl.FRAMEBUFFER, null);
}

function stretchPass_DrawToCanvas() {
  // render to default framebuffer (bottomCanvas) sampling eqTexture
  gl.viewport(0, 0, bottomCanvas.width, bottomCanvas.height);
  gl.clearColor(0, 0, 0, 0);
  gl.clear(gl.COLOR_BUFFER_BIT);

  gl.useProgram(stretchProgram);
  gl.bindVertexArray(quadVAO);

  gl.activeTexture(gl.TEXTURE0);
  gl.bindTexture(gl.TEXTURE_2D, eqTexture);
  gl.uniform1i(stretchProgram._locs.uEq1D, 0);
  gl.uniform1f(stretchProgram._locs.uVerticalFadePower, verticalFadePower);
  gl.uniform1f(stretchProgram._locs.uBloomStrength, bloomStrength);
  gl.uniform3f(stretchProgram._locs.uTbColour,
    tbColour.r / 255,
    tbColour.g / 255,
    tbColour.b / 255);
  gl.uniform1f(stretchProgram._locs.uTbDimming, Math.max(0.0, Math.min(1.0, tbDimming)));
  gl.drawArrays(gl.TRIANGLES, 0, 6);
  gl.bindVertexArray(null);
}

/* ===== RESIZE ===== */
function resizeCanvas() {
  topCanvas.width = window.innerWidth;
  topCanvas.height = Math.max(0, window.innerHeight - bottomHeight);
  bottomCanvas.width = window.innerWidth;
  bottomCanvas.height = bottomHeight;
}
window.addEventListener("resize", resizeCanvas);

/* ===== INIT ===== */
allocateBuffers();
createGLResources();
resizeCanvas();

/* ===== MAIN LOOP ===== */
let lastFrameTime = 0;
const TARGET_FPS = 60;
const FRAME_MS = 1000 / TARGET_FPS;

function renderLoop(now) {
  requestAnimationFrame(renderLoop);
  if (!now) now = performance.now();
  if (now - lastFrameTime < FRAME_MS) return;
  lastFrameTime = now;

  // clear top
  ctxTop.clearRect(0, 0, topCanvas.width, topCanvas.height);

  // prepare latestFrame for top bars (for display)
  const latestFrame = new Float32Array(bottomResolution);
  if (ringBuffer) {
    const latestIndex = (writeIndex - 1 + bufferLength) % bufferLength;
    for (let i = 0; i < bottomResolution; i++) {
      latestFrame[i] = ringBuffer[latestIndex * bottomResolution + i];
    }
  }
  if (topEnabled && latestRawFrame) {
    if (topType === 0) drawTopBars(latestRawFrame);
    else if (topType === 1) drawTopWave(latestRawFrame);
  }
  
  // GPU path: upload amp history, compute 1D EQ, then stretch (which applies vertical fade + bloom)
  if (ringBuffer) {
    uploadAmpTextureFromRingBuffer();
    computePass_To1D();
    stretchPass_DrawToCanvas();
  }
}
requestAnimationFrame(renderLoop);

/* ===== Lively hooks ===== */
function livelyAudioListener(audioArray) {
  if (!ringBuffer) allocateBuffers();
  pushFrameFromAudio(audioArray);
}
function livelyPropertyListener(name, val) {
  switch (name) {
    case "color1":
    case "color2": {
      if (name === "color1") color1 = val; else color2 = val;
      precomputeHueMap();
      // reupload hue texture
      const hueData = new Uint8Array(bottomResolution * 3);
      for (let i = 0; i < bottomResolution; ++i) {
        hueData[i * 3 + 0] = Math.round(huesRGB[i * 3 + 0]);
        hueData[i * 3 + 1] = Math.round(huesRGB[i * 3 + 1]);
        hueData[i * 3 + 2] = Math.round(huesRGB[i * 3 + 2]);
      }
      gl.bindTexture(gl.TEXTURE_2D, hueTexture);
      gl.texSubImage2D(gl.TEXTURE_2D, 0, 0, 0, bottomResolution, 1, gl.RGB, gl.UNSIGNED_BYTE, hueData);
      break;
    }
    case "tbColour":
      tbColour = hexToRgb(val);
      break;
    case "tbDimming":
      tbDimming = Number(val);
      break;
    case "topType":
      topType = val;
      break;
    case "reverse":
      reverseHue = val;
      precomputeHueMap();
      // reupload hue texture
      const hueData = new Uint8Array(bottomResolution * 3);
      for (let i = 0; i < bottomResolution; ++i) {
        hueData[i * 3 + 0] = Math.round(huesRGB[i * 3 + 0]);
        hueData[i * 3 + 1] = Math.round(huesRGB[i * 3 + 1]);
        hueData[i * 3 + 2] = Math.round(huesRGB[i * 3 + 2]);
      }
      gl.bindTexture(gl.TEXTURE_2D, hueTexture);
      gl.texSubImage2D(gl.TEXTURE_2D, 0, 0, 0, bottomResolution, 1, gl.RGB, gl.UNSIGNED_BYTE, hueData);
      break;
    case "opacity":
      topOpacity = Number(val) / 100;
      break;
    case "amplitude":
      amplitude = Number(val);
      break;
    case "topSamples":
      topSamples = Math.max(8, Math.round(val));
      topPrev = new Float32Array(Math.max(1, topSamples));
      break;
    case "frameSpread":
      frameSpread = Math.max(1, Math.round(val));
      computeGaussianWeights(frameSpread);
      break;
    case "bottomHeight":
      bottomHeight = Math.max(2, Math.round(val));
      resizeCanvas();
      break;
    case "bufferLength":
      bufferLength = Math.max(2, Math.round(val));
      allocateBuffers();
      gl.bindTexture(gl.TEXTURE_2D, ampTexture);
      gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA, bottomResolution, bufferLength, 0, gl.RGBA, gl.UNSIGNED_BYTE, null);
      if (computeProgram && computeProgram._framebuffer) {
        gl.deleteFramebuffer(computeProgram._framebuffer);
      }
      const fb = gl.createFramebuffer();
      gl.bindFramebuffer(gl.FRAMEBUFFER, fb);
      gl.framebufferTexture2D(gl.FRAMEBUFFER, gl.COLOR_ATTACHMENT0, gl.TEXTURE_2D, eqTexture, 0);
      gl.bindFramebuffer(gl.FRAMEBUFFER, null);
      computeProgram._framebuffer = fb;
      break;
    case "speed":
      speed = Number(val);
      break;
    case "decayFactor":
      decayFactor = Number(val);
      break;
    case "bloomMultiplier":
      bloomMultiplier = Number(val);
      break;
    case "bloomStrength":
      bloomStrength = Number(val);
      break;
    case "ampPower":
      ampPower = Number(val);
      break;
    case "verticalFadePower":
      verticalFadePower = Number(val);
      break;
    case "sustainMix":
      // clamp to [0,1]
      sustainMix = Math.max(0, Math.min(1, Number(val)));
      break;
    case "topEnabled":
      topEnabled = !!val;
      break;
    case "topBarSpacing":
      topBarSpacing = Number(val);
      break;
    case "topBarSmoothing":
      topBarSmoothing = Math.max(0, Math.min(1, Number(val)));
      break;
    case "background":
      document.body.style.backgroundColor = val;
      break;
  }
}
window.livelyAudioListener = livelyAudioListener;
window.livelyPropertyListener = livelyPropertyListener;
