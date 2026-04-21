/* ═══════════════════════════════════════════════════════════
   StereoVision 3D — Interactive Demo
   ═══════════════════════════════════════════════════════════ */

// ── State ──
const state = {
    uploadedImage: null,
    processing: false,
    scene: null, camera: null, renderer: null, pointCloud: null,
    autoRotate: true,
};

// ── Init ──
document.addEventListener('DOMContentLoaded', () => {
    initScrollAnimations();
    initTriangulation();
    initAttentionViz();
    initCounters();
    autoLoadDefaultImage();
});

// ── Scroll Animations ──
function initScrollAnimations() {
    const obs = new IntersectionObserver((entries) => {
        entries.forEach(e => { if (e.isIntersecting) e.target.classList.add('visible'); });
    }, { threshold: 0.1 });
    document.querySelectorAll('.fade-up').forEach(el => obs.observe(el));
}

// ── Auto-load default image on startup ──
function autoLoadDefaultImage() {
    const canvas = document.createElement('canvas');
    canvas.width = 640; canvas.height = 480;
    drawUrbanScene(canvas.getContext('2d'), 640, 480);

    const preview = document.getElementById('auto-preview');
    if (preview) {
        preview.src = canvas.toDataURL('image/png');
    }

    state.uploadedImage = canvas.toDataURL('image/png');
    processImage();
}

async function processImage() {
    if (state.processing || !state.uploadedImage) return;
    state.processing = true;

    const loader = document.getElementById('process-loader');
    loader.classList.add('active');
    document.getElementById('demo-results').style.opacity = '0.3';

    const numDisp = parseInt(document.getElementById('num-disp').value);
    const blockSize = parseInt(document.getElementById('block-size').value);
    const shift = parseInt(document.getElementById('shift').value);

    try {
        const res = await fetch('/api/process', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                image: state.uploadedImage,
                numDisparities: numDisp, blockSize, shift,
            }),
        });
        const data = await res.json();
        document.getElementById('img-left').src = data.left;
        document.getElementById('img-right').src = data.right;
        document.getElementById('img-disparity').src = data.disparity;
        document.getElementById('demo-results').style.opacity = '1';

        if (data.pointcloud && data.pointcloud.points.length > 0) {
            render3DPointCloud(data.pointcloud);
        }
    } catch (err) {
        console.error('Processing error:', err);
    } finally {
        state.processing = false;
        loader.classList.remove('active');
    }
}

// ── 3D Point Cloud Viewer (Three.js) ──
function render3DPointCloud(pc) {
    const container = document.getElementById('viewer-3d');
    container.innerHTML = '';

    const w = container.clientWidth, h = container.clientHeight;
    const scene = new THREE.Scene();
    scene.background = new THREE.Color(0x0a0a0f);
    scene.fog = new THREE.Fog(0x0a0a0f, 30, 80);

    const camera = new THREE.PerspectiveCamera(60, w / h, 0.1, 200);
    camera.position.set(0, 0, 15);

    const renderer = new THREE.WebGLRenderer({ antialias: true });
    renderer.setSize(w, h);
    renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
    container.appendChild(renderer.domElement);

    const controls = new THREE.OrbitControls(camera, renderer.domElement);
    controls.enableDamping = true;
    controls.dampingFactor = 0.05;
    controls.autoRotate = true;
    controls.autoRotateSpeed = 1.2;

    // Build point cloud
    const geo = new THREE.BufferGeometry();
    const pts = pc.points, cols = pc.colors;
    const positions = new Float32Array(pts.length * 3);
    const colors = new Float32Array(pts.length * 3);

    // Normalize points
    let cx = 0, cy = 0, cz = 0;
    pts.forEach(p => { cx += p[0]; cy += p[1]; cz += p[2]; });
    cx /= pts.length; cy /= pts.length; cz /= pts.length;
    let maxR = 0;
    pts.forEach(p => {
        const r = Math.sqrt((p[0]-cx)**2 + (p[1]-cy)**2 + (p[2]-cz)**2);
        if (r > maxR) maxR = r;
    });
    const scale = 10 / (maxR || 1);

    for (let i = 0; i < pts.length; i++) {
        positions[i*3]   = (pts[i][0] - cx) * scale;
        positions[i*3+1] = -(pts[i][1] - cy) * scale;
        positions[i*3+2] = -(pts[i][2] - cz) * scale;
        colors[i*3]   = cols[i][0] / 255;
        colors[i*3+1] = cols[i][1] / 255;
        colors[i*3+2] = cols[i][2] / 255;
    }

    geo.setAttribute('position', new THREE.BufferAttribute(positions, 3));
    geo.setAttribute('color', new THREE.BufferAttribute(colors, 3));

    const mat = new THREE.PointsMaterial({
        size: 0.06, vertexColors: true, sizeAttenuation: true,
        transparent: true, opacity: 0.9,
    });
    const cloud = new THREE.Points(geo, mat);
    scene.add(cloud);
    state.pointCloud = cloud;

    // Ambient grid
    const grid = new THREE.GridHelper(30, 30, 0x1a1a2e, 0x1a1a2e);
    grid.position.y = -6;
    scene.add(grid);

    function animate() {
        requestAnimationFrame(animate);
        controls.update();
        renderer.render(scene, camera);
    }
    animate();

    // Resize
    const ro = new ResizeObserver(() => {
        const nw = container.clientWidth, nh = container.clientHeight;
        camera.aspect = nw / nh;
        camera.updateProjectionMatrix();
        renderer.setSize(nw, nh);
    });
    ro.observe(container);

    // Controls
    document.getElementById('btn-rotate').onclick = () => {
        controls.autoRotate = !controls.autoRotate;
    };
    document.getElementById('btn-reset').onclick = () => {
        camera.position.set(0, 0, 15);
        controls.reset();
    };
    document.getElementById('btn-size-up').onclick = () => {
        mat.size = Math.min(mat.size + 0.02, 0.3);
    };
    document.getElementById('btn-size-down').onclick = () => {
        mat.size = Math.max(mat.size - 0.02, 0.01);
    };
}

// ── Triangulation Playground ──
function initTriangulation() {
    const canvas = document.getElementById('tri-canvas');
    const ctx = canvas.getContext('2d');
    canvas.width = 500; canvas.height = 350;

    const inputs = ['tri-baseline', 'tri-focal', 'tri-disparity'];
    inputs.forEach(id => {
        const el = document.getElementById(id);
        el.addEventListener('input', () => {
            document.getElementById(id + '-val').textContent = el.value;
            updateTriangulation(ctx, canvas);
        });
    });
    updateTriangulation(ctx, canvas);
}

function updateTriangulation(ctx, canvas) {
    const b = parseFloat(document.getElementById('tri-baseline').value);
    const f = parseFloat(document.getElementById('tri-focal').value);
    const d = parseFloat(document.getElementById('tri-disparity').value);
    const z = (b * f) / Math.max(d, 0.01);

    document.getElementById('tri-depth-val').textContent = z.toFixed(2) + 'm';

    // Draw visualization
    const W = canvas.width, H = canvas.height;
    ctx.clearRect(0, 0, W, H);

    // Background
    ctx.fillStyle = '#12121a';
    ctx.fillRect(0, 0, W, H);

    // Grid
    ctx.strokeStyle = 'rgba(255,255,255,0.03)';
    ctx.lineWidth = 1;
    for (let i = 0; i < W; i += 25) { ctx.beginPath(); ctx.moveTo(i, 0); ctx.lineTo(i, H); ctx.stroke(); }
    for (let i = 0; i < H; i += 25) { ctx.beginPath(); ctx.moveTo(0, i); ctx.lineTo(W, i); ctx.stroke(); }

    const camY = H - 60;
    const camLX = W / 2 - b * 200;
    const camRX = W / 2 + b * 200;
    const objX = W / 2;
    const objY = Math.max(40, camY - Math.min(z * 15, camY - 50));

    // Depth lines
    ctx.setLineDash([4, 4]);
    ctx.strokeStyle = 'rgba(108, 92, 231, 0.3)';
    ctx.lineWidth = 1;
    ctx.beginPath(); ctx.moveTo(camLX, camY); ctx.lineTo(objX, objY); ctx.stroke();
    ctx.beginPath(); ctx.moveTo(camRX, camY); ctx.lineTo(objX, objY); ctx.stroke();
    ctx.setLineDash([]);

    // Convergence lines (solid)
    const grad1 = ctx.createLinearGradient(camLX, camY, objX, objY);
    grad1.addColorStop(0, '#6c5ce7'); grad1.addColorStop(1, '#00cec9');
    ctx.strokeStyle = grad1; ctx.lineWidth = 2;
    ctx.beginPath(); ctx.moveTo(camLX, camY); ctx.lineTo(objX, objY); ctx.stroke();

    const grad2 = ctx.createLinearGradient(camRX, camY, objX, objY);
    grad2.addColorStop(0, '#fd79a8'); grad2.addColorStop(1, '#00cec9');
    ctx.strokeStyle = grad2;
    ctx.beginPath(); ctx.moveTo(camRX, camY); ctx.lineTo(objX, objY); ctx.stroke();

    // Baseline
    ctx.strokeStyle = 'rgba(255,255,255,0.2)'; ctx.lineWidth = 1;
    ctx.setLineDash([3, 3]);
    ctx.beginPath(); ctx.moveTo(camLX, camY); ctx.lineTo(camRX, camY); ctx.stroke();
    ctx.setLineDash([]);

    ctx.fillStyle = '#888'; ctx.font = '11px JetBrains Mono';
    ctx.textAlign = 'center';
    ctx.fillText(`b = ${b.toFixed(2)}m`, W / 2, camY + 20);

    // Depth label
    ctx.save();
    ctx.translate(W - 40, (camY + objY) / 2);
    ctx.rotate(-Math.PI / 2);
    ctx.fillStyle = '#00cec9'; ctx.font = '12px JetBrains Mono';
    ctx.fillText(`Z = ${z.toFixed(2)}m`, 0, 0);
    ctx.restore();

    // Cameras
    drawCamera(ctx, camLX, camY, '#6c5ce7', 'L');
    drawCamera(ctx, camRX, camY, '#fd79a8', 'R');

    // Object point
    ctx.beginPath();
    ctx.arc(objX, objY, 8, 0, Math.PI * 2);
    ctx.fillStyle = '#00cec9';
    ctx.shadowColor = '#00cec9'; ctx.shadowBlur = 20;
    ctx.fill();
    ctx.shadowBlur = 0;

    ctx.fillStyle = '#fff'; ctx.font = 'bold 11px JetBrains Mono';
    ctx.fillText('P', objX, objY - 16);
}

function drawCamera(ctx, x, y, color, label) {
    ctx.fillStyle = color;
    ctx.shadowColor = color; ctx.shadowBlur = 12;
    ctx.beginPath();
    ctx.moveTo(x - 12, y + 8);
    ctx.lineTo(x + 12, y + 8);
    ctx.lineTo(x + 8, y - 8);
    ctx.lineTo(x - 8, y - 8);
    ctx.closePath();
    ctx.fill();
    ctx.shadowBlur = 0;

    ctx.fillStyle = '#fff'; ctx.font = 'bold 11px Inter';
    ctx.textAlign = 'center';
    ctx.fillText(label, x, y + 24);
}

// ── Cross-Attention Visualization ──
function initAttentionViz() {
    animateAttentionMap('attn-canvas-1', 'query');
    animateAttentionMap('attn-canvas-2', 'attention');
}

function animateAttentionMap(canvasId, type) {
    const canvas = document.getElementById(canvasId);
    if (!canvas) return;
    const ctx = canvas.getContext('2d');
    canvas.width = 400; canvas.height = 300;

    const cols = 16, rows = 12;
    const cw = canvas.width / cols, ch = canvas.height / rows;
    let frame = 0;

    function draw() {
        ctx.clearRect(0, 0, canvas.width, canvas.height);
        ctx.fillStyle = '#0a0a0f';
        ctx.fillRect(0, 0, canvas.width, canvas.height);

        for (let r = 0; r < rows; r++) {
            for (let c = 0; c < cols; c++) {
                let val;
                if (type === 'query') {
                    val = Math.sin(c * 0.4 + frame * 0.03) *
                          Math.cos(r * 0.5 + frame * 0.02) * 0.5 + 0.5;
                } else {
                    const cx2 = cols / 2 + Math.sin(frame * 0.02) * 4;
                    const cy2 = rows / 2 + Math.cos(frame * 0.015) * 3;
                    const dist = Math.sqrt((c - cx2) ** 2 + (r - cy2) ** 2);
                    val = Math.exp(-dist * 0.3) * (Math.sin(frame * 0.05) * 0.3 + 0.7);
                }
                val = Math.max(0, Math.min(1, val));

                const hue = type === 'query' ? 260 : 175;
                ctx.fillStyle = `hsla(${hue}, 80%, ${20 + val * 50}%, ${0.3 + val * 0.7})`;
                ctx.fillRect(c * cw + 1, r * ch + 1, cw - 2, ch - 2);

                if (val > 0.7) {
                    ctx.fillStyle = `hsla(${hue}, 90%, 70%, ${val * 0.5})`;
                    ctx.shadowColor = `hsla(${hue}, 90%, 60%, 0.5)`;
                    ctx.shadowBlur = 8;
                    ctx.fillRect(c * cw + 2, r * ch + 2, cw - 4, ch - 4);
                    ctx.shadowBlur = 0;
                }
            }
        }

        // Epipolar line
        const ey = Math.floor(rows / 2 + Math.sin(frame * 0.01) * 2);
        ctx.strokeStyle = 'rgba(253, 121, 168, 0.6)';
        ctx.lineWidth = 1; ctx.setLineDash([4, 4]);
        ctx.beginPath();
        ctx.moveTo(0, ey * ch + ch / 2);
        ctx.lineTo(canvas.width, ey * ch + ch / 2);
        ctx.stroke();
        ctx.setLineDash([]);

        frame++;
        requestAnimationFrame(draw);
    }
    draw();
}

// ── Counter Animation ──
function initCounters() {
    const counters = document.querySelectorAll('.stat-value[data-target]');
    const obs = new IntersectionObserver((entries) => {
        entries.forEach(e => {
            if (e.isIntersecting) {
                animateCounter(e.target);
                obs.unobserve(e.target);
            }
        });
    }, { threshold: 0.5 });
    counters.forEach(c => obs.observe(c));
}

function animateCounter(el) {
    const target = parseFloat(el.dataset.target);
    const suffix = el.dataset.suffix || '';
    const duration = 2000;
    const start = performance.now();

    function update(now) {
        const t = Math.min((now - start) / duration, 1);
        const ease = 1 - Math.pow(1 - t, 3);
        const val = target * ease;
        el.textContent = (Number.isInteger(target) ? Math.floor(val) : val.toFixed(1)) + suffix;
        if (t < 1) requestAnimationFrame(update);
    }
    requestAnimationFrame(update);
}

// ── Generate Sample Procedural Images ──
function generateSampleImages() {
    const container = document.getElementById('sample-images');
    if (!container) return;

    const samples = [
        { name: 'Urban Scene', fn: drawUrbanScene },
        { name: 'Geometric', fn: drawGeometric },
        { name: 'Depth Layers', fn: drawDepthLayers },
        { name: 'Corridor', fn: drawCorridor },
    ];

    samples.forEach(s => {
        const canvas = document.createElement('canvas');
        canvas.width = 640; canvas.height = 480;
        canvas.className = 'sample-thumb';
        canvas.title = s.name;
        s.fn(canvas.getContext('2d'), canvas.width, canvas.height);
        canvas.addEventListener('click', () => loadSampleImage(canvas));
        container.appendChild(canvas);
    });
}

function drawUrbanScene(ctx, w, h) {
    // Sky gradient
    const sky = ctx.createLinearGradient(0, 0, 0, h * 0.6);
    sky.addColorStop(0, '#1a1a3e'); sky.addColorStop(1, '#2d1b69');
    ctx.fillStyle = sky; ctx.fillRect(0, 0, w, h * 0.6);

    // Ground
    const gnd = ctx.createLinearGradient(0, h * 0.6, 0, h);
    gnd.addColorStop(0, '#2a2a3a'); gnd.addColorStop(1, '#1a1a2a');
    ctx.fillStyle = gnd; ctx.fillRect(0, h * 0.6, w, h * 0.4);

    // Buildings
    const buildings = [
        { x: 50, w: 100, h: 250, c: '#3a3a5a' },
        { x: 170, w: 80, h: 320, c: '#4a4a6a' },
        { x: 270, w: 120, h: 200, c: '#353555' },
        { x: 410, w: 90, h: 280, c: '#454565' },
        { x: 520, w: 110, h: 230, c: '#3d3d5d' },
    ];
    buildings.forEach(b => {
        ctx.fillStyle = b.c;
        ctx.fillRect(b.x, h * 0.6 - b.h, b.w, b.h);
        // Windows
        ctx.fillStyle = 'rgba(255, 220, 100, 0.6)';
        for (let wy = h * 0.6 - b.h + 20; wy < h * 0.6 - 20; wy += 30) {
            for (let wx = b.x + 10; wx < b.x + b.w - 15; wx += 20) {
                if (Math.random() > 0.3) ctx.fillRect(wx, wy, 10, 15);
            }
        }
    });

    // Road lines
    ctx.strokeStyle = '#ffeaa7'; ctx.lineWidth = 3; ctx.setLineDash([20, 15]);
    ctx.beginPath(); ctx.moveTo(0, h * 0.8); ctx.lineTo(w, h * 0.8); ctx.stroke();
    ctx.setLineDash([]);
}

function drawGeometric(ctx, w, h) {
    ctx.fillStyle = '#0f0f1a'; ctx.fillRect(0, 0, w, h);
    const shapes = 40;
    for (let i = 0; i < shapes; i++) {
        const x = Math.random() * w, y = Math.random() * h;
        const size = 20 + Math.random() * 80;
        const depth = i / shapes;
        const hue = 200 + depth * 160;
        ctx.fillStyle = `hsla(${hue}, 70%, ${30 + depth * 30}%, ${0.4 + depth * 0.4})`;
        ctx.strokeStyle = `hsla(${hue}, 80%, 60%, 0.3)`;
        ctx.lineWidth = 1;
        if (i % 3 === 0) {
            ctx.fillRect(x, y, size, size * 0.8);
            ctx.strokeRect(x, y, size, size * 0.8);
        } else if (i % 3 === 1) {
            ctx.beginPath(); ctx.arc(x, y, size / 2, 0, Math.PI * 2);
            ctx.fill(); ctx.stroke();
        } else {
            ctx.beginPath();
            ctx.moveTo(x, y - size / 2);
            ctx.lineTo(x + size / 2, y + size / 2);
            ctx.lineTo(x - size / 2, y + size / 2);
            ctx.closePath(); ctx.fill(); ctx.stroke();
        }
    }
}

function drawDepthLayers(ctx, w, h) {
    const layers = [
        { y: 0.9, color: '#1a1a3e', objects: 8 },
        { y: 0.7, color: '#2a2a5e', objects: 6 },
        { y: 0.5, color: '#3a3a7e', objects: 4 },
        { y: 0.3, color: '#4a4a9e', objects: 3 },
    ];
    ctx.fillStyle = '#0a0a1a'; ctx.fillRect(0, 0, w, h);

    layers.forEach((l, li) => {
        ctx.fillStyle = l.color;
        ctx.fillRect(0, h * l.y, w, h * 0.25);
        for (let i = 0; i < l.objects; i++) {
            const ox = (w / (l.objects + 1)) * (i + 1);
            const oy = h * l.y + 10;
            const s = 30 + li * 15;
            ctx.fillStyle = `hsl(${260 + li * 30}, 60%, ${35 + li * 10}%)`;
            ctx.fillRect(ox - s / 2, oy, s, s * 1.5);
        }
    });
}

function drawCorridor(ctx, w, h) {
    ctx.fillStyle = '#0f0f1a'; ctx.fillRect(0, 0, w, h);
    const cx = w / 2, cy = h / 2;

    for (let i = 12; i >= 0; i--) {
        const t = i / 12;
        const s = t * 0.45;
        const x1 = cx - w * s, y1 = cy - h * s;
        const rw = w * s * 2, rh = h * s * 2;
        const hue = 260 - i * 8;
        ctx.strokeStyle = `hsla(${hue}, 70%, ${40 + i * 3}%, ${0.3 + t * 0.5})`;
        ctx.lineWidth = 2;
        ctx.strokeRect(x1, y1, rw, rh);

        // Floor lines
        if (i < 10) {
            ctx.strokeStyle = `rgba(108, 92, 231, ${0.1 + t * 0.2})`;
            ctx.lineWidth = 1;
            const fy = cy + h * s;
            ctx.beginPath(); ctx.moveTo(x1, fy); ctx.lineTo(x1 + rw, fy); ctx.stroke();
        }
    }

    // Vanishing point glow
    const glow = ctx.createRadialGradient(cx, cy, 0, cx, cy, 60);
    glow.addColorStop(0, 'rgba(0, 206, 201, 0.4)');
    glow.addColorStop(1, 'transparent');
    ctx.fillStyle = glow;
    ctx.fillRect(cx - 60, cy - 60, 120, 120);
}

// ── Slider Updates ──
function updateSlider(id) {
    const el = document.getElementById(id);
    document.getElementById(id + '-val').textContent = el.value;
}
