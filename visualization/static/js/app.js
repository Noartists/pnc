// RRT/仿真 可视化 - Three.js

let scene, camera, renderer, controls;
let treeGroup, pathGroup, smoothedPathGroup, actualPathGroup, obstacleGroup;
let aircraftMesh = null;  // 飞行器模型
let data = null;

// 播放状态
let isPlaying = false;
let currentTime = 0;
let totalDuration = 0;
let playbackSpeed = 1;
let animationId = null;
let lastFrameTime = 0;

// 初始化
function init() {
    const container = document.getElementById('container');

    // 场景
    scene = new THREE.Scene();
    scene.background = new THREE.Color(0x1a1a2e);

    // 相机（从斜上方看，Z轴向上）
    camera = new THREE.PerspectiveCamera(
        60,
        window.innerWidth / window.innerHeight,
        1,
        10000
    );
    camera.position.set(800, -800, 800);
    camera.up.set(0, 0, 1);

    // 渲染器
    renderer = new THREE.WebGLRenderer({ antialias: true });
    renderer.setSize(window.innerWidth, window.innerHeight);
    container.appendChild(renderer.domElement);

    // 控制器
    controls = new THREE.OrbitControls(camera, renderer.domElement);
    controls.enableDamping = true;
    controls.dampingFactor = 0.05;
    controls.enablePan = true;
    controls.panSpeed = 1.0;
    controls.mouseButtons = {
        LEFT: THREE.MOUSE.ROTATE,
        MIDDLE: THREE.MOUSE.DOLLY,
        RIGHT: THREE.MOUSE.PAN
    };
    controls.up = new THREE.Vector3(0, 0, 1);
    camera.up = new THREE.Vector3(0, 0, 1);

    // 光照
    const ambientLight = new THREE.AmbientLight(0xffffff, 0.6);
    scene.add(ambientLight);

    const directionalLight = new THREE.DirectionalLight(0xffffff, 0.8);
    directionalLight.position.set(500, 1000, 500);
    scene.add(directionalLight);

    // 地面网格
    const gridHelper = new THREE.GridHelper(3000, 30, 0x444444, 0x333333);
    gridHelper.rotation.x = Math.PI / 2;
    gridHelper.position.set(750, 500, 0);
    scene.add(gridHelper);

    // 坐标轴
    const axesHelper = new THREE.AxesHelper(200);
    scene.add(axesHelper);

    // 组
    treeGroup = new THREE.Group();
    pathGroup = new THREE.Group();
    smoothedPathGroup = new THREE.Group();
    actualPathGroup = new THREE.Group();
    obstacleGroup = new THREE.Group();
    scene.add(treeGroup);
    scene.add(pathGroup);
    scene.add(smoothedPathGroup);
    scene.add(actualPathGroup);
    scene.add(obstacleGroup);

    // 创建飞行器模型
    createAircraftMesh();

    // 窗口调整
    window.addEventListener('resize', onWindowResize);

    // 键盘控制
    window.addEventListener('keydown', onKeyDown);

    // 加载文件列表，然后加载最新数据
    refreshFileList();
    setTimeout(() => loadLatest(), 500);

    // 动画
    animate();
}

function createAircraftMesh() {
    // 创建一个简单的飞行器模型（锥形+翅膀）
    const group = new THREE.Group();

    // 机身 - 锥形
    const bodyGeom = new THREE.ConeGeometry(8, 30, 8);
    const bodyMat = new THREE.MeshPhongMaterial({ color: 0xff6600 });
    const body = new THREE.Mesh(bodyGeom, bodyMat);
    body.rotation.x = Math.PI / 2;
    group.add(body);

    // 翼伞 - 半圆形
    const wingGeom = new THREE.SphereGeometry(25, 16, 8, 0, Math.PI * 2, 0, Math.PI / 2);
    const wingMat = new THREE.MeshPhongMaterial({
        color: 0x4fc3f7,
        transparent: true,
        opacity: 0.7,
        side: THREE.DoubleSide
    });
    const wing = new THREE.Mesh(wingGeom, wingMat);
    wing.position.z = 15;
    wing.rotation.x = Math.PI;
    group.add(wing);

    aircraftMesh = group;
    aircraftMesh.visible = false;
    scene.add(aircraftMesh);
}

function onWindowResize() {
    camera.aspect = window.innerWidth / window.innerHeight;
    camera.updateProjectionMatrix();
    renderer.setSize(window.innerWidth, window.innerHeight);
}

function onKeyDown(event) {
    if (event.code === 'Space') {
        event.preventDefault();
        togglePlay();
    } else if (event.code === 'ArrowRight') {
        stepForward();
    } else if (event.code === 'ArrowLeft') {
        stepBackward();
    }
}

function animate() {
    requestAnimationFrame(animate);
    controls.update();

    // 播放动画
    if (isPlaying && data && data.data_type === 'simulation') {
        const now = performance.now();
        const delta = (now - lastFrameTime) / 1000 * playbackSpeed;
        lastFrameTime = now;

        currentTime += delta;
        if (currentTime >= totalDuration) {
            currentTime = totalDuration;
            pausePlayback();
        }
        updatePlaybackUI();
        updateAircraftPosition();
    }

    renderer.render(scene, camera);
}

// ============================================================
//                     文件加载
// ============================================================

function refreshFileList() {
    fetch('/api/files')
        .then(response => response.json())
        .then(d => {
            const select = document.getElementById('file-select');
            select.innerHTML = '';

            if (d.files.length === 0) {
                select.innerHTML = '<option value="">-- 无数据文件 --</option>';
                return;
            }

            d.files.forEach((f, idx) => {
                const opt = document.createElement('option');
                opt.value = f.name;

                // 显示文件类型标识
                const typeIcon = f.type === 'simulation' ? '🎮' : '🌲';
                let displayName = f.name;

                // 解析文件名
                if (f.type === 'simulation') {
                    const parts = f.name.match(/sim_(\d{4})(\d{2})(\d{2})_(\d{2})(\d{2})(\d{2})\.json/);
                    if (parts) {
                        displayName = `${typeIcon} ${parts[1]}-${parts[2]}-${parts[3]} ${parts[4]}:${parts[5]}`;
                    }
                } else {
                    const parts = f.name.match(/rrt_(\d{4})(\d{2})(\d{2})_(\d{2})(\d{2})(\d{2})_\d+_(ok|fail)\.json/);
                    if (parts) {
                        const status = parts[7] === 'ok' ? '✓' : '✗';
                        displayName = `${typeIcon}${status} ${parts[1]}-${parts[2]}-${parts[3]} ${parts[4]}:${parts[5]}`;
                    }
                }

                opt.textContent = displayName;
                if (idx === 0) opt.selected = true;
                select.appendChild(opt);
            });
        })
        .catch(err => console.error('Error loading file list:', err));
}

function loadSelectedFile() {
    const select = document.getElementById('file-select');
    const filename = select.value;
    if (!filename) return;
    loadDataFromFile(filename);
}

function loadLatest() {
    document.getElementById('loading').style.display = 'block';

    fetch('/api/latest')
        .then(response => response.json())
        .then(d => {
            data = d;
            document.getElementById('loading').style.display = 'none';

            if (data.error) {
                alert(data.error);
                return;
            }

            const select = document.getElementById('file-select');
            if (data.filename) {
                select.value = data.filename;
            }

            onDataLoaded();
        })
        .catch(err => {
            document.getElementById('loading').style.display = 'none';
            console.error('Error loading data:', err);
        });
}

function loadDataFromFile(filename) {
    document.getElementById('loading').style.display = 'block';

    fetch('/api/data/' + filename)
        .then(response => response.json())
        .then(d => {
            data = d;
            document.getElementById('loading').style.display = 'none';

            if (data.error) {
                alert(data.error);
                return;
            }

            onDataLoaded();
        })
        .catch(err => {
            document.getElementById('loading').style.display = 'none';
            console.error('Error loading data:', err);
        });
}

function onDataLoaded() {
    // 停止播放
    pausePlayback();
    currentTime = 0;

    // 更新UI
    updateInfo();
    renderScene();

    // 显示/隐藏播放控件
    const playbackPanel = document.getElementById('playback-panel');
    if (data.data_type === 'simulation') {
        playbackPanel.classList.add('visible');
        totalDuration = data.info ? data.info.duration : 0;
        updatePlaybackUI();
        // 显示飞行器
        aircraftMesh.visible = true;
        updateAircraftPosition();
    } else {
        playbackPanel.classList.remove('visible');
        aircraftMesh.visible = false;
    }
}

// ============================================================
//                     信息面板
// ============================================================

function updateInfo() {
    if (!data) return;

    document.getElementById('current-file').textContent = data.filename || '-';

    // 数据类型标识
    const badge = document.getElementById('data-type-badge');
    const rrtInfo = document.getElementById('rrt-info');
    const simInfo = document.getElementById('sim-info');

    if (data.data_type === 'simulation') {
        badge.textContent = '仿真';
        badge.className = 'data-type-badge badge-sim';
        rrtInfo.style.display = 'none';
        simInfo.style.display = 'block';

        // 仿真信息
        document.getElementById('sim-duration').textContent =
            data.info ? data.info.duration.toFixed(1) + 's' : '-';
        document.getElementById('ref-traj-count').textContent =
            data.reference_trajectory ? data.reference_trajectory.length + ' 点' : '-';
        document.getElementById('actual-traj-count').textContent =
            data.actual_trajectory ? data.actual_trajectory.length + ' 点' : '-';
    } else {
        badge.textContent = 'RRT';
        badge.className = 'data-type-badge badge-rrt';
        rrtInfo.style.display = 'block';
        simInfo.style.display = 'none';

        // RRT 信息
        document.getElementById('node-count').textContent = data.nodes ? data.nodes.length : 0;
        document.getElementById('path-count').textContent = data.path ? data.path.length : 0;
        document.getElementById('path-length').textContent = data.path_length ?
            data.path_length.toFixed(1) + 'm' : '-';

        // 平滑轨迹信息
        const smoothedInfo = document.getElementById('smoothed-info');
        if (data.smoothed_path && data.smoothed_path.length > 0) {
            const info = data.smoothed_info || {};
            smoothedInfo.textContent = `${data.smoothed_path.length} 点, ${(info.duration || 0).toFixed(1)}s`;
            smoothedInfo.style.color = '#00ffff';
        } else {
            smoothedInfo.textContent = '无';
            smoothedInfo.style.color = '#888';
        }
    }

    // 通用信息
    if (data.start) {
        document.getElementById('start-pos').textContent =
            `(${data.start.x.toFixed(0)}, ${data.start.y.toFixed(0)}, ${data.start.z.toFixed(0)})`;
    }
    if (data.goal) {
        document.getElementById('goal-pos').textContent =
            `(${data.goal.x.toFixed(0)}, ${data.goal.y.toFixed(0)}, ${data.goal.z.toFixed(0)})`;
    }
    if (data.config) {
        document.getElementById('glide-range').textContent =
            `${data.config.min_glide.toFixed(2)} ~ ${data.config.max_glide.toFixed(2)}`;
    } else {
        document.getElementById('glide-range').textContent = '-';
    }
}

// ============================================================
//                     场景渲染
// ============================================================

function renderScene() {
    clearGroup(treeGroup);
    clearGroup(pathGroup);
    clearGroup(smoothedPathGroup);
    clearGroup(actualPathGroup);
    clearGroup(obstacleGroup);

    if (!data) return;

    renderObstacles();
    renderTree();
    renderPath();
    renderSmoothedPath();
    renderActualPath();
    renderStartGoal();

    // 调整相机
    if (data.start && data.goal) {
        const cx = (data.start.x + data.goal.x) / 2;
        const cy = (data.start.y + data.goal.y) / 2;
        controls.target.set(cx, cy, 200);
    }
}

function clearGroup(group) {
    while (group.children.length > 0) {
        group.remove(group.children[0]);
    }
}

function heightToColor(z, zMin, zMax) {
    const t = (z - zMin) / (zMax - zMin + 1);
    return new THREE.Color(t, 0, 1 - t);
}

function renderTree() {
    if (!data.nodes || data.nodes.length === 0) return;

    const zMin = data.goal ? data.goal.z : 0;
    const zMax = data.start ? data.start.z : 500;
    const colorByHeight = document.getElementById('color-by-height').checked;

    const nodeGeometry = new THREE.SphereGeometry(5, 8, 8);

    data.nodes.forEach((node, idx) => {
        const color = colorByHeight ?
            heightToColor(node.z, zMin, zMax) :
            new THREE.Color(0x4fc3f7);

        const material = new THREE.MeshBasicMaterial({ color: color });
        const sphere = new THREE.Mesh(nodeGeometry, material);
        sphere.position.set(node.x, node.y, node.z);
        sphere.userData = { index: idx, ...node };
        treeGroup.add(sphere);
    });

    const edgeMaterial = new THREE.LineBasicMaterial({
        color: 0x333366,
        transparent: true,
        opacity: 0.3
    });

    data.nodes.forEach((node, idx) => {
        if (node.parent_idx !== null && node.parent_idx !== undefined) {
            const parent = data.nodes[node.parent_idx];
            if (parent) {
                const points = [
                    new THREE.Vector3(parent.x, parent.y, parent.z),
                    new THREE.Vector3(node.x, node.y, node.z)
                ];
                const geometry = new THREE.BufferGeometry().setFromPoints(points);
                const line = new THREE.Line(geometry, edgeMaterial);
                treeGroup.add(line);
            }
        }
    });
}

function renderPath() {
    if (!data.path || data.path.length < 2) return;

    const points = data.path.map(p => new THREE.Vector3(p[0], p[1], p[2]));
    const geometry = new THREE.BufferGeometry().setFromPoints(points);
    const material = new THREE.LineBasicMaterial({ color: 0xffff00, linewidth: 3 });
    const line = new THREE.Line(geometry, material);
    pathGroup.add(line);

    const sphereGeometry = new THREE.SphereGeometry(8, 16, 16);
    const sphereMaterial = new THREE.MeshBasicMaterial({ color: 0xffff00 });

    data.path.forEach((p, idx) => {
        if (idx % 5 === 0) {
            const sphere = new THREE.Mesh(sphereGeometry, sphereMaterial);
            sphere.position.set(p[0], p[1], p[2]);
            pathGroup.add(sphere);
        }
    });
}

function renderSmoothedPath() {
    // 支持两种格式：smoothed_path（RRT数据）和 reference_trajectory（仿真数据）
    let trajData = data.smoothed_path || data.reference_trajectory;
    if (!trajData || trajData.length < 2) return;

    const points = trajData.map(p => new THREE.Vector3(p.x, p.y, p.z));
    const geometry = new THREE.BufferGeometry().setFromPoints(points);
    const material = new THREE.LineBasicMaterial({ color: 0x00ffff, linewidth: 2 });
    const line = new THREE.Line(geometry, material);
    smoothedPathGroup.add(line);

    // 显示少量点
    const step = Math.max(1, Math.floor(trajData.length / 50));
    const sphereGeometry = new THREE.SphereGeometry(4, 8, 8);
    const sphereMaterial = new THREE.MeshBasicMaterial({ color: 0x00ffff });

    trajData.forEach((p, idx) => {
        if (idx % step === 0) {
            const sphere = new THREE.Mesh(sphereGeometry, sphereMaterial);
            sphere.position.set(p.x, p.y, p.z);
            smoothedPathGroup.add(sphere);
        }
    });
}

function renderActualPath() {
    if (!data.actual_trajectory || data.actual_trajectory.length < 2) return;

    const points = data.actual_trajectory.map(p => new THREE.Vector3(p.x, p.y, p.z));
    const geometry = new THREE.BufferGeometry().setFromPoints(points);
    const material = new THREE.LineBasicMaterial({ color: 0xff00ff, linewidth: 2 });
    const line = new THREE.Line(geometry, material);
    actualPathGroup.add(line);

    // 显示少量点
    const step = Math.max(1, Math.floor(data.actual_trajectory.length / 30));
    const sphereGeometry = new THREE.SphereGeometry(5, 8, 8);
    const sphereMaterial = new THREE.MeshBasicMaterial({ color: 0xff00ff });

    data.actual_trajectory.forEach((p, idx) => {
        if (idx % step === 0) {
            const sphere = new THREE.Mesh(sphereGeometry, sphereMaterial);
            sphere.position.set(p.x, p.y, p.z);
            actualPathGroup.add(sphere);
        }
    });
}

function renderObstacles() {
    if (!data.obstacles) return;

    data.obstacles.forEach(obs => {
        if (obs.type === 'cylinder') {
            const height = obs.z_max - obs.z_min;
            const geometry = new THREE.CylinderGeometry(obs.radius, obs.radius, height, 32);
            const material = new THREE.MeshPhongMaterial({
                color: 0x666666,
                transparent: true,
                opacity: 0.6
            });
            const cylinder = new THREE.Mesh(geometry, material);
            cylinder.rotation.x = Math.PI / 2;
            cylinder.position.set(obs.center[0], obs.center[1], obs.z_min + height / 2);
            obstacleGroup.add(cylinder);
        } else if (obs.type === 'prism') {
            const verts = obs.vertices;
            if (verts && verts.length >= 3) {
                const height = obs.z_max - obs.z_min;
                const positions = [];
                const indices = [];
                const n = verts.length;

                for (let i = 0; i < n; i++) {
                    positions.push(verts[i][0], verts[i][1], obs.z_min);
                }
                for (let i = 0; i < n; i++) {
                    positions.push(verts[i][0], verts[i][1], obs.z_max);
                }

                for (let i = 0; i < n; i++) {
                    const i1 = i;
                    const i2 = (i + 1) % n;
                    const i3 = i + n;
                    const i4 = (i + 1) % n + n;
                    indices.push(i1, i2, i3);
                    indices.push(i2, i4, i3);
                }

                for (let i = 1; i < n - 1; i++) {
                    indices.push(0, i + 1, i);
                    indices.push(n, n + i, n + i + 1);
                }

                const geometry = new THREE.BufferGeometry();
                geometry.setAttribute('position', new THREE.Float32BufferAttribute(positions, 3));
                geometry.setIndex(indices);
                geometry.computeVertexNormals();

                const material = new THREE.MeshPhongMaterial({
                    color: 0xcc8800,
                    transparent: true,
                    opacity: 0.6,
                    side: THREE.DoubleSide
                });
                const mesh = new THREE.Mesh(geometry, material);
                obstacleGroup.add(mesh);
            }
        }
    });
}

function renderStartGoal() {
    if (data.start) {
        const geometry = new THREE.SphereGeometry(20, 32, 32);
        const material = new THREE.MeshBasicMaterial({ color: 0x00ff00 });
        const sphere = new THREE.Mesh(geometry, material);
        sphere.position.set(data.start.x, data.start.y, data.start.z);
        pathGroup.add(sphere);
    }

    if (data.goal) {
        const geometry = new THREE.SphereGeometry(25, 32, 32);
        const material = new THREE.MeshBasicMaterial({ color: 0xff0000 });
        const sphere = new THREE.Mesh(geometry, material);
        sphere.position.set(data.goal.x, data.goal.y, data.goal.z);
        pathGroup.add(sphere);
    }
}

// ============================================================
//                     播放控制
// ============================================================

function togglePlay() {
    if (isPlaying) {
        pausePlayback();
    } else {
        startPlayback();
    }
}

function startPlayback() {
    if (!data || data.data_type !== 'simulation') return;
    if (currentTime >= totalDuration) {
        currentTime = 0;
    }
    isPlaying = true;
    lastFrameTime = performance.now();
    document.getElementById('play-btn').textContent = '⏸';
}

function pausePlayback() {
    isPlaying = false;
    document.getElementById('play-btn').textContent = '▶';
}

function stepForward() {
    if (!data || data.data_type !== 'simulation') return;
    pausePlayback();
    const step = totalDuration / (data.actual_trajectory ? data.actual_trajectory.length : 100);
    currentTime = Math.min(currentTime + step * 10, totalDuration);
    updatePlaybackUI();
    updateAircraftPosition();
}

function stepBackward() {
    if (!data || data.data_type !== 'simulation') return;
    pausePlayback();
    const step = totalDuration / (data.actual_trajectory ? data.actual_trajectory.length : 100);
    currentTime = Math.max(currentTime - step * 10, 0);
    updatePlaybackUI();
    updateAircraftPosition();
}

function seekProgress(event) {
    if (!data || data.data_type !== 'simulation') return;
    const bar = document.getElementById('progress-bar');
    const rect = bar.getBoundingClientRect();
    const ratio = (event.clientX - rect.left) / rect.width;
    currentTime = ratio * totalDuration;
    updatePlaybackUI();
    updateAircraftPosition();
}

function setPlaybackSpeed() {
    playbackSpeed = parseFloat(document.getElementById('speed-select').value);
}

function updatePlaybackUI() {
    const progress = totalDuration > 0 ? (currentTime / totalDuration * 100) : 0;
    document.getElementById('progress-fill').style.width = progress + '%';
    document.getElementById('time-display').textContent =
        `${currentTime.toFixed(1)}s / ${totalDuration.toFixed(1)}s`;
}

function updateAircraftPosition() {
    if (!aircraftMesh || !data || !data.actual_trajectory) return;

    // 找到当前时间对应的位置
    const traj = data.actual_trajectory;
    let idx = 0;
    for (let i = 0; i < traj.length; i++) {
        if (traj[i].t >= currentTime) {
            idx = i;
            break;
        }
        idx = i;
    }

    const pt = traj[idx];
    if (pt) {
        aircraftMesh.position.set(pt.x, pt.y, pt.z);

        // 设置朝向
        if (pt.yaw !== undefined) {
            aircraftMesh.rotation.z = pt.yaw;
        }
        if (pt.pitch !== undefined) {
            aircraftMesh.rotation.x = pt.pitch;
        }
        if (pt.roll !== undefined) {
            aircraftMesh.rotation.y = -pt.roll;
        }
    }
}

// ============================================================
//                     显示切换
// ============================================================

function toggleTree() {
    treeGroup.visible = document.getElementById('show-tree').checked;
}

function togglePath() {
    pathGroup.visible = document.getElementById('show-path').checked;
}

function toggleSmoothedPath() {
    smoothedPathGroup.visible = document.getElementById('show-smoothed').checked;
}

function toggleActualPath() {
    actualPathGroup.visible = document.getElementById('show-actual').checked;
}

function toggleObstacles() {
    obstacleGroup.visible = document.getElementById('show-obstacles').checked;
}

function updateNodeColors() {
    renderTree();
}

function resetCamera() {
    camera.position.set(800, -800, 800);
    camera.up.set(0, 0, 1);
    controls.target.set(750, 500, 200);
    controls.update();
}

// 兼容旧接口
function loadData() {
    loadLatest();
}

// 启动
init();
