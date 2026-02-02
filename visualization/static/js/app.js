// RRT 可视化 - Three.js

let scene, camera, renderer, controls;
let treeGroup, pathGroup, obstacleGroup;
let data = null;

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
    camera.position.set(800, -800, 800);  // X, Y, Z（Z向上）
    camera.up.set(0, 0, 1);  // Z轴向上
    
    // 渲染器
    renderer = new THREE.WebGLRenderer({ antialias: true });
    renderer.setSize(window.innerWidth, window.innerHeight);
    container.appendChild(renderer.domElement);
    
    // 控制器
    controls = new THREE.OrbitControls(camera, renderer.domElement);
    controls.enableDamping = true;
    controls.dampingFactor = 0.05;
    controls.enablePan = true;           // 启用平移（右键）
    controls.panSpeed = 1.0;
    controls.mouseButtons = {
        LEFT: THREE.MOUSE.ROTATE,        // 左键旋转
        MIDDLE: THREE.MOUSE.DOLLY,       // 中键缩放
        RIGHT: THREE.MOUSE.PAN           // 右键平移
    };
    
    // 设置上方向为Z轴（因为数据是Z轴向上）
    controls.up = new THREE.Vector3(0, 0, 1);
    camera.up = new THREE.Vector3(0, 0, 1);
    
    // 光照
    const ambientLight = new THREE.AmbientLight(0xffffff, 0.6);
    scene.add(ambientLight);
    
    const directionalLight = new THREE.DirectionalLight(0xffffff, 0.8);
    directionalLight.position.set(500, 1000, 500);
    scene.add(directionalLight);
    
    // 地面网格（XY平面，Z=0）
    const gridHelper = new THREE.GridHelper(3000, 30, 0x444444, 0x333333);
    gridHelper.rotation.x = Math.PI / 2;  // 旋转使其在XY平面
    gridHelper.position.set(750, 500, 0); // 移动到场景中心
    scene.add(gridHelper);
    
    // 坐标轴
    const axesHelper = new THREE.AxesHelper(200);
    scene.add(axesHelper);
    
    // 组
    treeGroup = new THREE.Group();
    pathGroup = new THREE.Group();
    obstacleGroup = new THREE.Group();
    scene.add(treeGroup);
    scene.add(pathGroup);
    scene.add(obstacleGroup);
    
    // 窗口调整
    window.addEventListener('resize', onWindowResize);
    
    // 加载文件列表，然后加载最新数据
    refreshFileList();
    setTimeout(() => loadLatest(), 500);
    
    // 动画
    animate();
}

function onWindowResize() {
    camera.aspect = window.innerWidth / window.innerHeight;
    camera.updateProjectionMatrix();
    renderer.setSize(window.innerWidth, window.innerHeight);
}

function animate() {
    requestAnimationFrame(animate);
    controls.update();
    renderer.render(scene, camera);
}

// 刷新文件列表
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
                // 解析文件名显示时间
                const parts = f.name.match(/rrt_(\d{4})(\d{2})(\d{2})_(\d{2})(\d{2})(\d{2})_\d+_(ok|fail)\.json/);
                let displayName = f.name;
                if (parts) {
                    const status = parts[7] === 'ok' ? '✓' : '✗';
                    displayName = `${status} ${parts[1]}-${parts[2]}-${parts[3]} ${parts[4]}:${parts[5]}:${parts[6]}`;
                }
                opt.textContent = displayName;
                if (idx === 0) opt.selected = true;
                select.appendChild(opt);
            });
        })
        .catch(err => console.error('Error loading file list:', err));
}

// 加载选中的文件
function loadSelectedFile() {
    const select = document.getElementById('file-select');
    const filename = select.value;
    if (!filename) return;
    loadDataFromFile(filename);
}

// 加载最新文件
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
            
            // 更新下拉框选中项
            const select = document.getElementById('file-select');
            if (data.filename) {
                select.value = data.filename;
            }
            
            updateInfo();
            renderScene();
        })
        .catch(err => {
            document.getElementById('loading').style.display = 'none';
            console.error('Error loading data:', err);
        });
}

// 从指定文件加载数据
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
            
            updateInfo();
            renderScene();
        })
        .catch(err => {
            document.getElementById('loading').style.display = 'none';
            console.error('Error loading data:', err);
        });
}

// 兼容旧的loadData调用
function loadData() {
    loadLatest();
}

// 更新信息面板
function updateInfo() {
    if (!data) return;
    
    document.getElementById('current-file').textContent = data.filename || '-';
    document.getElementById('node-count').textContent = data.nodes ? data.nodes.length : 0;
    document.getElementById('path-count').textContent = data.path ? data.path.length : 0;
    document.getElementById('path-length').textContent = data.path_length ? 
        data.path_length.toFixed(1) + 'm' : '-';
    
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
    }
}

// 渲染场景
function renderScene() {
    // 清空
    clearGroup(treeGroup);
    clearGroup(pathGroup);
    clearGroup(obstacleGroup);
    
    if (!data) return;
    
    // 渲染障碍物
    renderObstacles();
    
    // 渲染RRT树
    renderTree();
    
    // 渲染路径
    renderPath();
    
    // 渲染起点终点
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

// 高度到颜色
function heightToColor(z, zMin, zMax) {
    const t = (z - zMin) / (zMax - zMin + 1);
    // 蓝(低) -> 红(高)
    const r = t;
    const g = 0;
    const b = 1 - t;
    return new THREE.Color(r, g, b);
}

// 渲染RRT树
function renderTree() {
    if (!data.nodes || data.nodes.length === 0) return;
    
    const zMin = data.goal ? data.goal.z : 0;
    const zMax = data.start ? data.start.z : 500;
    const colorByHeight = document.getElementById('color-by-height').checked;
    
    // 节点
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
    
    // 边
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

// 渲染路径
function renderPath() {
    if (!data.path || data.path.length < 2) return;
    
    // 路径线
    const points = data.path.map(p => new THREE.Vector3(p[0], p[1], p[2]));
    const geometry = new THREE.BufferGeometry().setFromPoints(points);
    const material = new THREE.LineBasicMaterial({ 
        color: 0xffff00, 
        linewidth: 3 
    });
    const line = new THREE.Line(geometry, material);
    pathGroup.add(line);
    
    // 路径点
    const sphereGeometry = new THREE.SphereGeometry(8, 16, 16);
    const sphereMaterial = new THREE.MeshBasicMaterial({ color: 0xffff00 });
    
    data.path.forEach((p, idx) => {
        if (idx % 5 === 0) { // 每5个点显示一个球
            const sphere = new THREE.Mesh(sphereGeometry, sphereMaterial);
            sphere.position.set(p[0], p[1], p[2]);
            pathGroup.add(sphere);
        }
    });
}

// 渲染障碍物
function renderObstacles() {
    if (!data.obstacles) return;
    
    data.obstacles.forEach(obs => {
        if (obs.type === 'cylinder') {
            const height = obs.z_max - obs.z_min;
            const geometry = new THREE.CylinderGeometry(
                obs.radius, obs.radius, height, 32
            );
            const material = new THREE.MeshPhongMaterial({ 
                color: 0x666666, 
                transparent: true, 
                opacity: 0.6 
            });
            const cylinder = new THREE.Mesh(geometry, material);
            // Three.js cylinder 默认沿Y轴，需要旋转到Z轴
            cylinder.rotation.x = Math.PI / 2;
            cylinder.position.set(
                obs.center[0], 
                obs.center[1], 
                obs.z_min + height / 2
            );
            obstacleGroup.add(cylinder);
        } else if (obs.type === 'prism') {
            // 棱柱 - 使用BufferGeometry手动构建
            const verts = obs.vertices;
            if (verts && verts.length >= 3) {
                const height = obs.z_max - obs.z_min;
                
                // 创建底面和顶面的顶点
                const positions = [];
                const indices = [];
                const n = verts.length;
                
                // 底面顶点 (0 ~ n-1)
                for (let i = 0; i < n; i++) {
                    positions.push(verts[i][0], verts[i][1], obs.z_min);
                }
                // 顶面顶点 (n ~ 2n-1)
                for (let i = 0; i < n; i++) {
                    positions.push(verts[i][0], verts[i][1], obs.z_max);
                }
                
                // 侧面三角形
                for (let i = 0; i < n; i++) {
                    const i1 = i;
                    const i2 = (i + 1) % n;
                    const i3 = i + n;
                    const i4 = (i + 1) % n + n;
                    indices.push(i1, i2, i3);
                    indices.push(i2, i4, i3);
                }
                
                // 底面和顶面（简单三角扇）
                for (let i = 1; i < n - 1; i++) {
                    indices.push(0, i + 1, i);  // 底面
                    indices.push(n, n + i, n + i + 1);  // 顶面
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

// 渲染起点终点
function renderStartGoal() {
    // 起点 - 绿色
    if (data.start) {
        const geometry = new THREE.SphereGeometry(20, 32, 32);
        const material = new THREE.MeshBasicMaterial({ color: 0x00ff00 });
        const sphere = new THREE.Mesh(geometry, material);
        sphere.position.set(data.start.x, data.start.y, data.start.z);
        pathGroup.add(sphere);
    }
    
    // 终点 - 红色星形
    if (data.goal) {
        const geometry = new THREE.SphereGeometry(25, 32, 32);
        const material = new THREE.MeshBasicMaterial({ color: 0xff0000 });
        const sphere = new THREE.Mesh(geometry, material);
        sphere.position.set(data.goal.x, data.goal.y, data.goal.z);
        pathGroup.add(sphere);
    }
}

// 切换显示
function toggleTree() {
    treeGroup.visible = document.getElementById('show-tree').checked;
}

function togglePath() {
    pathGroup.visible = document.getElementById('show-path').checked;
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

// 启动
init();
