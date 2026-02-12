import os
import json
import socket
from flask import Flask, render_template_string, request, jsonify, send_file

# ================= 1. 配置区域 =================
# JSON 文件绝对路径
JSON_PATH = '/Users/10294814/task/cldfeed/abner/data_clean/data_cldfeed/cld_bottle_black_audit_datas/cld_bottle_black_audit_datas.json'

# 图片文件夹绝对路径
IMG_DIR = '/Users/10294814/task/cldfeed/abner/data_clean/data_cldfeed/cld_bottle_black_audit_datas/cld_bottle_black_audit_datas_download'

# 定义标签类别
LABELS = [
    "cld_bottle_clip",
    "cld_bottle_pillow",
    "cld_bottle_straw",
    "white",
    "uncertain",
    "black",
]
# ==============================================

app = Flask(__name__)

# 前端 HTML/CSS/JS 模板
HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="zh-CN">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>图片数据标注工具</title>
    <style>
        body { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Helvetica, Arial, sans-serif; background-color: #f4f6f8; margin: 0; padding: 20px; }
        
        /* 顶部悬浮状态栏 */
        .header {
            position: sticky; top: 0; z-index: 1000;
            background: rgba(255, 255, 255, 0.95);
            backdrop-filter: blur(10px);
            padding: 15px 20px;
            box-shadow: 0 4px 12px rgba(0,0,0,0.08);
            display: flex; justify-content: space-between; align-items: center;
            border-radius: 8px; margin-bottom: 20px;
            border: 1px solid rgba(0,0,0,0.05);
        }
        .stats span { margin-left: 10px; font-weight: 600; font-size: 14px; }
        .stats .done { color: #2ecc71; }
        .stats .todo { color: #e74c3c; }

        /* 网格布局 */
        .grid {
            display: grid;
            grid-template-columns: repeat(auto-fill, minmax(280px, 1fr));
            gap: 20px;
        }

        /* 卡片样式 */
        .card {
            background: white; border-radius: 12px; overflow: hidden;
            box-shadow: 0 2px 8px rgba(0,0,0,0.04);
            transition: transform 0.2s, box-shadow 0.2s;
            display: flex; flex-direction: column;
            border: 2px solid transparent;
            height: 100%;
        }
        .card:hover { transform: translateY(-4px); box-shadow: 0 12px 24px rgba(0,0,0,0.1); }
        .card.labeled { border-color: #2ecc71; background-color: #f9fffb; }

        /* 图片区域 */
        .img-wrapper {
            width: 100%; height: 220px; background: #eaeff2;
            display: flex; align-items: center; justify-content: center;
            border-bottom: 1px solid #f0f0f0;
            position: relative;
        }
        .img-wrapper img {
            max-width: 100%; max-height: 100%; object-fit: contain;
        }

        /* 内容区域 */
        .content { padding: 12px; display: flex; flex-direction: column; gap: 10px; flex-grow: 1; }
        
        /* 可复制元素的通用样式 */
        .copyable {
            cursor: pointer; position: relative;
            transition: background-color 0.2s;
            border-radius: 4px; padding: 4px;
        }
        .copyable:hover { background-color: #e3f2fd; color: #1976d2; }
        .copyable:active { background-color: #bbdefb; }
        /* 鼠标悬停提示 */
        .copyable::after {
            content: "点击复制"; position: absolute; right: 5px; top: 50%; transform: translateY(-50%);
            font-size: 10px; color: #1976d2; opacity: 0; pointer-events: none;
            background: rgba(255,255,255,0.9); padding: 2px 4px; border-radius: 3px;
        }
        .copyable:hover::after { opacity: 1; }

        .filename {
            font-size: 13px; font-weight: 600; color: #333;
            white-space: nowrap; overflow: hidden; text-overflow: ellipsis;
        }

        .desc-box {
            font-size: 11px; color: #555; background: #f8f9fa;
            border: 1px dashed #ced4da;
            height: 60px; overflow-y: auto; /* 内容过长则滚动 */
            line-height: 1.4;
        }

        /* 标签按钮组 */
        .btn-group { 
            display: grid; grid-template-columns: 1fr 1fr; gap: 8px; margin-top: auto; 
        }
        .btn {
            border: 1px solid #dfe1e5; background: white; padding: 8px 0;
            border-radius: 6px; cursor: pointer; font-size: 12px;
            transition: all 0.2s; text-align: center; color: #444;
        }
        .btn:hover { background: #f1f3f4; border-color: #dadce0; }

        /* 选中状态 */
        .btn.active { color: white; border-color: transparent; font-weight: 600; box-shadow: 0 2px 5px rgba(0,0,0,0.15); }
        
        /* 类别颜色映射 */
        .btn[data-label="cld_bottle_clip"].active { background-color: #3498db; }
        .btn[data-label="cld_bottle_pillow"].active { background-color: #9b59b6; }
        .btn[data-label="cld_bottle_straw"].active { background-color: #e67e22; }
        .btn[data-label="white"].active { background-color: #95a5a6; }
        .btn[data-label="uncertain"].active { background-color: #e74c3c; }

        /* 复制成功提示 Toast */
        #toast {
            visibility: hidden; min-width: 200px; background-color: #333; color: #fff;
            text-align: center; border-radius: 50px; padding: 12px; position: fixed;
            z-index: 2000; left: 50%; bottom: 30px; transform: translateX(-50%);
            font-size: 14px; box-shadow: 0 4px 12px rgba(0,0,0,0.2);
        }
        #toast.show { visibility: visible; animation: fadein 0.3s, fadeout 0.5s 2.0s; }
        @keyframes fadein { from {bottom: 0; opacity: 0;} to {bottom: 30px; opacity: 1;} }
        @keyframes fadeout { from {bottom: 30px; opacity: 1;} to {bottom: 0; opacity: 0;} }

    </style>
</head>
<body>

<div class="header">
    <h3 style="margin:0; color:#2c3e50;">X Label Tool v3.1</h3>
    <div class="stats">
        总数: <span id="total">0</span> |
        已标: <span id="labeled" class="done">0</span> |
        剩余: <span id="remain" class="todo">0</span>
    </div>
</div>

<div class="grid" id="gallery"></div>

<!-- 提示框 -->
<div id="toast">复制成功</div>

<script>
    const LABELS = {{ labels | tojson }};
    let allData = [];

    // 1. 获取数据
    fetch('/api/data')
        .then(res => res.json())
        .then(data => {
            allData = data;
            render();
            updateStats();
        });

    // 2. 渲染界面
    function render() {
        const container = document.getElementById('gallery');
        let html = '';

        allData.forEach((item, index) => {
            const url = item.img_orgn_url || "";
            const filename = url.split('/').pop() || "unknown.jpg";
            const desc = item.goods_desc || "暂无描述";
            
            // 状态判断
            const currentLabel = item.human_label;
            const cardClass = currentLabel ? 'card labeled' : 'card';

            // 生成标签按钮
            let btnsHtml = '';
            LABELS.forEach(lbl => {
                const activeClass = (currentLabel === lbl) ? 'active' : '';
                btnsHtml += `<div class="btn ${activeClass}" 
                                  data-label="${lbl}" 
                                  onclick="setLabel(${index}, '${lbl}')">
                                ${lbl}
                             </div>`;
            });

            html += `
                <div class="${cardClass}" id="card-${index}">
                    <div class="img-wrapper">
                        <img src="/api/image/${index}" loading="lazy" alt="加载失败">
                    </div>
                    <div class="content">
                        <!-- 点击复制文件名 -->
                        <div class="filename copyable" 
                             onclick="copyText('${filename}')" 
                             title="${filename}">
                             ${filename}
                        </div>
                        
                        <!-- 点击复制商品描述 -->
                        <div class="desc-box copyable" 
                             onclick="copyText(this.innerText)" 
                             title="点击复制描述">
                             ${desc}
                        </div>

                        <div class="btn-group">${btnsHtml}</div>
                    </div>
                </div>
            `;
        });
        container.innerHTML = html;
    }

    // 3. 标注功能
    window.setLabel = function(index, label) {
        allData[index].human_label = label;
        
        // 局部更新 DOM 样式
        const card = document.getElementById(`card-${index}`);
        card.className = 'card labeled'; // 设为已标注样式
        
        const btns = card.querySelectorAll('.btn');
        btns.forEach(b => {
            if (b.dataset.label === label) b.classList.add('active');
            else b.classList.remove('active');
        });

        updateStats();

        // 异步保存
        fetch('/api/update', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ index: index, label: label })
        }).catch(err => alert('保存失败，请检查后端服务'));
    };

    // 4. 复制功能 (兼容写法)
    window.copyText = function(text) {
        if (!text) return;
        
        // 尝试使用现代 API
        if (navigator.clipboard && navigator.clipboard.writeText) {
            navigator.clipboard.writeText(text).then(() => {
                showToast(`复制成功: ${text.substring(0, 15)}...`);
            });
        } else {
            // 回退方案
            const textArea = document.createElement("textarea");
            textArea.value = text;
            document.body.appendChild(textArea);
            textArea.select();
            document.execCommand("copy");
            document.body.removeChild(textArea);
            showToast("复制成功");
        }
    };

    function showToast(msg) {
        const x = document.getElementById("toast");
        x.innerText = msg;
        x.className = "show";
        setTimeout(() => { x.className = x.className.replace("show", ""); }, 2000);
    }

    function updateStats() {
        const total = allData.length;
        const done = allData.filter(d => d.human_label).length;
        document.getElementById('total').innerText = total;
        document.getElementById('labeled').innerText = done;
        document.getElementById('remain').innerText = total - done;
    }
</script>

</body>
</html>
"""

# 全局数据
json_data = []

def load_json():
    global json_data
    if os.path.exists(JSON_PATH):
        try:
            with open(JSON_PATH, 'r', encoding='utf-8') as f:
                json_data = json.load(f)
                print(f"[INFO] 成功加载 {len(json_data)} 条数据")
        except Exception as e:
            print(f"[ERROR] JSON 读取失败: {e}")
    else:
        print(f"[ERROR] 文件未找到: {JSON_PATH}")

def save_json():
    try:
        with open(JSON_PATH, 'w', encoding='utf-8') as f:
            json.dump(json_data, f, ensure_ascii=False, indent=4)
    except Exception as e:
        print(f"[ERROR] 保存失败: {e}")

@app.route('/')
def index():
    return render_template_string(HTML_TEMPLATE, labels=LABELS)

@app.route('/api/data')
def get_data():
    return jsonify(json_data)

@app.route('/api/image/<int:index>')
def get_image(index):
    if 0 <= index < len(json_data):
        item = json_data[index]
        url = item.get('img_orgn_url', '')
        if url:
            filename = os.path.basename(url)
            local_path = os.path.join(IMG_DIR, filename)
            if os.path.exists(local_path):
                return send_file(local_path)
            else:
                print(f"[WARN] 图片缺失: {local_path}")
    return "Image not found", 404

@app.route('/api/update', methods=['POST'])
def update_label():
    req = request.json
    idx = req.get('index')
    label = req.get('label')
    if idx is not None and 0 <= idx < len(json_data):
        json_data[idx]['human_label'] = label
        save_json()
        return jsonify({"success": True})
    return jsonify({"success": False}), 400

def check_port(port):
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        return s.connect_ex(('localhost', port)) != 0

if __name__ == '__main__':
    load_json()
    
    # 自动选择端口，优先5000，如果占用则尝试5001
    PORT = 5000
    if not check_port(PORT):
        print(f"[WARN] 端口 {PORT} 被占用，尝试使用 5001...")
        PORT = 5001
    
    print("--------------------------------------------------")
    print(f"服务启动中...")
    print(f"数据路径: {JSON_PATH}")
    print(f"请访问: http://127.0.0.1:{PORT}")
    print("--------------------------------------------------")
    app.run(host='0.0.0.0', port=PORT, debug=True)
