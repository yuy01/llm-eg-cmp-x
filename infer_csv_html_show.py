import os
import json
import pandas as pd
import requests
from pathlib import Path

# ================= 配置区域 =================

# CSV文件路径
# CSV_PATH = '/home/hadoop/data/cldfeed/dinov3/inference_results/infer_01282028/10_model_20260128_203926_predictions.csv'
# CSV_PATH = '/home/hadoop/data/cldfeed/dinov3/inference_results/infer_01291503/10_model_20260129_151241_predictions.csv'
# CSV_PATH = '/home/hadoop/data/cldfeed/dinov3/inference_results/infer_01291801/10_model_20260129_184802_predictions.csv'
CSV_PATH = '/home/hadoop/data/cldfeed/dinov3/inference_results/infer_02031525/12_model_20260203_152529_predictions.csv'

# JSON文件路径 
JSON_PATH = '/home/hadoop/data/cldfeed/data_clean/data_cldfeed/20260122.json' 
JSON_PATH = '/home/hadoop/data/share/Tongying_dataset/Raw_high_exposure/raw_whole_high_exposure_datas.json' 

# 图片下载存放路径
IMG_SAVE_DIR = '/home/hadoop/data/cldfeed/data_clean/data_cldfeed/json_download/visualize_pic'
IMG_SAVE_DIR = '/home/hadoop/data/cldfeed/IC_datas/IC_2class_round2_pillow/Ratio_30_1/train/cld_bottle_pillow'

# 筛选的预测类别 (可以是多个)
TARGET_CLASSES = ['cld_bottle_pillow','white']

# 匹配模式选择
# 可选模式: 
# 1. 'basename'    -> 仅使用文件名 (不含后缀)
# 2. 'sn'          -> 使用 goods_sn
# 3. 'sn_basename' -> 使用 goods_sn + "_" + basename
MATCH_MODE = 'basename' 
MATCH_MODE = 'sn_basename' 

# ===========================================

def get_basename_no_ext(filename_or_url):
    """提取不带后缀的文件名"""
    filename = os.path.basename(filename_or_url)
    return os.path.splitext(filename)[0]

def get_match_key(item, mode):
    """根据模式从JSON项中提取匹配键"""
    url_basename = get_basename_no_ext(item.get('img_orgn_url', ''))
    sn = item.get('goods_sn', '')
    
    if mode == 'basename':
        return url_basename
    elif mode == 'sn':
        return sn
    elif mode == 'sn_basename':
        return f"{sn}_{url_basename}"
    return url_basename

def download_image(url, save_path):
    """下载图片"""
    if os.path.exists(save_path):
        return True # 已存在，跳过下载
    try:
        response = requests.get(url, timeout=10)
        if response.status_code == 200:
            with open(save_path, 'wb') as f:
                f.write(response.content)
            return True
    except Exception as e:
        print(f"下载失败: {url}, 错误: {e}")
    return False

def generate_html(data_list, output_path, total_csv, matched_count):
    """生成HTML文件"""
    html_content = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <meta charset="utf-8">
        <title>Inference Visualization</title>
        <style>
            body {{ font-family: Arial, sans-serif; margin: 20px; }}
            .header {{ margin-bottom: 20px; }}
            .container {{ display: flex; flex-wrap: wrap; gap: 15px; }}
            .card {{ 
                border: 1px solid #ddd; 
                border-radius: 8px; 
                padding: 10px; 
                width: 220px; 
                box-shadow: 0 2px 4px rgba(0,0,0,0.1);
                background-color: #fff;
            }}
            .card img {{ width: 100%; height: auto; border-radius: 4px; display: block; }}
            .info {{ margin-top: 10px; font-size: 12px; word-wrap: break-word; }}
            .label {{ font-weight: bold; color: #333; }}
            .score {{ color: #d9534f; font-weight: bold; }}
        </style>
    </head>
    <body>
        <div class="header">
            <h2>可视化结果报告</h2>
            <p><strong>CSV路径:</strong> {CSV_PATH}</p>
            <p><strong>匹配模式:</strong> {MATCH_MODE}</p>
            <p><strong>筛选类别:</strong> {', '.join(TARGET_CLASSES)}</p>
            <p><strong>统计:</strong> CSV中目标总数: {total_csv} | 成功匹配并展示: {matched_count}</p>
        </div>
        <div class="container">
    """

    for item in data_list:
        # 计算图片的相对路径，以便HTML在不同位置打开时也能找到图片（只要相对位置不变）
        # 这里为了兼容性，使用基于生成的HTML文件的相对路径
        try:
            rel_img_path = os.path.relpath(item['local_path'], os.path.dirname(output_path))
        except ValueError:
            # 如果不在同一个磁盘分区，使用绝对路径
            rel_img_path = item['local_path']

        html_content += f"""
            <div class="card">
                <img src="{rel_img_path}" alt="img" loading="lazy" onclick="window.open(this.src)">
                <div class="info">
                    <div class="label">SN: {item['sn']}</div>
                    <div>Pred: {item['pred_class']}</div>
                    <div>Score: <span class="score">{item['score']}</span></div>
                    <div style="color:#888; font-size:10px;">{item['img_name']}</div>
                </div>
            </div>
        """

    html_content += """
        </div>
    </body>
    </html>
    """
    
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(html_content)
    print(f"HTML生成完毕: {output_path}")

def main():
    # 1. 准备目录
    if not os.path.exists(IMG_SAVE_DIR):
        os.makedirs(IMG_SAVE_DIR)
        print(f"创建图片下载目录: {IMG_SAVE_DIR}")

    # 2. 加载 JSON 数据并建立索引
    print("正在加载 JSON 数据...")
    json_lookup = {}
    try:
        with open(JSON_PATH, 'r', encoding='utf-8') as f:
            json_data = json.load(f)
            for item in json_data:
                # 根据配置的模式生成 key
                key = get_match_key(item, MATCH_MODE)
                json_lookup[key] = item
    except FileNotFoundError:
        print(f"错误: 找不到JSON文件 {JSON_PATH}")
        return

    # 3. 加载 CSV 数据
    print("正在加载 CSV 数据...")
    df = pd.read_csv(CSV_PATH)
    
    # 4. 筛选数据
    df_filtered = df[df['predicted_class'].isin(TARGET_CLASSES)]
    total_target_count = len(df_filtered)
    print(f"筛选类别 {TARGET_CLASSES} 后，共有 {total_target_count} 条数据。")

    matched_list = []
    missing_list = []

    print("开始匹配和下载图片...")
    for index, row in df_filtered.iterrows():
        img_name_csv = row['image_name']
        score = row['score']
        pred_class = row['predicted_class']
        
        # CSV中的匹配键处理 (同样依据 basename 模式)
        # 假设 csv 中的 image_name 也是 basename.ext
        csv_key = get_basename_no_ext(img_name_csv)
        
        if csv_key in json_lookup:
            json_info = json_lookup[csv_key]
            url = json_info.get('img_orgn_url')
            sn = json_info.get('goods_sn')
            
            if url:
                # 定义本地保存的文件名，保持原始后缀
                ext = os.path.splitext(url)[1]
                if not ext: ext = '.jpg'
                local_filename = f"{csv_key}{ext}"
                local_file_path = os.path.join(IMG_SAVE_DIR, local_filename)
                
                # 下载图片
                download_image(url, local_file_path)
                
                matched_list.append({
                    'sn': sn,
                    'score': score,
                    'pred_class': pred_class,
                    'local_path': local_file_path,
                    'img_name': img_name_csv
                })
            else:
                missing_list.append(f"{img_name_csv} (JSON中无URL)")
        else:
            # === 修改处：如果在json文件中没有找到，就用原来的文件名 ===
            
            # 使用 CSV 中的原始文件名作为本地文件名
            # 确保使用 basename 避免路径拼接问题
            local_filename = os.path.basename(img_name_csv)
            local_file_path = os.path.join(IMG_SAVE_DIR, local_filename)
            
            # 既然 JSON 中没有，就没有 URL，无法下载，假设图片已存在本地目录
            # 或者用户希望在 HTML 看到这个条目（即使图片裂开）
            matched_list.append({
                'sn': 'No JSON Info', # 缺失信息
                'score': score,
                'pred_class': pred_class,
                'local_path': local_file_path,
                'img_name': img_name_csv
            })
            
            # 注释掉原来的逻辑，不再认为是丢失的数据
            # missing_list.append(img_name_csv)

    # 5. 生成 HTML
    html_filename = Path(CSV_PATH).stem + '_visualization.html'
    html_output_path = os.path.join(os.path.dirname(CSV_PATH), html_filename)
    
    # === 新增代码：按分数降序排序 ===
    matched_list.sort(key=lambda x: x['score'], reverse=True)
    # ==============================
    
    generate_html(matched_list, html_output_path, total_target_count, len(matched_list))

    # 6. 打印统计信息
    print("=" * 30)
    print("统计结果:")
    print(f"总筛选数量: {total_target_count}")
    print(f"匹配并展示数量: {len(matched_list)}")
    print(f"未匹配/无URL数量: {len(missing_list)}")
    
    if missing_list:
        print("\n未找到信息的图片 (前10个):")
        for m in missing_list[:10]:
            print(m)
        if len(missing_list) > 10:
            print(f"... 还有 {len(missing_list) - 10} 个")

if __name__ == "__main__":
    main()
