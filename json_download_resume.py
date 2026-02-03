import os
import json
import csv
import requests
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed

# ================= 配置区域 =================
# 原有的文件夹配置
JSON_DIR = "/home/hadoop/data/cldfeed/data_clean/data_cldfeed/cld_bottle_black_audit_datas"

# SAVE_DIR = "/home/hadoop/data/cldfeed/data_clean/data_cldfeed/cld_bottle_black_audit_datas/cld_bottle_black_audit_datas_download"
# SAVE_DIR = "/home/hadoop/data/cldfeed/data_clean/data_cldfeed/json_download_20260122"
# SAVE_DIR = "/home/hadoop/data/cldfeed/data_clean/data_cldfeed/json_download/other_data"
SAVE_DIR = "/home/hadoop/data/cldfeed/dinov3/inference_results/infer_01281338/black_pictures"
SAVE_DIR = "/home/hadoop/data/cldfeed/data_clean/data_cldfeed/json_download/sample_datas_1_0128_infer"
SAVE_DIR = "/Users/10294814/task/cldfeed/abner/data_clean/data_cldfeed/audit_result_0202"

# 【新增功能】指定单个JSON文件路径
# 如果这个变量不为 None 且路径存在，脚本将忽略 JSON_DIR，只处理这个文件
# 示例：SINGLE_JSON_PATH = "/home/hadoop/data/test_data.json"
SINGLE_JSON_PATH = None 
# SINGLE_JSON_PATH = "/home/hadoop/data/cldfeed/data_clean/data_cldfeed/20260122.json" # 解除注释以启用单文件模式
# SINGLE_JSON_PATH = "/home/hadoop/data/cldfeed/data_clean/data_cldfeed/other_data.json" # 解除注释以启用单文件模式
SINGLE_JSON_PATH = "/home/hadoop/data/cldfeed/data_clean/data_cldfeed/20260126.json" # 解除注释以启用单文件模式
SINGLE_JSON_PATH = "/home/hadoop/data/cldfeed/data_clean/data_cldfeed/20260122.json" # 解除注释以启用单文件模式
SINGLE_JSON_PATH = "/home/hadoop/data/cldfeed/data_clean/data_cldfeed/sample_datas_1.json" # 解除注释以启用单文件模式
SINGLE_JSON_PATH = "/Users/10294814/task/cldfeed/abner/data_clean/data_cldfeed/audit_result_0202.json" # 解除注释以启用单文件模式

# 【新增功能】指定CSV文件路径（基于推理结果下载）
# 如果这个变量不为 None 且路径存在，脚本将：
# 1. 读取CSV中包含图片文件名的列 (格式如 sn_xxxx.jpg 或 sn.jpg)
# 2. 提取sn号
# 3. 去JSON中查找对应的url并下载
# 注意：开启此模式时，会忽略下方的 WHITE_TOP_SCORE 筛选逻辑，只下载CSV中存在的sn
CSV_PATH = None
# CSV_PATH = "/home/hadoop/data/cldfeed/dinov3/inference_results/infer_01281338/8_model_20260128_142628_predictions.csv" # 解除注释以启用CSV筛选模式

# 【新增配置】指定需要下载的预测类别
# 在这里定义需要筛选的 predicted_class，可以有多个
# 如果留空 []，在CSV模式下表示不筛选类别，全部匹配
TARGET_PREDICTED_CLASSES = [] 

# TARGET_PREDICTED_CLASSES = ["cld_bottle_pillow"] 
# 以后如果需要增加其他类，可以这样写： TARGET_PREDICTED_CLASSES = ["cld_bottle_pillow", "other_class"]


MAX_WORKERS = 20

# white标签的top_score筛选区间
WHITE_TOP_SCORE_MIN = 0.0  # 最小阈值
WHITE_TOP_SCORE_MAX = 1.0  # 最大阈值

# 【新增配置】
# 1. 是否跳过主程序的本地文件预检查
# True: 不在主线程遍历检查文件是否存在（启动极快），直接全部进入下载队列，由下载线程决定是否下载。
# False: 先遍历检查本地文件（启动较慢），只将不存在的文件放入下载队列。
SKIP_PRE_CHECK = True  

# 2. 是否强制覆盖已存在的文件
# True: 即使文件存在也重新下载（常用于修复损坏文件或更新图片）。
# False: 如果文件存在则跳过（节省带宽和时间）。
FORCE_OVERWRITE = False

# 【新增配置】指定JSON中图片链接的字段名
# 你的新数据是 "img_url"，旧数据可能是 "img_orgn_url"
# 请在此处定义，后续代码将使用这个变量
IMG_URL_KEY = "img_url" 
# IMG_URL_KEY = "img_orgn_url" 

# 【新增配置】 是否下载所有数据（忽略infer_label和分数筛选）
# True: 不管标签是white还是其他，也不管分数多少，JSON里有什么就下载什么
# False: 按照 infer_label 和 WHITE_TOP_SCORE 逻辑筛选
DOWNLOAD_ALL_DATA = True 

# 【新增配置】 IDX 范围筛选 (idx通常为整数)
# 如果不想按idx筛选，请将以下两个变量设置为 None
# 示例：只下载 idx 为 1 到 50 的数据 -> IDX_MIN = 1, IDX_MAX = 50
IDX_MIN = 1       # 起始 idx (包含)
IDX_MAX = 10      # 结束 idx (包含)
# 若要关闭 idx 筛选，请解开下面两行注释：
# IDX_MIN = None
# IDX_MAX = None

# 【新增配置】 是否在文件名前添加 idx (例如: 1_sn123_img.jpg)
# True: 添加 idx_ 前缀 (前提是数据中有idx字段)
# False: 不添加
ADD_IDX_TO_FILENAME = True
ADD_IDX_TO_FILENAME = False
# ===========================================

os.makedirs(SAVE_DIR, exist_ok=True)

# def get_save_path(item, save_dir):
#     """
#     根据item信息生成本地保存的绝对路径。
#     """
#     img_url = item.get(IMG_URL_KEY)
#     if not img_url:
#         return None
    
#     filename = os.path.basename(img_url)
#     if not filename:
#         filename = f"{item.get('goods_sn', 'unknown')}.jpg"
    
#     return os.path.join(save_dir, filename)

def get_save_path(item, save_dir):
    """
    根据item信息生成本地保存的绝对路径。
    【修改】强制使用 goods_sn 命名图片
    【修改】支持在文件名前添加 idx
    【修改】支持根据 label 建立子文件夹
    """
    # 【修改】使用配置的 IMG_URL_KEY 获取链接
    img_url = item.get(IMG_URL_KEY)
    goods_sn = item.get('goods_sn')

    # 如果没有图片链接或没有sn号，则无法正确命名，跳过
    if not img_url or not goods_sn:
        return None
    # 构造文件名
    
    # 1. 获取URL最后的文件名部分 (e.g., img_01.png 或 img_02.webp?v=1)
    url_base_name = os.path.basename(img_url)
    
    # 2. 去除可能存在的URL参数（即 ? 之后的内容）
    url_base_name_clean = url_base_name.split('?')[0]
    
    # 3. 使用 os.path.splitext 分离文件名和原有后缀 
    # (e.g. 'my_image_01.png' -> 'my_image_01')
    # 注意：这里的 name_without_ext 会完整保留原文件名中的下划线
    name_without_ext = os.path.splitext(url_base_name_clean)[0]
    
    # 4. 强制拼接 .jpg 后缀
    # 基础文件名
    filename = f"{goods_sn}_{name_without_ext}.jpg"
    
    # 【新增逻辑】如果开启了添加IDX且数据中有idx，则拼接到最前面
    if ADD_IDX_TO_FILENAME:
        idx = item.get('idx')
        if idx is not None:
            filename = f"{idx}_{filename}"
    
    # 【新增逻辑】处理 label 文件夹
    # 获取 label 字段
    label = item.get('label')
    if label:
        # 清洗 label 名称，防止包含非法字符（如 / 或 \），将其替换为下划线
        safe_label_name = str(label).replace('/', '_').replace('\\', '_').strip()
        # 将保存路径指向子文件夹
        save_dir = os.path.join(save_dir, safe_label_name)

    return os.path.join(save_dir, filename)

def get_sns_from_csv(csv_path, target_classes):
    """
    从CSV文件中读取数据，筛选指定类别的行，并提取SN号。
    使用 csv.DictReader 基于表头读取 (image_name, predicted_class, score)
    """
    target_sns = set()
    print(f"正在从CSV读取SN: {csv_path}")
    print(f"目标筛选类别: {target_classes}")

    try:
        with open(csv_path, 'r', encoding='utf-8', errors='ignore') as f:
            # 使用 DictReader 自动识别表头
            reader = csv.DictReader(f)
            
            for row in reader:
                # 1. 获取当前行的预测类别 (去除前后空格)
                p_class = row.get('predicted_class', '').strip()
                
                # 2. 判断是否在目标类别中
                # 【修改】如果 target_classes 不为空，才进行筛选；为空则默认全部包含
                if target_classes and p_class not in target_classes:
                    continue
                
                # 3. 获取文件名
                filename = row.get('image_name', '').strip()
                if not filename:
                    continue
                
                # 4. 提取SN逻辑
                # 支持 sn_xxxx.jpg 和 sn.jpg
                # 去除后缀 (pants181025908.jpg -> pants181025908)
                name_no_ext = os.path.splitext(filename)[0]
                
                # 提取SN：取第一个下划线前面的部分
                # 如果没有下划线，split返回原字符串，逻辑通用
                sn = name_no_ext.split('_')[0]
                
                if sn:
                    target_sns.add(sn)

    except Exception as e:
        print(f"读取CSV失败: {e}")
        print("请确保CSV文件包含表头: image_name, predicted_class")
    
    print(f"从CSV中提取到 {len(target_sns)} 个符合条件的唯一SN号")
    return target_sns

def download_one(item):
    """
    下载单张图片
    """
    # 【修改】使用配置的 IMG_URL_KEY 获取链接
    img_url = item.get(IMG_URL_KEY)
    if not img_url:
        return

    try:
        # 获取保存路径
        save_path = get_save_path(item, SAVE_DIR)
        if not save_path:
            return
        
        # 【新增逻辑】确保子文件夹存在（因为路径现在包含了label子目录）
        save_folder = os.path.dirname(save_path)
        if not os.path.exists(save_folder):
            os.makedirs(save_folder, exist_ok=True)
        
        # 【修改逻辑】如果不强制覆盖，且文件存在且大小大于0，则跳过
        if not FORCE_OVERWRITE:
            # 双重检查：防止并发写入
            if os.path.exists(save_path) and os.path.getsize(save_path) > 0:
                return

        headers = {'User-Agent': 'Mozilla/5.0'}
        response = requests.get(img_url, headers=headers, timeout=10)
        
        if response.status_code == 200:
            # 写入临时文件再重命名，防止中断导致破损文件
            temp_path = save_path + ".tmp"
            with open(temp_path, 'wb') as f:
                f.write(response.content)
            os.rename(temp_path, save_path)
        else:
            pass
    except Exception as e:
        pass

def collect_items(source_path, target_sns=None):
    """
    【修改】收集符合条件的条目
    source_path: 可以是文件夹路径，也可以是单个json文件路径
    target_sns: set集合，如果传入不为None，则只筛选goods_sn在集合中的item
    """
    items = []
    if not os.path.exists(source_path):
        print(f"错误: 路径不存在 -> {source_path}")
        return items

    # 【核心逻辑修改】判断传入的是文件还是目录，生成待处理的文件列表
    file_list = []
    if os.path.isfile(source_path):
        print(f"正在读取 单个JSON 文件: {source_path}")
        file_list.append(source_path)
    else:
        # 如果是目录，遍历目录下的json
        raw_files = [f for f in os.listdir(source_path) if f.lower().endswith(".json")]
        print(f"正在读取 目录下的 JSON 文件 ({len(raw_files)} 个)...")
        file_list = [os.path.join(source_path, f) for f in raw_files]

    # 统一处理文件列表
    for file_path in file_list:
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                
                # 确保data是列表
                if not isinstance(data, list):
                    continue

                filtered = []
                for item in  data:# 加上了 data 和冒号
                    # 【修改】这里使用 IMG_URL_KEY 来判断是否有图片链接
                    if not isinstance(item, dict) or not item.get(IMG_URL_KEY):
                        continue

                    # ==================== IDX 筛选逻辑 ====================
                    # 获取当前item的idx
                    item_idx = item.get('idx')
                    
                    # 仅当 用户设置了范围 且 当前数据存在idx 时，才进行判断
                    if IDX_MIN is not None and IDX_MAX is not None and item_idx is not None:
                        try:
                            val = int(item_idx)
                            # 如果不在范围内，则跳过
                            if not (IDX_MIN <= val <= IDX_MAX):
                                continue
                        except ValueError:
                            # 如果idx转不成数字，默认忽略该条件，继续下载（防止漏下）
                            pass
                    # ====================================================
                    
                    goods_sn = str(item.get('goods_sn', ''))

                    # logic branch 1: 如果有 target_sns (CSV模式)，则优先匹配 SN
                    if target_sns is not None:
                        # 检查当前 item 的 sn 是否在 CSV 提取的名单里
                        if goods_sn in target_sns:
                            filtered.append(item)
                        continue # CSV模式下，只要SN对上就加入，不看label，直接进入下一个循环

                    # logic branch 2: 原有逻辑 (White标签筛选)
                    # 【修改】如果开启了 DOWNLOAD_ALL_DATA，则直接加入下载队列，忽略标签判断
                    if DOWNLOAD_ALL_DATA:
                        filtered.append(item)
                        continue

                    infer_label = item.get('infer_label')
                    if infer_label != 'white':
                        # 非white标签的图片，直接下载
                        filtered.append(item)
                    else:
                        # white标签的图片，根据top_score区间筛选
                        top_score = item.get('top_score', 0)
                        if WHITE_TOP_SCORE_MIN <= top_score <= WHITE_TOP_SCORE_MAX:
                            filtered.append(item)
                
                items.extend(filtered)
        except Exception as e:
            print(f"读取{file_path}失败: {e}")
            
    return items

def main():
    print("="*50)
    
    # 【新增】逻辑判断：决定处理目标是 文件夹 还是 单个文件
    target_source = JSON_DIR
    mode_msg = "文件夹模式"
    
    if SINGLE_JSON_PATH and os.path.exists(SINGLE_JSON_PATH):
        target_source = SINGLE_JSON_PATH
        mode_msg = "单文件模式"
    elif SINGLE_JSON_PATH and not os.path.exists(SINGLE_JSON_PATH):
        print(f"警告: 指定的单文件路径不存在: {SINGLE_JSON_PATH}，将回退到文件夹模式。")

    # 【新增】CSV 模式判断
    target_sns_set = None
    if CSV_PATH and os.path.exists(CSV_PATH):
        print(f"CSV模式已开启，将根据CSV文件筛选SN: {CSV_PATH}")
        # 传入需要筛选的类别列表
        target_sns_set = get_sns_from_csv(CSV_PATH, TARGET_PREDICTED_CLASSES)
    elif CSV_PATH:
         print(f"警告: 指定的CSV路径不存在: {CSV_PATH}，将回退到普通筛选模式。")

    print(f"运行模式: {mode_msg}")
    print(f"处理路径: {target_source}")
    print(f"保存文件夹: {SAVE_DIR}")
    if target_sns_set is None:
        if DOWNLOAD_ALL_DATA:
            print("筛选模式: [全量下载] 忽略标签和分数，下载所有内容")
        else:
            print(f"筛选模式: White标签筛选区间: {WHITE_TOP_SCORE_MIN} - {WHITE_TOP_SCORE_MAX}")
    else:
        print(f"筛选模式: 仅下载 CSV 中类别为 {TARGET_PREDICTED_CLASSES if TARGET_PREDICTED_CLASSES else 'ALL'} 的 {len(target_sns_set)} 个 SN")
    
    # 打印 IDX 筛选状态
    if IDX_MIN is not None and IDX_MAX is not None:
        print(f"IDX 范围: {IDX_MIN} - {IDX_MAX} (包含)")
    else:
        print("IDX 范围: 不筛选 (全量或不存在 idx)")

    print(f"跳过本地预检: {SKIP_PRE_CHECK}")
    print(f"强制覆盖下载: {FORCE_OVERWRITE}")
    print(f"图片字段Key : {IMG_URL_KEY}")
    print(f"文件名加IDX : {ADD_IDX_TO_FILENAME}")
    print("="*50)

    # 1. 收集所有符合业务逻辑的条目 (传入 target_source 和 可能存在的 target_sns_set)
    target_items = collect_items(target_source, target_sns=target_sns_set)
    
    # 统计各标签的数量
    label_stats = {}
    for item in target_items:
        # 优先取 infer_label，取不到取 label，都取不到则 unknown
        label = item.get('infer_label') or item.get('label') or 'unknown'
        label_stats[label] = label_stats.get(label, 0) + 1
    
    print("\n[标签统计]:")
    for label, count in label_stats.items():
        print(f"  {label}: {count}")


    need_download_items = [] # 初始化为空列表
    existing_count = 0

    if SKIP_PRE_CHECK:
        print("\n提示: 已配置跳过本地文件对比，所有图片均加入下载队列...")
        need_download_items = target_items
        existing_count = 0 # 无法统计，置为0
    else:
        print("\n正在比对本地文件...")
        for item in tqdm(target_items, desc="比对本地文件"):
            # 如果配置了强制覆盖，则不需要检查本地是否存在，直接加入下载列表
            if FORCE_OVERWRITE:
                need_download_items.append(item)
                continue

            save_path = get_save_path(item, SAVE_DIR)
            if save_path and os.path.exists(save_path) and os.path.getsize(save_path) > 0:
                existing_count += 1
            else:
                need_download_items.append(item)

    # 3. 打印统计信息
    print("="*50)
    print(f"JSON统计符合条件总数: {len(target_items)}")
    if not SKIP_PRE_CHECK and not FORCE_OVERWRITE:
        print(f"文件夹已存在图片数量: {existing_count}")
    print(f"本次需下载图片数量  : {len(need_download_items)}")
    print("="*50)

    if not need_download_items:
        print("✅ 所有图片均已存在，无需下载。")
        return

    # 4. 执行下载
    print(f"\n🚀 启动 {MAX_WORKERS} 个线程开始下载...")
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = [executor.submit(download_one, item) for item in need_download_items]
        for _ in tqdm(as_completed(futures), total=len(need_download_items), desc="下载进度"):
            pass

    print(f"\n✅ 任务全部完成，图片保存在: {SAVE_DIR}")

if __name__ == '__main__':
    main()
