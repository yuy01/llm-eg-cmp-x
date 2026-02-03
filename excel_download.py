import os
import pandas as pd
import requests
from urllib.parse import urlparse
from PIL import Image
from io import BytesIO
import concurrent.futures # [新增] 用于多线程
import threading # [新增] 用于线程锁
from tqdm import tqdm # [新增] 用于进度条显示
import time # [新增] 用于计时

# [新增] 全局变量和锁，用于多线程间的计数和同步
success_count = 0
fail_count = 0
lock = threading.Lock()

# [新增辅助函数] 提取生成文件路径的逻辑，避免代码重复
def get_file_info(row, save_dir):
    """
    解析行数据，返回 (完整保存路径, 是否有效URL, 原始文件名)
    """
    goods_sn = str(row.get('goods_sn', '')).strip()
    img_url = row.get('img_orgn_url')

    if pd.isna(img_url) or str(img_url).strip() == '':
        return None, False, None
    
    img_url = str(img_url).strip()
    
    try:
        parsed_path = urlparse(img_url).path
        original_basename = os.path.basename(parsed_path)
        name_part, ext = os.path.splitext(original_basename)
        
        # 规则: sn_basename.jpg
        new_filename = f"{goods_sn}_{name_part}.jpg"
        save_path = os.path.join(save_dir, new_filename)
        return save_path, True, new_filename
    except:
        return None, False, None

def download_images_from_excel():
    global success_count, fail_count

    # 1. 配置路径
    # 输入的Excel文件路径
    # excel_path = '/home/hadoop/data/cldfeed/data_clean/data_cldfeed/cld_bottle_pillow_0129.xlsx'
    excel_path = '/home/hadoop/data/cldfeed/data_clean/data_cldfeed/json_download/存量表数据.xlsx'
    # excel_path = '/home/hadoop/data/cldfeed/data_clean/data_cldfeed/json_download/公文数据.xlsx'
    
    # 输出的文件夹路径
    # save_dir = '/home/hadoop/data/cldfeed/data_clean/data_cldfeed/json_download/cld_bottle_pillow_0129'
    save_dir = '/home/hadoop/data/cldfeed/data_clean/data_cldfeed/json_download/white_20w_excel_0203'
    save_dir = '/home/hadoop/data/share/Tongying_dataset/Raw_high_exposure'
        
    # [新增配置] 设置随机下载的数量
    # 注意：这个数量是“尝试处理的数据量”，如果开启了预检查，这里面可能包含已经下载过的。
    download_limit = 200000

    # [新增配置] 是否开启预检查
    # True: 先检查文件是否存在，只下载不存在的。
    # False: 直接全部丢进线程池，边下载边判断（会导致进度条不准，且包含已存在的计数）。
    check_exist = True

    # [新增配置] 线程并发数量
    max_workers = 30

    # 2. 创建保存目录
    if not os.path.exists(save_dir):
        try:
            os.makedirs(save_dir)
            print(f"已创建目录: {save_dir}")
        except Exception as e:
            print(f"创建目录失败: {e}")
            return

    # 3. 读取 Excel
    print(f"正在读取文件: {excel_path} ...")
    try:
        df = pd.read_excel(excel_path)
    except FileNotFoundError:
        print("错误: 找不到 Excel 文件，请检查路径。")
        return

    # [新增] 获取并打印 Excel 中的总数据量
    total_rows = len(df)
    print(f"Excel 文件中共有数据: {total_rows} 条")

    # [逻辑] 随机打乱数据顺序
    print("正在随机打乱数据顺序...")
    df = df.sample(frac=1).reset_index(drop=True)
    
    # [逻辑] 截取 download_limit 数量的数据
    # 如果 Excel 总数小于设定的限制，则取全部
    if total_rows > download_limit:
        df = df.head(download_limit)
        print(f"已根据限制截取前 {download_limit} 条随机数据进行处理。")
    else:
        print(f"数据量小于限制，将处理全部 {total_rows} 条数据。")

    # 将 DataFrame 转为字典列表
    data_list = df.to_dict('records')
    
    # ================= 预检查阶段 (Matching Phase) =================
    tasks_to_run = [] # 真正需要运行下载的任务
    
    if check_exist:
        print("-" * 30)
        print("开始预检查文件是否存在 (Matching Phase)...")
        match_start_time = time.time()
        
        exist_count = 0
        missing_count = 0
        
        # 遍历检查
        for row in data_list:
            save_path, valid, _ = get_file_info(row, save_dir)
            
            if not valid:
                continue # URL无效跳过
            
            if os.path.exists(save_path):
                exist_count += 1
            else:
                missing_count += 1
                tasks_to_run.append(row)
        
        match_end_time = time.time()
        print(f"预检查完成。")
        print(f"已存在文件: {exist_count} (跳过)")
        print(f"待下载文件: {missing_count} (加入队列)")
        print(f"匹配用时: {match_end_time - match_start_time:.2f} 秒")
        print("-" * 30)
    else:
        # 如果不配置检查，则全部任务都加入
        print("未开启预检查，所有任务直接进入下载队列...")
        tasks_to_run = data_list

    # ================= 下载阶段 (Download Phase) =================
    
    final_target = len(tasks_to_run)
    
    if final_target == 0:
        print("没有需要下载的文件，程序结束。")
        return

    print(f"开始多线程下载，线程数: {max_workers}，需下载数量: {final_target}...")
    
    # 伪装 User-Agent
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
    }

    # 初始化进度条
    # total=final_target 现在的 total 是真正需要下载的数量，进度条会很准
    pbar = tqdm(total=final_target, desc="下载进度", unit="img")
    
    download_start_time = time.time()

    # --- 定义单张图片处理函数 (供线程调用) ---
    def process_one_image(row):
        global success_count, fail_count
        
        # 获取路径 (虽然预检查做过一次，但为了线程内逻辑完整再次调用，开销很小)
        # 注意：如果是 check_exist=False 模式，这里需要再次判断 exists
        save_path, valid, _ = get_file_info(row, save_dir)
        
        if not valid:
            return

        img_url = str(row.get('img_orgn_url', '')).strip()
        goods_sn = str(row.get('goods_sn', '')).strip()

        # 双重保险：如果不开启预检查，或者多线程间有极小概率重复，这里再挡一下
        if os.path.exists(save_path):
             # 如果之前没检查，这里遇到了已存在的，我们手动更新一下进度条并返回
             # 这样能保证 check_exist=False 时进度条也能走完
             if not check_exist:
                 with lock:
                     pbar.update(1)
             return

        try:
            # 下载请求
            response = requests.get(img_url, headers=headers, timeout=15)
            response.raise_for_status()

            # 8. 图片处理与保存 (使用 PIL)
            # 使用 BytesIO 将二进制流转换为图片对象
            image = Image.open(BytesIO(response.content))
            if image.mode in ("RGBA", "P"):
                image = image.convert("RGB")

            image.save(save_path, 'JPEG', quality=90)
            
            # [线程安全区域] 修改计数器并更新进度条
            with lock:
                success_count += 1
                pbar.update(1)
        
        except Exception as e:
            with lock:
                # 失败也算处理了一个任务，更新进度条，以免最后进度条不满
                pbar.update(1) 
                # 打印失败信息（可选 pbar.write 防止进度条错位）
                # pbar.write(f"[失败] sn: {goods_sn} | {e}")
                fail_count += 1

    # 4. 执行多线程下载
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        # 提交所有任务
        futures = [executor.submit(process_one_image, row) for row in tasks_to_run]
        # 等待所有任务完成
        concurrent.futures.wait(futures)

    # 关闭进度条
    pbar.close()
    
    download_end_time = time.time()

    print("-" * 30)
    print(f"下载处理完成。")
    print(f"成功下载: {success_count}")
    print(f"下载失败: {fail_count}")
    print(f"下载用时: {download_end_time - download_start_time:.2f} 秒")
    
    if check_exist:
        total_time = (download_end_time - download_start_time) + (match_end_time - match_start_time)
        print(f"总流程用时: {total_time:.2f} 秒")

if __name__ == "__main__":
    download_images_from_excel()
