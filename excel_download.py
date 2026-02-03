import os
import pandas as pd
import requests
from urllib.parse import urlparse
from PIL import Image
from io import BytesIO
import concurrent.futures # [新增] 用于多线程
import threading # [新增] 用于线程锁
from tqdm import tqdm # [新增] 用于进度条显示

# [新增] 全局变量和锁，用于多线程间的计数和同步
success_count = 0
fail_count = 0
lock = threading.Lock()

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

    # [新增配置] 设置随机下载的数量
    download_limit = 200000

    # [新增配置] 日志打印频率 (由于改为进度条，此变量暂时只作为保留配置，不再用于print)
    log_interval = 10000

    # [新增配置] 线程并发数量
    # 建议设置为 20-50，根据网络带宽和机器性能调整
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
    # frac=1 表示抽取 100% 的数据（即全部），从而达到打乱顺序的目的
    print("正在随机打乱数据顺序...")
    df = df.sample(frac=1).reset_index(drop=True)

    # 伪装 User-Agent 防止被简单的反爬拦截
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
    }

    # [修改] 计算实际的下载目标数量（用于打印和逻辑判断）
    # 如果 Excel 总数小于设定的限制，则目标就是 Excel 总数，否则是设定限制
    if total_rows < download_limit:
        final_target = total_rows
    else:
        final_target = download_limit

    print(f"开始多线程下载，线程数: {max_workers}，目标数量: {final_target}...")

    # [新增] 初始化进度条
    # total=final_target 表示进度条的总长度是我们要下载的目标数量
    pbar = tqdm(total=final_target, desc="下载进度", unit="img")

    # --- 定义单张图片处理函数 (供线程调用) ---
    def process_one_image(row):
        global success_count, fail_count
        
        # [优化] 如果已经达到目标数量，直接退出，减少不必要的请求
        # 这里为了效率先不加锁读取，下面写入时再加锁
        if success_count >= final_target:
            return

        goods_sn = str(row.get('goods_sn', '')).strip()
        img_url = row.get('img_orgn_url')

        # 检查 URL 是否有效
        if pd.isna(img_url) or str(img_url).strip() == '':
            # URL为空直接跳过
            return
        
        img_url = str(img_url).strip()

        try:
            # 5. 解析文件名
            # 获取 URL 的 basename (例如: .../abc.png -> abc.png)
            parsed_path = urlparse(img_url).path
            original_basename = os.path.basename(parsed_path)
            
            # 分离文件名和后缀 (abc.png -> abc, .png)
            name_part, ext = os.path.splitext(original_basename)
            
            # 6. 构建新文件名
            # 规则: sn_basename.jpg
            # 无论原后缀是什么，最终都保存为 .jpg
            new_filename = f"{goods_sn}_{name_part}.jpg"
            save_path = os.path.join(save_dir, new_filename)

            # 如果文件已存在，可以选择跳过，这里默认覆盖或跳过请根据需求调整
            if os.path.exists(save_path):
                # [修改] 为了减少日志，文件已存在时不打印
                return

            # 7. 下载请求
            response = requests.get(img_url, headers=headers, timeout=15)
            response.raise_for_status()  # 检查是否请求成功 (200 OK)

            # 8. 图片处理与保存 (使用 PIL)
            # 使用 BytesIO 将二进制流转换为图片对象
            image = Image.open(BytesIO(response.content))

            # 如果是 PNG 等包含透明通道(RGBA)或调色板模式(P)的图片，转换为 RGB 模式
            if image.mode in ("RGBA", "P"):
                image = image.convert("RGB")

            # 保存为 JPG
            image.save(save_path, 'JPEG', quality=90)
            
            # [线程安全区域] 修改计数器并更新进度条
            with lock:
                # 再次检查是否超过目标（防止多个线程同时冲过终点）
                if success_count < final_target:
                    success_count += 1
                    # [新增] 更新进度条，步长为1
                    pbar.update(1)
        
        except Exception as e:
            with lock:
                # [可选] 失败时如果不希望打断进度条显示，可以使用 pbar.write 代替 print
                # pbar.write(f"[失败] sn: {goods_sn}, url: {img_url}, Err: {e}")
                
                # 错误日志建议保留，以便排查问题 (这里保留 print，tqdm会自动处理换行)
                # print(f"[失败] sn: {goods_sn}, url: {img_url}")
                # print(f"       错误信息: {e}")
                fail_count += 1

    # 4. 遍历每一行进行下载 (改为多线程执行)
    # 将 DataFrame 转为字典列表，方便遍历提交
    data_list = df.to_dict('records')

    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        # 提交所有任务
        futures = [executor.submit(process_one_image, row) for row in data_list]
        
        # 等待所有任务完成
        concurrent.futures.wait(futures)

    # 关闭进度条
    pbar.close()

    print("-" * 30)
    print(f"处理完成。成功: {success_count}, 失败: {fail_count}")

if __name__ == "__main__":
    download_images_from_excel()
