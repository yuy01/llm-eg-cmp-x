import os
import shutil
import torch
import faiss
import numpy as np
import networkx as nx
from PIL import Image
from tqdm import tqdm
from sentence_transformers import SentenceTransformer
from datetime import datetime
from torch.utils.data import Dataset, DataLoader
from pathlib import Path

# --- 1. 数据集定义 (用于并行加速) ---
class ImageDataset(Dataset):
    def __init__(self, image_paths):
        self.image_paths = image_paths

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        path = self.image_paths[idx]
        try:
            # convert('RGB') 极其重要！
            # 1. 解决 PNG 4通道(RGBA)导致模型报错的问题
            # 2. 解决灰度图维度不匹配问题
            return Image.open(path).convert('RGB')
        except Exception as e:
            # 读取失败返回全黑图，避免程序崩溃，后续可通过逻辑剔除
            return Image.new('RGB', (224, 224), (0, 0, 0))

def custom_collate(batch):
    # SentenceTransformer 需要 List[Image]，不需要 Tensor，所以原样返回
    return batch

# --- 2. 核心去重类 ---
class ImageDeduplicator:
    def __init__(self, model_path, threshold=0.95):
        print(f"正在加载模型: {os.path.basename(model_path)}...")
        if torch.backends.mps.is_available():
            self.device = 'mps'
        elif torch.cuda.is_available():
            self.device = 'cuda'
        else:
            self.device = 'cpu'
        print(f'[INFO] Device: {self.device}')
        
        self.model = SentenceTransformer(model_path, device=self.device)
        self.threshold = threshold

    def extract_features(self, image_paths):
        """提取特征 (使用 DataLoader 并行加速)"""
        print(f"正在提取 {len(image_paths)} 张图片的特征...")
        
        # 1. 创建数据集和加载器
        dataset = ImageDataset(image_paths)
        dataloader = DataLoader(
            dataset, 
            batch_size=16, 
            shuffle=False, 
            num_workers=12,      # <--- 核心加速点：8个进程同时读图
            collate_fn=custom_collate,
            pin_memory=True     # 加速 CPU -> GPU 传输
        )
        
        all_embeddings = []
        
        # 2. 批量推理
        for batch_images in tqdm(dataloader, desc="Encoding"):
            batch_emb = self.model.encode(
                batch_images, 
                batch_size=128, 
                show_progress_bar=False, 
                convert_to_numpy=True
            )
            all_embeddings.append(batch_emb)
            
            # 显式关闭图片对象
            for img in batch_images:
                img.close()

        if not all_embeddings:
            return np.array([])
            
        embeddings = np.vstack(all_embeddings)
        embeddings = embeddings.astype('float32')
        faiss.normalize_L2(embeddings)
        
        return embeddings
    
    def find_duplicates(self, image_paths):
        """核心逻辑：使用 Range Search (范围搜索) 替代 KNN"""
        embeddings = self.extract_features(image_paths)
        
        print("正在构建索引并进行范围搜索...")
        index = faiss.IndexFlatIP(embeddings.shape[1])
        index.add(embeddings)
        
        # --- 核心修改：使用 range_search ---
        # 不再限制 k=50，而是找出所有相似度 > threshold 的邻居
        lims, D, I = index.range_search(embeddings, self.threshold)

        print("正在构建图结构...")
        G = nx.Graph()
        G.add_nodes_from(range(len(image_paths)))
        
        # --- 优化：预读取文件大小 ---
        # 避免在后续排序循环中重复进行 IO 操作
        print("预读取文件大小以优化排序...")
        file_sizes = [os.path.getsize(p) for p in image_paths]

        # 解析 range_search 结果构建图
        for i in range(len(image_paths)):
            start = lims[i]
            end = lims[i+1]
            for j in range(start, end):
                neighbor_idx = I[j]
                # i < neighbor_idx 确保无向图边只添加一次，且排除自环
                if i < neighbor_idx:
                    G.add_edge(i, neighbor_idx)

        components = list(nx.connected_components(G))
        
        structured_results = []
        print(f"正在分析 {len(components)} 个连通分量...")
        
        for component in components:
            if len(component) > 1:
                # 在簇内按文件大小排序 (保留最大的)
                sorted_idx = sorted(list(component), key=lambda x: file_sizes[x], reverse=True)
                
                # --- 保留策略 ---
                # 默认：只保留 1 张最大的
                num_to_keep = 1 
                
                # 如果你想恢复之前的“每20张留1张”逻辑，取消下面这行的注释：
                # num_to_keep = max(1, (len(component) - 1) // 20 + 1)
                
                structured_results.append({
                    'keeps': [image_paths[i] for i in sorted_idx[:num_to_keep]],
                    'duplicates': [image_paths[i] for i in sorted_idx[num_to_keep:]]
                })
        
        return structured_results

# --- 3. 辅助功能函数 ---
def remove_duplicates(results, mode='move', backup_dir='./duplicates_backup'):
    if not results:
        print("✅ 没有需要删除的重复文件")
        return 0
    
    total_duplicates = sum(len(cluster['duplicates']) for cluster in results)
    
    print(f"\n{'='*60}")
    print(f"📊 去重统计:")
    print(f"   - 发现重复簇: {len(results)} 个")
    print(f"   - 重复文件总数: {total_duplicates} 个")
    print(f"   - 操作模式: {'🔒 安全移动' if mode == 'move' else '⚠️ 直接删除'}")
    print(f"{'='*60}\n")
    
    confirm = input(f"⚠️  确认要{('移动' if mode == 'move' else '删除')} {total_duplicates} 个重复文件吗? (yes/no): ")
    if confirm.lower() not in ['yes', 'y']:
        print("❌ 操作已取消")
        return 0
    
    if mode == 'move':
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_dir = f"{backup_dir}_{timestamp}"
        os.makedirs(backup_dir, exist_ok=True)
        print(f"📁 备份文件夹: {backup_dir}\n")
    
    removed_count = 0
    failed_files = []
    
    for i, cluster in enumerate(tqdm(results, desc="处理中")):
        for dup_path in cluster['duplicates']:
            try:
                if mode == 'move':
                    rel_path = os.path.basename(dup_path)
                    dest_path = os.path.join(backup_dir, f"cluster_{i+1}_{rel_path}")
                    shutil.move(dup_path, dest_path)
                elif mode == 'delete':
                    os.remove(dup_path)
                removed_count += 1
            except Exception as e:
                failed_files.append((dup_path, str(e)))
    
    print(f"\n✅ 成功处理: {removed_count}/{total_duplicates} 个文件")
    if failed_files:
        print(f"⚠️  {len(failed_files)} 个文件处理失败")
    
    return removed_count

def generate_html_report(results, threshold, output_html="dedup_report.html"):
    print(f"正在生成可视化报告...")
    report_abs_dir = os.path.dirname(os.path.abspath(output_html))
    
    html_template = f"""
    <html>
    <head>
        <meta charset="UTF-8">
        <title>图片去重报告 (Threshold: {threshold})</title>
        <style>
            body {{ font-family: sans-serif; background: #f8f9fa; padding: 20px; }}
            .cluster {{ background: white; padding: 20px; margin-bottom: 20px; border-radius: 8px; box-shadow: 0 2px 5px rgba(0,0,0,0.1); }}
            .image-grid {{ display: flex; flex-wrap: wrap; gap: 15px; }}
            .img-card {{ width: 150px; text-align: center; }}
            img {{ width: 100%; height: 150px; object-fit: cover; border-radius: 5px; border: 3px solid #eee; }}
            .keep-img {{ border-color: #28a745; }}
            .remove-img {{ border-color: #dc3545; opacity: 0.6; }}
            .badge {{ padding: 2px 6px; border-radius: 4px; color: white; font-size: 10px; }}
            .badge-keep {{ background: #28a745; }}
            .badge-remove {{ background: #dc3545; }}
        </style>
    </head>
    <body>
        <h1>去重报告 (阈值: {threshold})</h1>
        <p>发现 {len(results)} 组重复，共 {sum(len(c['duplicates']) for c in results)} 张待删除。</p>
    """

    for i, cluster in enumerate(results):
        html_template += f'<div class="cluster"><h3>Group {i+1}</h3><div class="image-grid">'
        
        for keep_path in cluster['keeps']:
            rel_path = os.path.relpath(keep_path, start=report_abs_dir)
            html_template += f"""
                <div class="img-card">
                    <span class="badge badge-keep">KEEP</span>
                    <img class="keep-img" src="{rel_path}">
                    <div style="font-size:10px">{os.path.basename(keep_path)}</div>
                </div>"""
        
        for dup_path in cluster['duplicates']:
            rel_path = os.path.relpath(dup_path, start=report_abs_dir)
            html_template += f"""
                <div class="img-card">
                    <span class="badge badge-remove">DEL</span>
                    <img class="remove-img" src="{rel_path}">
                    <div style="font-size:10px">{os.path.basename(dup_path)}</div>
                </div>"""
        html_template += '</div></div>'

    html_template += "</body></html>"
    with open(output_html, "w", encoding="utf-8") as f:
        f.write(html_template)
    print(f"✨ 报告已生成: {output_html}")

# --- 4. 主程序 ---
if __name__ == "__main__":
    # 切换目录
    os.chdir('/home/hadoop/data/deduplicate') 
    # 确认当前目录
    print("当前工作目录是:", os.getcwd())
    
    # 配置
    # MODEL_PATH = "/home/hadoop/.cache/modelscope/hub/models/sentence-transformers/clip-ViT-B-32"
    MODEL_PATH = "/home/hadoop/data/model_download/sentence-transformers/clip-ViT-B-32"
    # IMAGE_DIR = '/home/hadoop/data/feedcld/data/白样本'
    # IMAGE_DIR = '/home/hadoop/data/磁力球/data/磁力球-质检后'
    # IMAGE_DIR = '/home/hadoop/data/cldfeed/data/白样本'
    # IMAGE_DIR = '/home/hadoop/data/cldfeed/realdata/realdata_67w'
    # IMAGE_DIR = '/home/hadoop/data/cldfeed/IC_datas/IC_4class_round2_deduplicate/Ratio_30_1'
    IMAGE_DIR = '/home/hadoop/data/share/Tongying_dataset/Black/child_cloth_fire'
    OUTPUT_DIR = '/home/hadoop/data/deduplicate'
    
    THRESHOLD = 0.98

    # 扫描
    EXTS = ('.jpg', '.jpeg', '.png', '.webp', '.bmp', '.gif')
    all_images = []
    print(f"正在扫描: {IMAGE_DIR}")
    for root, dirs, files in os.walk(IMAGE_DIR):
        for file in files:
            if file.lower().endswith(EXTS):
                all_images.append(os.path.join(root, file))

    print(f"找到 {len(all_images)} 张图片")

    if all_images:

        
        
        deduper = ImageDeduplicator(MODEL_PATH, threshold=THRESHOLD)
        results = deduper.find_duplicates(all_images)
        
        if results:
            name = os.path.basename(IMAGE_DIR)
            current_dir = Path(__file__).parent.resolve()

            generate_html_report(results, THRESHOLD, output_html=f'{OUTPUT_DIR}/{name}_{THRESHOLD}.html')
            
            print("\n请选择操作模式:")
            print("1. 移动重复文件到备份 (推荐)")
            print("2. 直接删除")
            print("3. 退出")
            choice = input("输入选项: ").strip()
            
            if choice == '1':
                remove_duplicates(results, mode='move')
            elif choice == '2':
                remove_duplicates(results, mode='delete')
            elif choice == '3':
                print("✅ 已跳过删除操作")
            else:
                print("❌ 无效选项，已取消操作")
            # deduper1 = ImageDeduplicator(MODEL_PATH, threshold=0.96)
            # results1 = deduper.find_duplicates(all_images)
            # emove_duplicates(results1, mode='delete')
        else:
            print("✅ 未发现重复图片")
