import os
import json
import torch
import warnings
from pykeen.predict import predict_target
from pykeen.triples import TriplesFactory
from tqdm import tqdm

# ==========================================
# 1. 路径定义
# ==========================================
KGE_MODEL_DIR = "/home/star/zkx/iknow-audio/KGE_models/001/TFVpYwo2_RotatE_False"
TRAIN_TRIPLES_PATH = os.path.join(KGE_MODEL_DIR, "AKG_train_triples.tsv")

# 输出的 JSON 路径
OUTPUT_DIR = "/home/star/zkx/iknow-audio/提示词扩展/us8k"
OUTPUT_JSON_PATH = os.path.join(OUTPUT_DIR, "llm_prompts.json")

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

# 提取前 M 个预测的尾实体来生成句子 (为了保证覆盖率，这里稍微取大一点，比如 5)
TOP_M = 5

# ==========================================
# 2. US8K 类别与映射
# ==========================================
US8K_CLASSES = [
    'air_conditioner', 'car_horn', 'children_playing', 'dog_bark', 
    'drilling', 'engine_idling', 'gun_shot', 'jackhammer', 
    'siren', 'street_music'
]

def get_kg_entity(us8k_class):
    mapping = {
        'air_conditioner': 'air conditioner',
        'car_horn': 'car horn',
        'children_playing': 'children playing',
        'dog_bark': 'dog barking', 
        'drilling': 'drilling',
        'engine_idling': 'engine idling',
        'gun_shot': 'gunshot',
        'jackhammer': 'jackhammer',
        'siren': 'siren',
        'street_music': 'street music'
    }
    return mapping.get(us8k_class, us8k_class.replace('_', ' '))

# US8K 的基准类名集合（防泄漏用）
class_labels_set = set([get_kg_entity(c) for c in US8K_CLASSES])

# ==========================================
# 3. 自然语言生成模板
# ==========================================
def get_article(word):
    """简单的冠词判断器：元音开头用 an，辅音用 a"""
    if not word: return ""
    return "an" if word[0].lower() in "aeiou" else "a"

def generate_sentence(head, relation, tail):
    """根据关系将三元组转化为自然语言句子"""
    # 统一小写
    h = head.lower()
    t = tail.lower()
    r = relation.lower()
    
    art_h = get_article(h)
    art_t = get_article(t)
    
    # 针对 US8K 专属关系的句式模板
    if r == "overlaps with":
        return f"The sound of {h} often overlaps with {t}."
    elif r == "occurs in":
        return f"{art_h.capitalize()} {h} typically occurs in {art_t} {t}."
    elif r == "associated with environment":
        return f"The sound of {art_h} {h} is heavily associated with a {t} environment."
    elif r == "localized in":
        return f"{art_h.capitalize()} {h} is usually localized in {art_t} {t}."
    elif r == "used for":
        return f"{art_h.capitalize()} {h} is generally used for {t}."
    elif r == "part of scene":
        return f"The sound of {art_h} {h} is a common part of a {t} scene."
    else:
        # 兜底通用模板
        return f"{art_h.capitalize()} {h} {r} {t}."

# ==========================================
# 4. 主执行逻辑
# ==========================================
def main():
    print("🚀 开始提取知识图谱并生成 US8K 自然语言提示词...")
    
    # 确保输出目录存在
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # 加载模型和工厂
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        kge_model = torch.load(os.path.join(KGE_MODEL_DIR, "trained_model.pkl"), map_location=DEVICE)
    kge_model.eval()
    training_factory = TriplesFactory.from_path(TRAIN_TRIPLES_PATH)
    
    # 定义需要的关系
    TARGET_RELATIONS = [
        'overlaps with', 'occurs in', 'associated with environment', 
        'localized in', 'used for', 'part of scene'
    ]
    VALID_RELATIONS = [r for r in TARGET_RELATIONS if r in training_factory.relation_to_id]
    
    prompts_dict = {}
    
    # 遍历所有类别
    for us8k_cls in tqdm(US8K_CLASSES, desc="Generating Prompts"):
        kg_head = get_kg_entity(us8k_cls)
        
        # 处理头实体不在 KG 中的情况 (取第一个单词作为 fallback)
        query_entity = kg_head
        if query_entity not in training_factory.entity_to_id:
            fallback = query_entity.split(' ')[0] 
            if fallback in training_factory.entity_to_id:
                query_entity = fallback
            else:
                continue # 如果实在找不到就跳过
                
        # 遍历所有关系
        for rel in VALID_RELATIONS:
            try:
                # 预测尾实体
                pred = predict_target(
                    model=kge_model, 
                    head=query_entity, 
                    relation=rel, 
                    triples_factory=training_factory
                )
                tails = pred.df.sort_values(by="score", ascending=False).head(TOP_M)['tail_label'].tolist()
                
                for tail in tails:
                    t_clean = tail.lower().strip()
                    h_clean = kg_head.lower().strip()
                    
                    # 过滤掉自己或者是其他 US8K 核心类名，防止泄漏和同义反复
                    if t_clean == h_clean or t_clean in class_labels_set:
                        continue
                        
                    # 组装 JSON 键： "head||relation||tail"
                    key = f"{h_clean}||{rel.lower()}||{t_clean}"
                    
                    # 生成自然语言句子
                    sentence = generate_sentence(h_clean, rel, t_clean)
                    
                    prompts_dict[key] = sentence
                    
            except Exception as e:
                pass # 忽略预测失败的冷门关系
                
    # 保存为 JSON
    print(f"\n✅ 生成完毕！共提取了 {len(prompts_dict)} 条有效自然语言提示词。")
    print(f"📁 正在保存至: {OUTPUT_JSON_PATH}")
    
    with open(OUTPUT_JSON_PATH, 'w', encoding='utf-8') as f:
        json.dump(prompts_dict, f, indent=4, ensure_ascii=False)
        
    # 随便打印几条看看效果
    print("\n🧐 随机预览几条生成结果:")
    preview_keys = list(prompts_dict.keys())[:5]
    for k in preview_keys:
        print(f"  Key : {k}")
        print(f"  Text: {prompts_dict[k]}\n")

if __name__ == "__main__":
    main()