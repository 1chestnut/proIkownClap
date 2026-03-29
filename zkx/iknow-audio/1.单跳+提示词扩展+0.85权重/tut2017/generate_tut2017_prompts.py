import os
import json
import torch
import warnings
import pandas as pd
from pykeen.predict import predict_target
from pykeen.triples import TriplesFactory
from tqdm import tqdm

# ==========================================
# 1. 路径定义
# ==========================================
TUT_DIR = "/home/star/zkx/iknow-audio/data/TUT2017/development/TUT-acoustic-scenes-2017-development"
TUT_META = os.path.join(TUT_DIR, "meta.txt")

KGE_MODEL_DIR = "/home/star/zkx/iknow-audio/KGE_models/001/TFVpYwo2_RotatE_False"
TRAIN_TRIPLES_PATH = os.path.join(KGE_MODEL_DIR, "AKG_train_triples.tsv")

# 输出的 JSON 路径
OUTPUT_DIR = "/home/star/zkx/iknow-audio/提示词扩展/tut2017"
OUTPUT_JSON_PATH = os.path.join(OUTPUT_DIR, "llm_prompts.json")

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
TOP_M = 5

# ==========================================
# 2. TUT2017 专属映射
# ==========================================
def get_kg_entity(tut_class):
    mapping = {
        'cafe/restaurant': 'cafe or restaurant',
        'city_center': 'city center',
        'forest_path': 'forest path',
        'grocery_store': 'grocery store',
        'metro_station': 'metro station',
        'residential_area': 'residential area'
    }
    return mapping.get(tut_class, tut_class.replace('_', ' ').replace('/', ' or '))

# ==========================================
# 3. 场景专属自然语言模板
# ==========================================
def generate_sentence(head, relation, tail):
    h = head.lower()
    t = tail.lower()
    r = relation.lower()
    
    # 针对 TUT2017 "声学场景" 专属定制的高级句式
    if r == "scene contains":
        return f"The acoustic scene of a {h} typically contains the sound of {t}."
    elif r == "event composed of":
        return f"The audio environment of a {h} is largely composed of {t} sounds."
    elif r == "is variant of":
        return f"A {h} is an acoustic variant of a {t} environment."
    elif r == "has parent":
        return f"The {h} environment belongs to the broader acoustic category of {t}."
    elif r == "described by":
        return f"The auditory experience of a {h} can be described by {t}."
    else:
        return f"The scene of {h} {r} {t}."

# ==========================================
# 4. 主执行逻辑
# ==========================================
def main():
    print("🚀 开始提取知识图谱并生成 TUT2017 场景自然语言提示词...")
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # 获取类别
    data_records = []
    with open(TUT_META, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 2:
                data_records.append({'class': parts[1]})
    df = pd.DataFrame(data_records)
    unique_categories = sorted(df['class'].unique())
    clean_classes = [cat.replace('_', ' ').replace('/', ' or ') for cat in unique_categories]
    class_labels_set = set([c.lower() for c in clean_classes])
    
    # 加载模型
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        kge_model = torch.load(os.path.join(KGE_MODEL_DIR, "trained_model.pkl"), map_location=DEVICE)
    kge_model.eval()
    training_factory = TriplesFactory.from_path(TRAIN_TRIPLES_PATH)
    
    TARGET_RELATIONS = ['is variant of', 'has parent', 'scene contains', 'event composed of', 'described by']
    VALID_RELATIONS = [r for r in TARGET_RELATIONS if r in training_factory.relation_to_id]
    
    prompts_dict = {}
    
    for tut_cls in tqdm(unique_categories, desc="Generating Prompts"):
        kg_head = get_kg_entity(tut_cls)
        query_entity = kg_head
        if query_entity not in training_factory.entity_to_id:
            fallback = query_entity.split(' ')[0] 
            if fallback in training_factory.entity_to_id:
                query_entity = fallback
            else:
                continue
                
        for rel in VALID_RELATIONS:
            try:
                pred = predict_target(model=kge_model, head=query_entity, relation=rel, triples_factory=training_factory)
                tails = pred.df.sort_values(by="score", ascending=False).head(TOP_M)['tail_label'].tolist()
                
                for tail in tails:
                    t_clean = tail.lower().strip()
                    h_clean = kg_head.lower().strip()
                    
                    if t_clean == h_clean or t_clean in class_labels_set:
                        continue
                        
                    key = f"{h_clean}||{rel.lower()}||{t_clean}"
                    prompts_dict[key] = generate_sentence(h_clean, rel, t_clean)
            except: pass
                
    print(f"\n✅ 生成完毕！共提取了 {len(prompts_dict)} 条有效自然语言提示词。")
    with open(OUTPUT_JSON_PATH, 'w', encoding='utf-8') as f:
        json.dump(prompts_dict, f, indent=4, ensure_ascii=False)

if __name__ == "__main__":
    main()