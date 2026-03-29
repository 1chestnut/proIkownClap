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
KGE_MODEL_DIR = "/home/star/zkx/iknow-audio/KGE_models/001/TFVpYwo2_RotatE_False"
TRAIN_TRIPLES_PATH = os.path.join(KGE_MODEL_DIR, "AKG_train_triples.tsv")

# 输出的 JSON 路径
OUTPUT_DIR = "/home/star/zkx/iknow-audio/提示词扩展/dcase17"
OUTPUT_JSON_PATH = os.path.join(OUTPUT_DIR, "llm_prompts.json")

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
TOP_M = 3

# ==========================================
# 2. DCASE 类别与映射
# ==========================================
DCASE_17_CLASSES = [
    'ambulance (siren)', 'bicycle', 'bus', 'car', 'car alarm', 'car passing by', 
    'civil defense siren', 'fire engine, fire truck (siren)', 'motorcycle', 
    'police car (siren)', 'reversing beeps', 'screaming', 'skateboard', 
    'train', 'train horn', 'truck', 'air horn, truck horn'
]

def get_kg_entity(class_name):
    mapping = {
        'ambulance (siren)': 'ambulance',
        'fire engine, fire truck (siren)': 'fire engine',
        'police car (siren)': 'police car',
        'air horn, truck horn': 'horn',
        'civil defense siren': 'siren',
        'reversing beeps': 'beep',
        'car passing by': 'car'
    }
    return mapping.get(class_name, class_name)

class_labels_set = set(DCASE_17_CLASSES)

def get_article(word):
    if not word: return ""
    return "an" if word[0].lower() in "aeiou" else "a"

# ==========================================
# 3. 警告/车辆音专属自然语言模板
# ==========================================
def generate_sentence(head, relation, tail):
    h = head.lower()
    t = tail.lower()
    r = relation.lower()
    
    art_h = get_article(h)
    
    if r == "indicates":
        return f"The sound of {art_h} {h} typically indicates {t}."
    elif r == "described by":
        return f"The sound of {art_h} {h} can be described by {t}."
    elif r == "used for":
        return f"{art_h.capitalize()} {h} is generally used for {t}."
    elif r == "associated with environment":
        return f"The sound of {art_h} {h} is heavily associated with a {t} environment."
    elif r == "has parent":
        return f"The sound of {art_h} {h} belongs to the broader acoustic category of {t}."
    elif r == "is instance of":
        return f"{art_h.capitalize()} {h} is a specific instance of {t}."
    else:
        return f"The sound of {h} {r} {t}."

# ==========================================
# 4. 主执行逻辑
# ==========================================
def main():
    print("🚀 开始提取知识图谱并生成 DCASE17 自然语言提示词...")
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        kge_model = torch.load(os.path.join(KGE_MODEL_DIR, "trained_model.pkl"), map_location=DEVICE)
    kge_model.eval()
    training_factory = TriplesFactory.from_path(TRAIN_TRIPLES_PATH)
    
    TARGET_RELATIONS = ['indicates', 'described by', 'used for', 'associated with environment', 'has parent', 'is instance of']
    VALID_RELATIONS = [r for r in TARGET_RELATIONS if r in training_factory.relation_to_id]
    
    prompts_dict = {}
    
    for dcase_cls in tqdm(DCASE_17_CLASSES, desc="Generating Prompts"):
        kg_head = get_kg_entity(dcase_cls)
        query_entity = kg_head
        
        if query_entity not in training_factory.entity_to_id:
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