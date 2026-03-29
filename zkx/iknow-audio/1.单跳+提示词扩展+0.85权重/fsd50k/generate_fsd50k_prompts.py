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
FSD_DIR = "/home/star/zkx/iknow-audio/data/FSD50K-1"
FSD_VOCAB = os.path.join(FSD_DIR, "FSD50K.ground_truth/vocabulary.csv")

KGE_MODEL_DIR = "/home/star/zkx/iknow-audio/KGE_models/001/TFVpYwo2_RotatE_False"
TRAIN_TRIPLES_PATH = os.path.join(KGE_MODEL_DIR, "AKG_train_triples.tsv")

# 输出的 JSON 路径
OUTPUT_DIR = "/home/star/zkx/iknow-audio/提示词扩展/fsd50k"
OUTPUT_JSON_PATH = os.path.join(OUTPUT_DIR, "llm_prompts.json")

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
TOP_M = 3

# ==========================================
# 2. FSD50K 专属映射清洗
# ==========================================
def get_kg_entity(fsd_class):
    clean_name = fsd_class.replace('_', ' ').replace(' and ', ' ').lower()
    return clean_name

def get_article(word):
    if not word: return ""
    return "an" if word[0].lower() in "aeiou" else "a"

# ==========================================
# 3. 本体树关系专属自然语言模板
# ==========================================
def generate_sentence(head, relation, tail):
    h = head.lower()
    t = tail.lower()
    r = relation.lower()
    
    art_h = get_article(h)
    
    # FSD50K 全是上下位继承关系，句式要体现出层级归属感
    if r == "belongs to class":
        return f"The sound of {h} belongs to the broader class of {t}."
    elif r == "has parent":
        return f"The acoustic category of {h} has a parent category of {t}."
    elif r == "is a type of":
        return f"{art_h.capitalize()} {h} is a specific type of {t}."
    else:
        return f"The sound of {h} {r} {t}."

# ==========================================
# 4. 主执行逻辑
# ==========================================
def main():
    print("🚀 开始提取知识图谱并生成 FSD50K 自然语言提示词...")
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # 加载 FSD50K 词汇表
    vocab_df = pd.read_csv(FSD_VOCAB, header=None)
    unique_categories = vocab_df[1].tolist()
    
    clean_classes = [cat.replace('_', ' ').replace(' and ', ', ') for cat in unique_categories]
    class_labels_set = set([c.lower() for c in clean_classes])
    
    # 加载图谱模型
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        kge_model = torch.load(os.path.join(KGE_MODEL_DIR, "trained_model.pkl"), map_location=DEVICE)
    kge_model.eval()
    training_factory = TriplesFactory.from_path(TRAIN_TRIPLES_PATH)
    
    # FSD50K 核心关系
    TARGET_RELATIONS = ['belongs to class', 'has parent', 'is a type of']
    VALID_RELATIONS = [r for r in TARGET_RELATIONS if r in training_factory.relation_to_id]
    
    prompts_dict = {}
    
    for fsd_cls in tqdm(unique_categories, desc="Generating Prompts"):
        kg_head = get_kg_entity(fsd_cls)
        query_entity = kg_head
        
        # fallback 逻辑保持与你的推断代码一致
        if query_entity not in training_factory.entity_to_id:
            query_entity = query_entity.split(' ')[-1] 
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