# /data/zkx/zkx/iknow-audio/9.新 Prompt+6./prompts.py
import os
import json
import torch
import pandas as pd
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
from pykeen.predict import predict_target
from pykeen.triples import TriplesFactory

# ==========================================
# 1. 全局路径与环境配置
# ==========================================
os.environ["CUDA_VISIBLE_DEVICES"] = "0" 
# 输出目录
OUTPUT_DIR = "/data/zkx/zkx/iknow-audio/9.新Prompt+6."
os.makedirs(OUTPUT_DIR, exist_ok=True)

# 模型路径
QWEN_PATH = "/data/zkx/zkx/iknow-audio/data/model/千问"
KGE_MODEL_DIR = "/home/star/zkx/iknow-audio/KGE_models/001/TFVpYwo2_RotatE_False"
TRAIN_TRIPLES_PATH = os.path.join(KGE_MODEL_DIR, "AKG_train_triples.tsv")

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

# ==========================================
# 2. 加载大语言模型 (Qwen) 与图谱模型 (PyKEEN)
# ==========================================
print("Loading Qwen 2.5 7B model from local...")
tokenizer = AutoTokenizer.from_pretrained(QWEN_PATH, local_files_only=True)
model = AutoModelForCausalLM.from_pretrained(QWEN_PATH, dtype=torch.float16, local_files_only=True).to(DEVICE)

print("Loading Knowledge Graph Embedding model...")
kge_model = torch.load(os.path.join(KGE_MODEL_DIR, "trained_model.pkl"), map_location=DEVICE)
kge_model.eval()
training_factory = TriplesFactory.from_path(TRAIN_TRIPLES_PATH)
AVAILABLE_RELATIONS = list(training_factory.relation_to_id.keys())

# ==========================================
# 3. 万能自适应声学大模型提示词 (System Prompt)
# ==========================================
SYSTEM_PROMPT = """You are a world-class audio researcher and acoustic descriptor. Your task is to transform a knowledge graph triple [head, relation, tail] into a single, highly descriptive English sentence for training an audio-language model.

You MUST dynamically adapt your description style based on the nature of the relationship:
1. If the relation implies an ACOUSTIC SCENE (e.g., 'scene contains', 'associated with environment', 'occurs in'): Describe the overall atmospheric soundscape, background noise, and spatial reverberation of the environment.
2. If the relation implies an AUDIO EVENT or CAUSALITY (e.g., 'produces', 'used for', 'indicates', 'overlaps with'): Describe the physical action, transients, impact, and rhythmic envelope of the specific sound source.
3. If the relation implies a TAXONOMY or HIERARCHY (e.g., 'belongs to class', 'is a type of', 'has parent'): Describe the defining acoustic traits (timbre, frequency range, pitch) that distinguish this specific subclass within its broader parent category.

Rules:
- Output ONLY the final sentence. No intro, no notes.
- The sentence MUST naturally contain the head, relation meaning, and tail.
- Strongly emphasize acoustic characteristics (timbre, pitch, rhythm, transient, frequency)."""

EXAMPLES = """【Example 1: Acoustic Scene】
Task: Graph: [bus, scene contains, engine rumble]
Model answer: The atmospheric acoustic scene of a bus typically contains the constant, low-frequency broadband rumble of an engine, accompanied by resonant spatial reverberations.

【Example 2: Taxonomy/Hierarchy】
Task: Graph: [dog barking, belongs to class, animal]
Model answer: Belonging to the broader class of animal sounds, a dog barking is defined by its sharp, harsh vocal transients and repetitive rhythmic envelope.

【Example 3: Audio Event / Causality】
Task: Graph: [ambulance siren, indicates, emergency]
Model answer: Indicating an emergency, the sound of an ambulance siren features a piercing, high-frequency tonal sweep with a cyclical, undulating pitch envelope."""

def generate_sentence(head, relation, tail):
    user_input = f"{EXAMPLES}\n\nNow provide answer for the next task yourself.\nTask: Graph: [{head}, {relation}, {tail}]\nModel answer:"
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_input}
    ]
    
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    model_inputs = tokenizer([text], return_tensors="pt").to(model.device)
    
    generated_ids = model.generate(**model_inputs, max_new_tokens=100, temperature=0.3, do_sample=True)
    generated_ids = [output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)]
    
    response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()
    response = response.split('\n')[0].replace("Model answer:", "").replace("\"", "").strip()
    return response

# ==========================================
# 4. 数据集类别解析与配置组装
# ==========================================
print("\nPreparing Datasets Configuration...")
datasets_config = []

# --- 1. ESC-50 ---
df_esc = pd.read_csv("/home/star/zkx/CLAP/data/ESC-50/esc50.csv")
esc_classes = sorted(df_esc['category'].unique())
datasets_config.append({
    "name": "ESC-50",
    "out_file": "esc50_llm_prompts.json",
    "orig_classes": esc_classes,
    "kg_entities": [c.replace('_', ' ').split('(')[0].strip() for c in esc_classes],
    "relations": [r for r in ['belongs to class', 'has parent', 'perceived as', 'event composed of', 'has children'] if r in AVAILABLE_RELATIONS]
})

# --- 2. US8K ---
df_us8k = pd.read_csv("/home/star/zkx/iknow-audio/data/UrbanSound8K/UrbanSound8K/metadata/UrbanSound8K.csv")
us8k_classes = sorted(df_us8k['class'].unique())
def get_us8k_entity(c):
    m = {'air_conditioner': 'air conditioner', 'car_horn': 'car horn', 'children_playing': 'children playing', 'dog_bark': 'dog barking', 'drilling': 'drilling', 'engine_idling': 'engine idling', 'gun_shot': 'gunshot', 'jackhammer': 'jackhammer', 'siren': 'siren', 'street_music': 'street music'}
    return m.get(c, c.replace('_', ' '))
datasets_config.append({
    "name": "US8K",
    "out_file": "us8k_llm_prompts.json",
    "orig_classes": us8k_classes,
    "kg_entities": [get_us8k_entity(c) for c in us8k_classes],
    "relations": [r for r in ['overlaps with', 'occurs in', 'associated with environment', 'localized in', 'used for', 'part of scene'] if r in AVAILABLE_RELATIONS]
})

# --- 3. TUT2017 (完美复原你的读取与映射逻辑) ---
TUT_DIR = "/home/star/zkx/iknow-audio/data/TUT2017/development/TUT-acoustic-scenes-2017-development"
TUT_META = os.path.join(TUT_DIR, "meta.txt")
data_records = []
if os.path.exists(TUT_META):
    with open(TUT_META, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 2:
                data_records.append({'class': parts[1]})
df_tut = pd.DataFrame(data_records)
tut_classes = sorted(df_tut['class'].unique()) if not df_tut.empty else []

def get_tut_entity(tut_class):
    mapping = {
        'cafe/restaurant': 'cafe or restaurant',
        'city_center': 'city center',
        'forest_path': 'forest path',
        'grocery_store': 'grocery store',
        'metro_station': 'metro station',
        'residential_area': 'residential area'
    }
    return mapping.get(tut_class, tut_class.replace('_', ' ').replace('/', ' or '))

datasets_config.append({
    "name": "TUT2017",
    "out_file": "tut2017_llm_prompts.json",
    "orig_classes": tut_classes,
    "kg_entities": [get_tut_entity(c) for c in tut_classes],
    "relations": [r for r in ['scene contains', 'event composed of', 'is variant of', 'has parent', 'described by'] if r in AVAILABLE_RELATIONS]
})

# --- 4. FSD50K ---
df_fsd = pd.read_csv("/home/star/zkx/iknow-audio/data/FSD50K-1/FSD50K.ground_truth/vocabulary.csv", header=None)
fsd_classes = df_fsd[1].tolist()
datasets_config.append({
    "name": "FSD50K",
    "out_file": "fsd50k_llm_prompts.json",
    "orig_classes": fsd_classes,
    "kg_entities": [c.replace('_', ' ').replace(' and ', ' ').lower() for c in fsd_classes],
    "relations": [r for r in ['belongs to class', 'has parent', 'is a type of'] if r in AVAILABLE_RELATIONS]
})


# ==========================================
# 5. 自动化批量生成与保存
# ==========================================
for ds in datasets_config:
    # 判空保护（防备文件路径错误导致空类别）
    if not ds['orig_classes']:
        print(f"⚠️ 警告: {ds['name']} 数据集未找到类别，跳过生成。")
        continue

    print(f"\n========================================")
    print(f"🚀 开始生成数据集: {ds['name']} (共 {len(ds['orig_classes'])} 个类别)")
    print(f"========================================")
    
    prompt_dict = {}
    class_mapping = zip(ds['orig_classes'], ds['kg_entities'])
    
    for orig_cls, kg_ent in tqdm(class_mapping, total=len(ds['orig_classes']), desc=f"Processing {ds['name']}"):
        for r in ds['relations']:
            try:
                # 🌟 Fallback 保护机制 (完美复现你的首词截断逻辑)
                query_ent = kg_ent
                if query_ent not in training_factory.entity_to_id:
                    query_ent = kg_ent.split(' ')[0]
                    if query_ent not in training_factory.entity_to_id:
                        continue
                
                # PyKEEN 提取 Top 3 尾实体
                pred = predict_target(model=kge_model, head=query_ent, relation=r, triples_factory=training_factory)
                tails = pred.df.sort_values(by="score", ascending=False).head(3)['tail_label'].tolist()
                
                for t in tails:
                    if t.lower() != orig_cls.lower():
                        dict_key = f"{orig_cls.replace('_', ' ')}||{r}||{t}"
                        if dict_key not in prompt_dict:
                            # 调用 Qwen 生成大师级声学句子
                            sentence = generate_sentence(orig_cls.replace('_', ' '), r, t)
                            prompt_dict[dict_key] = sentence
            except Exception as e:
                continue
                
    # 保存该数据集专属的 JSON
    out_path = os.path.join(OUTPUT_DIR, ds['out_file'])
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(prompt_dict, f, ensure_ascii=False, indent=4)
        
    print(f"✅ {ds['name']} 生成完毕！共提取了 {len(prompt_dict)} 条精美声学提示词。")
    print(f"📁 存储路径: {out_path}")

print("\n🎉 所有数据集的声学大语言模型提示词扩充已全部完成！")