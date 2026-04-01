# build_prompt_dict.py
import os
import json
import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
from pykeen.predict import predict_target
from pykeen.triples import TriplesFactory
import pandas as pd

# ==========================================
# 1. 配置你的 Qwen 路径和 KGE 路径
# ==========================================
# 🌟 已经更新为你刚刚下载好的千问本地路径
os.environ["CUDA_VISIBLE_DEVICES"] = "0" 
QWEN_PATH = "/data/zkx/zkx/iknow-audio/data/model/千问"
KGE_MODEL_DIR = "/home/star/zkx/iknow-audio/KGE_models/001/TFVpYwo2_RotatE_False"
TRAIN_TRIPLES_PATH = os.path.join(KGE_MODEL_DIR, "AKG_train_triples.tsv")

# ==========================================
# 2. 加载 Qwen
# ==========================================
print("Loading Qwen model from local directory...")
# 使用 local_files_only=True 确保绝对不会连网，直接读本地文件
tokenizer = AutoTokenizer.from_pretrained(QWEN_PATH, local_files_only=True)
model = AutoModelForCausalLM.from_pretrained(QWEN_PATH, dtype=torch.float16, local_files_only=True).cuda()

# ==========================================
# 3. 加载 KGE 模型
# ==========================================
print("Loading KGE model...")
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
kge_model = torch.load(os.path.join(KGE_MODEL_DIR, "trained_model.pkl"), map_location=DEVICE)
kge_model.eval()
training_factory = TriplesFactory.from_path(TRAIN_TRIPLES_PATH)

# 针对 ESC-50 挑选的关系
TARGET_RELATIONS = ['belongs to class', 'has parent', 'perceived as', 'event composed of', 'has children']
VALID_RELATIONS = [r for r in TARGET_RELATIONS if r in training_factory.relation_to_id]

# ==========================================
# 4. 获取 ESC-50 所有类别
# ==========================================
ESC50_CSV = "/home/star/zkx/CLAP/data/ESC-50/esc50.csv" 
df = pd.read_csv(ESC50_CSV)
unique_categories = sorted(df['category'].unique())

# ==========================================
# 🌟 5. 专家级声学辨识度 Prompt 模板
# ==========================================
SYSTEM_PROMPT = """You are an expert audio describer and acoustic analyst. Given a knowledge graph triple (head, relation, tail) about an audio class, write a single, highly descriptive English sentence. 
Your description MUST naturally incorporate the head, relation, and tail. More importantly, you MUST imagine and describe the distinctive acoustic characteristics of the sound (such as its rhythm, pitch, timbre, transients, or frequency) to make it highly distinguishable from other sounds.
Output ONLY the final sentence. Do not include any introductory words, explanations, or notes."""

EXAMPLES = """【Example 1】
Task: Graph: [dog, belongs to class, animal]
Model answer: Belonging to the animal class, a dog bark is characterized by sharp, harsh transients and a repetitive rhythmic envelope.

【Example 2】
Task: Graph: [rain, perceived as, continuous]
Model answer: Perceived as continuous, rain produces a dense, broadband hissing timbre with random, overlapping droplet impacts.

【Example 3】
Task: Graph: [keyboard typing, occurs in, office]
Model answer: Occurring in an office environment, keyboard typing features rapid, crisp, and discrete clicking transients."""

def generate_sentence(head, relation, tail):
    user_input = f"{EXAMPLES}\n\nNow provide answer for the next task yourself.\nTask: Graph: [{head}, {relation}, {tail}]\nModel answer:"
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_input}
    ]
    
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    model_inputs = tokenizer([text], return_tensors="pt").to(model.device)
    
    # max_new_tokens=100 保证句子足够详细，temperature=0.3 让词汇更丰富
    generated_ids = model.generate(**model_inputs, max_new_tokens=100, temperature=0.3, do_sample=True)
    generated_ids = [output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)]
    
    response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()
    # 清理格式，只取最核心的第一句
    response = response.split('\n')[0].replace("Model answer:", "").strip()
    return response

# ==========================================
# 6. 遍历生成字典
# ==========================================
prompt_dict = {}
print("\n开始提取 KGE 路径并进行 Qwen 声学语义扩充...")

for cat in tqdm(unique_categories):
    orig_class_name = cat.replace('_', ' ').strip()
    kg_entity_name = orig_class_name.split('(')[0].strip() if '(' in orig_class_name else orig_class_name
    
    for r in VALID_RELATIONS:
        try:
            pred = predict_target(model=kge_model, head=kg_entity_name, relation=r, triples_factory=training_factory)
            tails = pred.df.sort_values(by="score", ascending=False).head(3)['tail_label'].tolist()
            
            for t in tails:
                if t.lower() != orig_class_name.lower():
                    dict_key = f"{orig_class_name}||{r}||{t}"
                    if dict_key not in prompt_dict:
                        # 核心：调用本地 Qwen 进行翻译和扩充
                        sentence = generate_sentence(orig_class_name, r, t)
                        prompt_dict[dict_key] = sentence
        except Exception as e:
            continue

# ==========================================
# 7. 保存为 JSON
# ==========================================
output_path = "llm_prompts_acoustic.json（2）"
with open(output_path, "w", encoding="utf-8") as f:
    json.dump(prompt_dict, f, ensure_ascii=False, indent=4)
print(f"\n✅ 专家级声学字典生成完毕！已保存至当前目录下的 {output_path}")