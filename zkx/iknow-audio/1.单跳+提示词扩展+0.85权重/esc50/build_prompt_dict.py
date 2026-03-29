# build_prompt_dict.py
import os
import json
import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
from pykeen.predict import predict_target
from pykeen.triples import TriplesFactory
import pandas as pd

# 1. 配置你的 Qwen 路径和 KGE 路径
QWEN_PATH = "/home/star/zkx/iknow-audio/提示词扩展/Qwen2.5-7B-Instruct"
KGE_MODEL_DIR = "/home/star/zkx/iknow-audio/KGE_models/001/TFVpYwo2_RotatE_False"
TRAIN_TRIPLES_PATH = os.path.join(KGE_MODEL_DIR, "AKG_train_triples.tsv")

# 2. 加载 Qwen
tokenizer = AutoTokenizer.from_pretrained(QWEN_PATH, local_files_only=True)
model = AutoModelForCausalLM.from_pretrained(QWEN_PATH, dtype=torch.float16, local_files_only=True).cuda()

# 3. 加载 KGE 模型 (复用你原来的逻辑)
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
kge_model = torch.load(os.path.join(KGE_MODEL_DIR, "trained_model.pkl"), map_location=DEVICE)
kge_model.eval()
training_factory = TriplesFactory.from_path(TRAIN_TRIPLES_PATH)
TARGET_RELATIONS = ['belongs to class', 'has parent', 'perceived as', 'event composed of', 'has children']
VALID_RELATIONS = [r for r in TARGET_RELATIONS if r in training_factory.relation_to_id]

# 4. 获取 ESC-50 所有类别
ESC50_CSV = "/home/star/zkx/CLAP/data/ESC-50/esc50.csv" # 换成你的路径
df = pd.read_csv(ESC50_CSV)
unique_categories = sorted(df['category'].unique())

# 神级 Prompt 模板
SYSTEM_PROMPT = """Act as a system which describes all nodes of the graph with edges as a connected text for audio classification. Follow the examples. Talk only about items from graph and use information only if graph contains it. Validate each written fact and correct it if mistake is found, do it silently without extra notes. Let's think step by step. For each step show described triple and check that all words from it is used in your description."""
EXAMPLES = "【Example 1】\nGraph: [siren, indicates, emergency]\nModel answer: A siren indicates an emergency.\n\n【Example 2】\nGraph: [helicopter, produces, engine_noise]\nModel answer: A helicopter produces engine noise."

def generate_sentence(head, relation, tail):
    user_input = f"{EXAMPLES}\nNow provide answer for the next task yourself.\nTask:\nGraph: [{head}, {relation}, {tail}]\nModel answer:"
    messages = [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": user_input}]
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    model_inputs = tokenizer([text], return_tensors="pt").to(model.device)
    generated_ids = model.generate(**model_inputs, max_new_tokens=100, temperature=0.1, do_sample=True)
    generated_ids = [output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)]
    return tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()

# 5. 遍历生成字典
prompt_dict = {}
print("开始提取所有的 KGE 路径并生成文本...")

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
                        # 调用 Qwen 翻译
                        sentence = generate_sentence(orig_class_name, r, t)
                        prompt_dict[dict_key] = sentence
        except Exception as e:
            continue

# 保存为 JSON
with open("llm_prompts.json", "w", encoding="utf-8") as f:
    json.dump(prompt_dict, f, ensure_ascii=False, indent=4)
print("✅ 字典生成完毕！已保存至 llm_prompts.json")