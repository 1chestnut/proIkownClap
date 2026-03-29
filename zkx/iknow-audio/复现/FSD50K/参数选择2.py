import sys
import os
import pandas as pd
import numpy as np
import torch
from tqdm import tqdm

sys.path.insert(0,"/home/star/zkx/CLAP/code/CLAP-main/src")
from laion_clap import CLAP_Module
from pykeen.triples import TriplesFactory

DEVICE='cuda' if torch.cuda.is_available() else 'cpu'

MODEL_PATH="/home/star/zkx/CLAP/model/630k-audioset-fusion-best.pt"

KG_TRIPLE_FILE="/home/star/zkx/iknow-audio/AKG_dataset/AKG_pruned.tsv"
KGE_MODEL="/home/star/zkx/iknow-audio/KGE_models/001/TFVpYwo2_RotatE_False/trained_model.pkl"

DEV_AUDIO="/home/star/zkx/iknow-audio/data/FSD50K-1/FSD50K.dev_audio"
DEV_CSV="/home/star/zkx/iknow-audio/data/FSD50K-1/FSD50K.ground_truth/dev.csv"
VOCAB="/home/star/zkx/iknow-audio/data/FSD50K-1/FSD50K.ground_truth/vocabulary.csv"

TOP_K_CLAP=10
TOP_M_PER_REL=1
ALPHA=0.8

RELATION_SETS=[
['has parent'],
['occurs in'],
['co-occurs with'],
['has parent','occurs in'],
['has parent','occurs in','co-occurs with']
]

def normalize(x):
    return x.lower().replace("_"," ")

print("Loading CLAP...")
clap=CLAP_Module(enable_fusion=True,amodel='HTSAT-tiny',tmodel='roberta')
clap.load_ckpt(MODEL_PATH)
clap.to(DEVICE)
clap.eval()

print("Loading KG...")
triples=TriplesFactory.from_path(KG_TRIPLE_FILE)
entity_to_id=triples.entity_to_id
relation_to_id=triples.relation_to_id
id_to_entity={v:k for k,v in entity_to_id.items()}

model=torch.load(KGE_MODEL,map_location=DEVICE)
model.eval()

print("Loading vocab...")
vocab=pd.read_csv(VOCAB,header=None,names=["id","label","mid"])
class_labels=vocab["label"].tolist()
class_to_idx={l:i for i,l in enumerate(class_labels)}

print("Classes:",len(class_labels))

matched=0
for c in class_labels:
    if normalize(c) in entity_to_id:
        matched+=1

print("KG matched:",matched)

print("Computing text embeddings...")
prompts=[f"This is a sound of {l.replace('_',' ')}." for l in class_labels]

text_embed=clap.get_text_embedding(prompts)
text_embed=torch.from_numpy(text_embed).to(DEVICE)
text_embed=text_embed/text_embed.norm(dim=-1,keepdim=True)

def kge_predict(head,rel):

    head=normalize(head)

    if head not in entity_to_id:
        return []

    if rel not in relation_to_id:
        return []

    h=entity_to_id[head]
    r=relation_to_id[rel]

    batch=torch.tensor([[h,r]],device=DEVICE)

    scores=model.score_t(batch)
    scores=scores.squeeze()

    idx=torch.topk(scores,TOP_M_PER_REL).indices.cpu().numpy()

    return [id_to_entity[i] for i in idx]

def evaluate(df,relations):

    hit1=hit3=hit5=0
    mrr=0
    total=0

    for _,row in tqdm(df.iterrows(),total=len(df)):

        fname=str(row["fname"])
        labels=row["labels"].split(",")

        gt=[]
        for l in labels:
            l=l.strip()
            if l in class_to_idx:
                gt.append(class_to_idx[l])

        if len(gt)==0:
            continue

        audio=os.path.join(DEV_AUDIO,fname+".wav")

        if not os.path.exists(audio):
            continue

        audio_embed=clap.get_audio_embedding_from_filelist([audio])
        audio_embed=torch.from_numpy(audio_embed).to(DEVICE)
        audio_embed=audio_embed/audio_embed.norm(dim=-1,keepdim=True)

        sim=torch.matmul(audio_embed,text_embed.T).squeeze().cpu().numpy()

        if relations is None:

            new_scores=sim

        else:

            new_scores=sim.copy()

            top_idx=np.argsort(sim)[::-1][:TOP_K_CLAP]

            for idx in top_idx:

                label=class_labels[idx]

                enhanced=[]

                for r in relations:

                    tails=kge_predict(label,r)

                    for t in tails:

                        prompt=f"{label} {t}"

                        emb=clap.get_text_embedding([prompt])
                        emb=torch.from_numpy(emb).to(DEVICE)
                        emb=emb/emb.norm(dim=-1,keepdim=True)

                        s=torch.matmul(audio_embed,emb.T).item()

                        enhanced.append(s)

                if len(enhanced)>0:

                    avg=np.mean(enhanced)

                    new_scores[idx]=ALPHA*sim[idx]+(1-ALPHA)*avg

        ranked=np.argsort(new_scores)[::-1]

        rank=min([np.where(ranked==i)[0][0]+1 for i in gt])

        if rank==1:
            hit1+=1

        if rank<=3:
            hit3+=1

        if rank<=5:
            hit5+=1

        mrr+=1/rank
        total+=1

    return hit1/total*100,hit3/total*100,hit5/total*100,mrr/total

print("Loading dev set...")

dev=pd.read_csv(DEV_CSV)

val=dev.sample(2000,random_state=0)

print("\n========== BASELINE ==========")

h1,h3,h5,mrr=evaluate(val,None)

print("Baseline")
print("Hit@1",h1)
print("Hit@3",h3)
print("Hit@5",h5)
print("MRR",mrr)

print("\n========== KG ==========")

best=0
best_rel=None

for rel in RELATION_SETS:

    h1,h3,h5,mrr=evaluate(val,rel)

    print("\nRelations:",rel)
    print("Hit@1",h1)
    print("MRR",mrr)

    if h1>best:
        best=h1
        best_rel=rel

print("\nBEST RELATION SET")
print(best_rel)