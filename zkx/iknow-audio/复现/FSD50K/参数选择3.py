import sys
import os
import torch
import pandas as pd
import numpy as np
from tqdm import tqdm

sys.path.insert(0,"/home/star/zkx/CLAP/code/CLAP-main/src")
from laion_clap import CLAP_Module
from pykeen.triples import TriplesFactory

DEVICE='cuda' if torch.cuda.is_available() else 'cpu'

MODEL_PATH="/home/star/zkx/CLAP/model/630k-audioset-fusion-best.pt"

KG_TRIPLE="/home/star/zkx/iknow-audio/AKG_dataset/AKG_pruned.tsv"
KGE_MODEL="/home/star/zkx/iknow-audio/KGE_models/001/TFVpYwo2_RotatE_False/trained_model.pkl"

DEV_AUDIO="/home/star/zkx/iknow-audio/data/FSD50K-1/FSD50K.dev_audio"
DEV_CSV="/home/star/zkx/iknow-audio/data/FSD50K-1/FSD50K.ground_truth/dev.csv"
VOCAB="/home/star/zkx/iknow-audio/data/FSD50K-1/FSD50K.ground_truth/vocabulary.csv"


TOP_K = 5
TOP_REL = 1

RELATIONS = [
    'has parent',
    'co-occurs with',
    'occurs in'
]


def normalize(x):
    return x.lower().replace("_"," ")


print("Loading CLAP...")
clap=CLAP_Module(enable_fusion=True,amodel='HTSAT-tiny',tmodel='roberta')
clap.load_ckpt(MODEL_PATH)
clap.to(DEVICE)
clap.eval()


print("Loading KG...")
triples=TriplesFactory.from_path(KG_TRIPLE)

entity_to_id=triples.entity_to_id
relation_to_id=triples.relation_to_id
id_to_entity={v:k for k,v in entity_to_id.items()}

kge=torch.load(KGE_MODEL,map_location=DEVICE)
kge.eval()


print("Loading vocab...")
vocab=pd.read_csv(VOCAB,header=None,names=['id','label','mid'])

classes=vocab['label'].tolist()
class_to_idx={c:i for i,c in enumerate(classes)}


print("Classes:",len(classes))


matched=0
for c in classes:
    if normalize(c) in entity_to_id:
        matched+=1

print("KG matched:",matched)


print("Encoding text labels...")

prompts=[c.replace("_"," ") for c in classes]

text_embed=clap.get_text_embedding(prompts)
text_embed=torch.from_numpy(text_embed).to(DEVICE)
text_embed=text_embed/text_embed.norm(dim=-1,keepdim=True)



def kge_expand(label):

    label=normalize(label)

    if label not in entity_to_id:
        return []

    h=entity_to_id[label]

    tails=[]

    for r in RELATIONS:

        if r not in relation_to_id:
            continue

        r_id=relation_to_id[r]

        batch=torch.tensor([[h,r_id]],device=DEVICE)

        scores=kge.score_t(batch).squeeze()

        idx=torch.topk(scores,TOP_REL).indices.cpu().numpy()

        for i in idx:
            tails.append(id_to_entity[i])

    return tails



def evaluate(df,kg=False):

    hit1=0
    hit3=0
    hit5=0
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


        ranked=np.argsort(sim)[::-1].tolist()


        if kg:

            candidates=list(ranked[:TOP_K])

            for idx in ranked[:TOP_K]:

                label=classes[idx]

                tails=kge_expand(label)

                for t in tails:

                    t=t.replace("_"," ")

                    if t in class_to_idx:

                        candidates.append(class_to_idx[t])


            candidates=list(set(candidates))

            cand_scores=sim[candidates]

            order=np.argsort(cand_scores)[::-1]

            ranked=[candidates[i] for i in order]+[i for i in ranked if i not in candidates]


        rank=min(ranked.index(i)+1 for i in gt if i in ranked)


        if rank==1: hit1+=1
        if rank<=3: hit3+=1
        if rank<=5: hit5+=1

        mrr+=1/rank
        total+=1


    return hit1/total*100,hit3/total*100,hit5/total*100,mrr/total



print("Loading dev set...")
dev=pd.read_csv(DEV_CSV)

val=dev.sample(2000,random_state=0)



print("\nBASELINE")

h1,h3,h5,mrr=evaluate(val,kg=False)

print("Hit@1",h1)
print("Hit@3",h3)
print("Hit@5",h5)
print("MRR",mrr)



print("\nCLAP + KG")

h1,h3,h5,mrr=evaluate(val,kg=True)

print("Hit@1",h1)
print("Hit@3",h3)
print("Hit@5",h5)
print("MRR",mrr)