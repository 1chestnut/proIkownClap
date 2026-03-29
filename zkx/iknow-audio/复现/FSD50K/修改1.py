import sys
import os
import pandas as pd
import numpy as np
import torch
from tqdm import tqdm
from rapidfuzz import process

sys.path.insert(0, "/home/star/zkx/CLAP/code/CLAP-main/src")
from laion_clap import CLAP_Module
from pykeen.triples import TriplesFactory

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

MODEL_PATH = "/home/star/zkx/CLAP/model/630k-audioset-fusion-best.pt"
KGE_MODEL_DIR = "/home/star/zkx/iknow-audio/KGE_models/001/TFVpYwo2_RotatE_False"

FSD50K_DEV_AUDIO = "/home/star/zkx/iknow-audio/data/FSD50K-1/FSD50K.dev_audio"
FSD50K_DEV_CSV = "/home/star/zkx/iknow-audio/data/FSD50K-1/FSD50K.ground_truth/dev.csv"
FSD50K_VOCAB_CSV = "/home/star/zkx/iknow-audio/data/FSD50K-1/FSD50K.ground_truth/vocabulary.csv"

TOP_K_CLAP = 5
TOP_M_PER_REL = 2

RELATIONS = [
    'has parent',
    'occurs in',
    'co-occurs with'
]


def normalize(x):
    return x.lower().replace('_', ' ')


def match_entity(label, kg_entities):

    label = normalize(label)

    if label in kg_entities:
        return label

    match = process.extractOne(label, kg_entities)

    if match and match[1] > 85:
        return match[0]

    return None


def load_kge():

    triples_file = os.path.join(KGE_MODEL_DIR,'AKG_train_triples.tsv')
    triples_factory = TriplesFactory.from_path(triples_file)

    model_path = os.path.join(KGE_MODEL_DIR,'trained_model.pkl')
    model = torch.load(model_path,map_location=DEVICE)

    model.eval()

    return triples_factory, model


def kge_predict(head,relation,model,triples_factory):

    entity_to_id = triples_factory.entity_to_id
    relation_to_id = triples_factory.relation_to_id
    id_to_entity = {v:k for k,v in entity_to_id.items()}

    if head not in entity_to_id:
        return []

    if relation not in relation_to_id:
        return []

    h = entity_to_id[head]
    r = relation_to_id[relation]

    batch = torch.tensor([[h,r]],device=DEVICE)

    scores = model.score_t(batch).squeeze()

    top_ids = torch.topk(scores,TOP_M_PER_REL).indices.cpu().numpy()

    return [id_to_entity[i] for i in top_ids]


def main():

    print("load CLAP")

    clap = CLAP_Module(enable_fusion=True,amodel='HTSAT-tiny',tmodel='roberta')
    clap.load_ckpt(MODEL_PATH)
    clap.to(DEVICE)
    clap.eval()

    triples_factory,kge = load_kge()

    vocab = pd.read_csv(FSD50K_VOCAB_CSV,header=None,names=['id','label','mid'])

    class_labels = vocab['label'].tolist()

    class_to_idx = {l:i for i,l in enumerate(class_labels)}

    print("classes:",len(class_labels))

    entity_to_id = triples_factory.entity_to_id
    kg_entities = list(entity_to_id.keys())

    matched = 0
    label2kg = {}

    for l in class_labels:

        m = match_entity(l,kg_entities)

        if m:
            matched+=1
            label2kg[l]=m

    print("KG matched:",matched)

    prompts = [f"This is a sound of {normalize(l)}." for l in class_labels]

    text_embed = clap.get_text_embedding(prompts)
    text_embed = torch.from_numpy(text_embed).to(DEVICE)
    text_embed = text_embed / text_embed.norm(dim=-1,keepdim=True)

    df = pd.read_csv(FSD50K_DEV_CSV)

    val_df = df.sample(n=2000,random_state=0)

    hit1=0
    hit3=0
    hit5=0
    rr=0
    total=0

    for _,row in tqdm(val_df.iterrows(),total=len(val_df)):

        fname=str(row['fname'])

        path=os.path.join(FSD50K_DEV_AUDIO,fname+'.wav')

        if not os.path.exists(path):
            continue

        labels=row['labels'].split(',')

        gt=[class_to_idx[l] for l in labels if l in class_to_idx]

        audio=clap.get_audio_embedding_from_filelist([path])
        audio=torch.from_numpy(audio).to(DEVICE)
        audio=audio/audio.norm(dim=-1,keepdim=True)

        sim=torch.matmul(audio,text_embed.T).squeeze().cpu().numpy()

        topk=np.argsort(sim)[::-1][:TOP_K_CLAP]

        new_scores=sim.copy()

        for idx in topk:

            label=class_labels[idx]

            if label not in label2kg:
                continue

            head=label2kg[label]

            enhance=[]

            for r in RELATIONS:

                tails=kge_predict(head,r,kge,triples_factory)

                for t in tails:

                    prompt=f"This is a sound of {normalize(label)} related to {t}"

                    pe=clap.get_text_embedding([prompt])

                    pe=torch.from_numpy(pe).to(DEVICE)
                    pe=pe/pe.norm(dim=-1,keepdim=True)

                    s=torch.matmul(audio,pe.T).item()

                    enhance.append(s)

            if enhance:

                new_scores[idx]=0.7*sim[idx]+0.3*np.mean(enhance)

        ranked=np.argsort(new_scores)[::-1]

        ranks=[np.where(ranked==g)[0][0]+1 for g in gt]

        rank=min(ranks)

        if rank==1:
            hit1+=1

        if rank<=3:
            hit3+=1

        if rank<=5:
            hit5+=1

        rr+=1/rank
        total+=1

    print()

    print("RESULT")

    print("Hit@1",hit1/total*100)
    print("Hit@3",hit3/total*100)
    print("Hit@5",hit5/total*100)
    print("MRR",rr/total)


if __name__ == "__main__":
    main()