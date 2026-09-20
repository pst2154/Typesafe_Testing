"""Head-only grounding fine-tune with deterministic, disjoint synthetic entities.

All frozen contextual features come from the pinned original Laya checkpoint.
Validation selects an epoch; the synthetic test split is scored only afterwards.
This is not a claim of general-document grounding quality.
"""
import copy
import argparse
import json
import random
from pathlib import Path
import torch
from safetensors.torch import save_file
from backends import LayaBackend

ROOT=Path(__file__).parent
FIELDS=[('retention period','14 days','90 days'),('listening port','443','8443'),
 ('maximum request size','8 MB','32 MB'),('cache lifetime','60 seconds','300 seconds'),
 ('default format','JSON','XML'),('transport','TCP','UDP'),
 ('compression','gzip','zstd'),('retry limit','2 attempts','5 attempts'),
 ('authentication method','API keys','client certificates')]

def examples(split,count,seed):
    rng=random.Random(seed)
    rows=[]
    for i in range(count):
        entity=f'{split.capitalize()}Service{i:03}'
        for field,a,b in FIELDS:
            query=f'What is the {field} of {entity}?'
            options={'a':f'{entity} has {field} {a}','b':f'{entity} has {field} {b}',
                     'insufficient_evidence':'Not stated in the passage'}
            for kind in ['a','b','insufficient_evidence']:
                if kind=='insufficient_evidence':
                    passage=f'{entity} is available. The {field} is not documented. OtherService has {field} {a}.'
                else:
                    value=a if kind=='a' else b
                    other=b if kind=='a' else a
                    passage=(f'Technical reference for {entity}. Its {field} is {value}, not {other}. '
                             f'OtherService has {field} {other}. These are separate services.')
                # Counterbalance label IDs as well as positions.
                keys=list(options)
                rng.shuffle(keys)
                criteria={k:options[k] for k in keys}
                questions={'decision':{'type':'choice',
                    'instructions':f'Based on this passage, {query} If not stated, select insufficient_evidence.',
                    'criteria':criteria}}
                rows.append({'split':split,'entity':entity,'state':passage,
                             'questions':questions,'label':kind,'target':keys.index(kind)})
    rng.shuffle(rows)
    return rows

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--reuse-features',action='store_true')
    parser.add_argument('--learning-rate',type=float,default=1e-5)
    parser.add_argument('--epochs',type=int,default=15)
    parser.add_argument('--output',default='finetuned')
    args=parser.parse_args()
    torch.manual_seed(31415)
    torch.set_num_threads(4)
    backend=LayaBackend()
    datasets={s:examples(s,n,seed) for s,n,seed in
              [('train',24,100),('validation',8,200),('test',8,300)]}
    dest=ROOT/args.output
    dest.mkdir(exist_ok=True)
    (dest/'examples.json').write_text(json.dumps(datasets,indent=2))
    captured=[]
    def hook(module,inputs): captured.append(inputs[0].detach().float().cpu())
    handle=backend.agent.model.scorer.register_forward_pre_hook(hook)
    tensors={}
    for split,rows in datasets.items():
        if args.reuse_features: break
        features=[]
        for i,row in enumerate(rows):
            if not backend.fits(row['state'],row['questions']): raise ValueError('Truncated training example')
            backend.predict(row['state'],row['questions'])
            features.append(captured.pop().squeeze(0))
            if i%100==0: print('features',split,i,'/',len(rows),flush=True)
        tensors[split]=(torch.stack(features),torch.tensor([r['target'] for r in rows]))
    handle.remove()
    if args.reuse_features:
        tensors=torch.load(ROOT/'finetuned'/'features.pt',weights_only=True)
    else:
        torch.save(tensors,dest/'features.pt')
    scorer=copy.deepcopy(backend.agent.model.scorer).float().cpu()
    original=copy.deepcopy(scorer.state_dict())
    optimizer=torch.optim.AdamW(scorer.parameters(),lr=args.learning_rate,weight_decay=.01)
    def accuracy(split):
        x,y=tensors[split]
        with torch.no_grad(): return (scorer(x).squeeze(-1).argmax(-1)==y).float().mean().item()
    baseline={s:accuracy(s) for s in tensors}
    best=baseline['validation']
    best_state=original
    history=[]
    x,y=tensors['train']
    for epoch in range(1,args.epochs+1):
        order=torch.randperm(len(y))
        total=0.
        for index in order.split(32):
            optimizer.zero_grad()
            loss=torch.nn.functional.cross_entropy(scorer(x[index]).squeeze(-1),y[index])
            loss.backward()
            torch.nn.utils.clip_grad_norm_(scorer.parameters(),1.)
            optimizer.step()
            total+=loss.item()*len(index)
        row={'epoch':epoch,'loss':total/len(y),'validation_accuracy':accuracy('validation')}
        history.append(row)
        if row['validation_accuracy']>best:
            best=row['validation_accuracy'];best_state=copy.deepcopy(scorer.state_dict())
        print(json.dumps(row),flush=True)
    scorer.load_state_dict(best_state)
    final={s:accuracy(s) for s in tensors}
    save_file({k:v.contiguous() for k,v in best_state.items()},str(dest/'scorer.safetensors'))
    report={'base_revision':'1c5edc17a7acd8701df6fc341c0d179f1c62c982',
            'trainable':'scorer only; encoder, contextual head, embeddings and act head frozen',
            'trainable_parameters':sum(p.numel() for p in scorer.parameters()),
            'examples':{s:len(v) for s,v in datasets.items()},'baseline':baseline,
            'final':final,'history':history,'seed':31415,'learning_rate':args.learning_rate,
            'limitation':'Synthetic templates with disjoint entity names, not unseen template families. Needs external document evaluation.'}
    (dest/'training.json').write_text(json.dumps(report,indent=2))
    print(json.dumps(report),flush=True)

if __name__=='__main__': main()
