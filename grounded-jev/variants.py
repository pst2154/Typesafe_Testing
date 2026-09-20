"""Experimental one-pass evidence-first decision paths."""
from time import perf_counter
from engine import Grounded, UNKNOWN
from context_probes import ask

class OnePass(Grounded):
    def __init__(self,corpus,backend,mode='entailment'):
        super().__init__(corpus,backend)
        self.mode=mode

    def decide(self,query,criteria,grounded=True):
        self.validate(query,criteria)
        started=perf_counter()
        search_query=query
        if self.mode in ('expanded','diverse'):
            search_query+=' '+ ' '.join(criteria.values())
        retrieved=self.corpus.search(search_query,k=4) if grounded else []
        if grounded and self.mode=='diverse':
            direct=self.corpus.search(query,k=4)
            seen=set()
            combined=[]
            for pair in zip(direct,retrieved):
                for c in pair:
                    if c['id'] not in seen:
                        combined.append(c);seen.add(c['id'])
            retrieved=combined[:4]
        retrieval_ms=(perf_counter()-started)*1000
        used=retrieved[:2]
        # Packing uses the same shape that ask() sends to the backend.
        if self.mode=='entailment':
            questions={k:{'type':'choice',
                'instructions':f'Does this passage entail the statement: {claim}?',
                'criteria':{'entailment':'The passage supports the statement',
                            'contradiction':'The passage contradicts the statement',
                            'neutral':'The passage does not say'}} for k,claim in criteria.items()}
        else:
            questions={'decision':{'type':'choice',
                'instructions':f'Based on this passage, {query} If not stated, select insufficient_evidence.',
                'criteria':dict(criteria,insufficient_evidence='Not stated in the passage')}}
        def evidence(): return '\n\n'.join(c['text'] for c in used)
        while used and not self.backend.fits(evidence(),questions): used.pop()
        if not self.backend.fits(evidence(),questions):
            raise ValueError('Question exceeds model token budget')
        before=perf_counter()
        mode='plain' if self.mode in ('expanded','diverse') else self.mode
        choice,answers=ask(self.backend,mode,query,criteria,evidence())
        if choice not in {*criteria,UNKNOWN}:
            raise ValueError('Backend returned invalid choice')
        if not used: choice=UNKNOWN
        return {'model':self.backend.name,'variant':self.mode,'choice':choice,
                'answer':criteria.get(choice),'abstained':choice==UNKNOWN,
                'raw_decision':answers,'verification':None,
                'citations':used if choice!=UNKNOWN else [],
                'retrieved_ids':[c['id'] for c in retrieved], 'used_ids':[c['id'] for c in used],
                'timing_ms':{'retrieval':retrieval_ms,'decision':(perf_counter()-before)*1000,
                             'verification':0,'total':(perf_counter()-started)*1000}}
