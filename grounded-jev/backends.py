import json
import os
from pathlib import Path
from urllib.request import Request, urlopen

class ContextBudget:
    def __init__(self):
        from transformers import AutoTokenizer
        path=Path(Path(__file__).with_name('model-path.txt').read_text().strip())
        self.tok=AutoTokenizer.from_pretrained(path/'tokenizer')
        self.cfg=json.loads((path/'rl_agent_config.json').read_text())

    def fits(self,state,questions):
        from laya.common import build_sequence, serialize_state, render_options
        state_ids=self.tok(serialize_state(state),add_special_tokens=False)['input_ids']
        for question in questions.values():
            q={'t':question['type'],'ins':question['instructions'],'crit':question.get('criteria')}
            empty,markers=build_sequence(self.tok,'',q,self.cfg.get('max_len',512),self.cfg.get('head_max_len',192))
            if len(markers)!=len(render_options(q)) or len(empty)+len(state_ids)>self.cfg.get('max_len',512):
                return False
        return True

class LayaBackend:
    def __init__(self):
        os.environ.setdefault('USE_TF', '0')
        import laya
        self.agent = laya.load(Path(__file__).with_name('model-path.txt').read_text().strip(),
                               device=os.environ.get('LAYA_DEVICE','mps'))
        self.name = 'laya'

    def fits(self, state, questions):
        # SDK truncates silently: reject oversized state before calling it.
        from laya.common import build_sequence, serialize_state, render_options
        tok = self.agent.tok
        state_ids = tok(serialize_state(state), add_special_tokens=False)['input_ids']
        for question in questions.values():
            q = self.agent._to_internal(question)
            empty, markers = build_sequence(tok, '', q, self.agent.cfg.get('max_len',512),
                                            self.agent.cfg.get('head_max_len',192))
            if len(markers) != len(render_options(q)):
                return False
            if len(empty)+len(state_ids) > self.agent.cfg.get('max_len',512):
                return False
        return True

    def predict(self, state, questions):
        if not self.fits(state,questions):
            raise ValueError('Input exceeds model context; refusing silent truncation')
        return self.agent.predict(state,questions)

class DiffusionBackend:
    name = 'diffusiongemma'
    def __init__(self):
        # Match Laya's evidence budget even though this engine accepts more.
        self.budget=ContextBudget()
    def fits(self,state,questions):
        return self.budget.fits(state,questions)

    def predict(self,state,questions):
        body = {'model':'jev-latest','state':state,'questions':questions}
        headers = {'Content-Type':'application/json'}
        if os.environ.get('DIFFUSION_API_KEY'):
            headers['Authorization'] = 'Bearer '+os.environ['DIFFUSION_API_KEY']
        request = Request(os.environ['DIFFUSION_URL'], data=json.dumps(body).encode(),headers=headers)
        with urlopen(request,timeout=30) as response:
            return json.load(response)
