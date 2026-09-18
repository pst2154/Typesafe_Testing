"""Tokenizer regression checks; no GPU inference or network requests."""
import importlib.util
import os
from transformers import AutoTokenizer

spec=importlib.util.spec_from_file_location('structured','/opt/vllm/examples/features/diffusion_reads/structured_server.py')
m=importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
m.CANVAS_LEN=32
m.init_tokenizer(AutoTokenizer.from_pretrained(os.environ.get('MODEL_ID','nvidia/diffusiongemma-26B-A4B-it-NVFP4'),local_files_only=True))
names=['user identity switch','reverse shell','bind shell']+[f'question {i}' for i in range(303)]
for n in (1,10,11,306):
    schema=m.parse_schema({'questions':[{'id':name,'type':'noul','instructions':'Is the proposition true?'} for name in names[:n]]})
    for group in m.question_groups(schema):
        _,slots=m.template_for(dict(schema,questions=group),m.SCAFFOLD,'')
        assert len(slots)==len(group)
        assert all(len(set(s['label_ids']))==2 for s in slots)
    print(f'{n} question template slots passed')
