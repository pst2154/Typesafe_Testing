"""Small in-memory BM25 index with immutable source provenance."""
import hashlib
import json
import math
import re
from collections import Counter
from pathlib import Path

def terms(text):
    return re.findall(r'[a-z0-9_]+', text.lower())

class Corpus:
    def __init__(self, directory):
        directory = Path(directory)
        self.chunks = []
        manifest = json.loads((directory/'manifest.json').read_text())
        for doc in manifest:
            raw = (directory/doc['file']).read_bytes()
            if hashlib.sha256(raw).hexdigest() != doc['sha256']:
                raise ValueError('Corpus checksum mismatch')
            # Preserve exact source spans, including their line numbers.
            lines = raw.decode().splitlines(keepends=True)
            start = 0
            while start < len(lines):
                end, words = start, 0
                # Prefer complete paragraphs; a hard word boundary can sever a
                # MUST NOT rule from its object or separating condition.
                while end < len(lines):
                    words += len(lines[end].split())
                    end += 1
                    if words >= 75 and (not lines[end-1].strip() or words >= 160):
                        break
                text = ''.join(lines[start:end])
                if len(terms(text)) > 15:
                    self.chunks.append({'id':f'{doc["id"]}:L{start+1}-{end}',
                        'url':doc['url'], 'sha256':doc['sha256'], 'text':text,
                        'line_start':start+1, 'line_end':end})
                start = end
        self.counts = [Counter(terms(c['text'])) for c in self.chunks]
        self.df = Counter(t for c in self.counts for t in c)
        self.lengths = [sum(c.values()) for c in self.counts]
        self.avg = sum(self.lengths)/max(1,len(self.lengths))

    def search(self, query, k=4):
        ts = set(terms(query))
        n = len(self.chunks)
        ranked = []
        for i, counts in enumerate(self.counts):
            score = 0
            for t in ts.intersection(counts):
                tf = counts[t]
                idf = math.log(1+(n-self.df[t]+.5)/(self.df[t]+.5))
                score += idf*tf*2.2/(tf+1.2*(.25+.75*self.lengths[i]/self.avg))
            if score > 0:
                ranked.append((score,i))
        return [dict(self.chunks[i], retrieval_score=score)
                for score,i in sorted(ranked, reverse=True)[:k]]
