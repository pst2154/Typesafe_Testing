import unittest
from pathlib import Path
from retrieval import Corpus
from engine import Grounded

class Fake:
    name='fake'
    def fits(self,state,questions): return True
    def predict(self,state,questions):
        key=next(iter(questions))
        return {'answers':{key:{'choice':'yes' if key=='decision' else 'unknown'}}}

class Tests(unittest.TestCase):
    def setUp(self):
        self.corpus=Corpus(Path(__file__).with_name('corpus'))
    def test_provenance(self):
        results=self.corpus.search('JSON UTF-8 encoding')
        self.assertTrue(results)
        for r in results:
            filename=r['id'].split(':')[0]+'.txt'
            lines=(Path(__file__).with_name('corpus')/filename).read_text().splitlines(keepends=True)
            self.assertEqual(r['text'],''.join(lines[r['line_start']-1:r['line_end']]))
    def test_failed_verification_abstains(self):
        r=Grounded(self.corpus,Fake()).decide('JSON encoding',{'yes':'UTF-8','no':'UTF-16'})
        self.assertTrue(r['abstained'])
        self.assertEqual(r['citations'],[])
    def test_empty_query_rejected(self):
        with self.assertRaises(ValueError):
            Grounded(self.corpus,Fake()).decide('',{'yes':'yes','no':'no'})

if __name__=='__main__': unittest.main()
