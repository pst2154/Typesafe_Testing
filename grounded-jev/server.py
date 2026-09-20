"""Local-first RAG decision service. Set an API key before exposing it remotely."""
import argparse
import json
import os
import secrets
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from backends import LayaBackend, DiffusionBackend
from engine import Grounded
from retrieval import Corpus

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--backend',choices=['laya','diffusiongemma'],default='laya')
    parser.add_argument('--host',default='127.0.0.1')
    parser.add_argument('--port',type=int,default=8790)
    parser.add_argument('--variant',choices=['baseline','plain','entailment','expanded','diverse'],default='diverse')
    args=parser.parse_args()
    key=os.environ.get('GROUNDED_API_KEY')
    if args.host not in ('127.0.0.1','localhost','::1') and not key:
        raise ValueError('GROUNDED_API_KEY is required for non-loopback binding')
    backend=LayaBackend() if args.backend=='laya' else DiffusionBackend()
    app=Grounded(Corpus(Path(__file__).with_name('corpus')),backend)
    if args.variant!='baseline':
        from variants import OnePass
        app=OnePass(app.corpus,backend,args.variant)
    gate=threading.BoundedSemaphore(1)
    class Handler(BaseHTTPRequestHandler):
        def send(self,status,value):
            data=json.dumps(value).encode()
            self.send_response(status)
            self.send_header('Content-Type','application/json')
            self.send_header('Content-Length',str(len(data)))
            self.end_headers()
            self.wfile.write(data)
        def do_GET(self):
            if self.path=='/health':
                self.send(200,{'ready':True,'model':backend.name,'variant':args.variant,'chunks':len(app.corpus.chunks)})
            else: self.send(404,{'error':'not found'})
        def do_POST(self):
            if self.path!='/v1/grounded/decision':
                return self.send(404,{'error':'not found'})
            if key and not secrets.compare_digest(self.headers.get('Authorization',''),'Bearer '+key):
                return self.send(401,{'error':'unauthorized'})
            if not gate.acquire(blocking=False):
                return self.send(429,{'error':'busy; retry later'})
            try:
                length=int(self.headers.get('Content-Length','0'))
                if not 0<length<=16384: raise ValueError('invalid body size')
                self.connection.settimeout(10)
                body=json.loads(self.rfile.read(length))
                self.send(200,app.decide(body['query'],body['criteria']))
            except (ValueError,KeyError,TypeError): self.send(400,{'error':'invalid request or token budget exceeded'})
            except Exception: self.send(502,{'error':'decision backend unavailable'})
            finally: gate.release()
    print(f'Ready: {args.host}:{args.port} ({backend.name})',flush=True)
    ThreadingHTTPServer((args.host,args.port),Handler).serve_forever()

if __name__=='__main__': main()
