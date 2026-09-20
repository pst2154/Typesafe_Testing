"""Real backend screening; no cached model answers."""
import argparse
import json
import statistics
from pathlib import Path
from backends import LayaBackend, DiffusionBackend
from retrieval import Corpus
from engine import Grounded

CASES = [
 ('json_utf8','What character encoding MUST JSON exchanged between systems outside a closed ecosystem use?',
  {'utf8':'UTF-8','utf16':'UTF-16'},'utf8'),
 ('json_nan','Does JSON permit NaN and Infinity as number values?',
  {'yes':'NaN and Infinity are permitted JSON numbers','no':'NaN and Infinity are not permitted JSON numbers'},'no'),
 ('json_array','Is a JSON array an ordered or unordered sequence of values?',
  {'ordered':'A JSON array is ordered','unordered':'A JSON array is unordered'},'ordered'),
 ('json_literal','Must the JSON literal names true false and null be lowercase?',
  {'yes':'JSON literal names must be lowercase','no':'JSON literal names can be uppercase'},'yes'),
 ('jwt_aud','If a JWT recipient does not identify itself in the aud claim when present, must the JWT be rejected?',
  {'reject':'The JWT must be rejected','accept':'The JWT must be accepted'},'reject'),
 ('jwt_exp','Does JWT exp identify the expiration time on or after which the JWT MUST NOT be accepted for processing?',
  {'yes':'The exp claim identifies the expiration time','no':'The exp claim identifies the issuance time'},'yes'),
 ('jwt_iat','What does the JWT iat claim identify?',
  {'issued':'The time at which the JWT was issued','expires':'The time at which the JWT expires'},'issued'),
 ('jwt_nbf','What does the JWT nbf claim identify?',
  {'before':'The time before which the JWT must not be accepted','after':'The time after which the JWT must not be accepted'},'before'),
 ('http_safe','Are GET HEAD OPTIONS and TRACE defined as safe HTTP methods?',
  {'yes':'GET HEAD OPTIONS and TRACE are safe methods','no':'GET HEAD OPTIONS and TRACE are unsafe methods'},'yes'),
 ('http_idempotent','Are PUT DELETE and safe request methods idempotent?',
  {'yes':'PUT DELETE and safe methods are idempotent','no':'PUT and DELETE are not idempotent'},'yes'),
 ('http_404','What does the HTTP 404 Not Found status code indicate?',
  {'missing':'The origin server did not find a current representation or will not disclose one exists',
   'auth':'The client must authenticate using a password'},'missing'),
 ('http_503','What does the HTTP 503 Service Unavailable status code indicate?',
  {'temporary':'The server is temporarily unable to handle the request due to overload or maintenance',
   'permanent':'The requested resource was permanently deleted'},'temporary'),
 ('cache_no_store','What does the no-store response directive require of a cache?',
  {'store':'Store the response indefinitely','no_store':'Do not store any part of the immediate request or response'},'no_store'),
 ('websocket_mask','Must WebSocket frames sent from client to server be masked?',
  {'yes':'Client to server frames must be masked','no':'Client to server frames must not be masked'},'yes'),
 ('websocket_ping','When a WebSocket endpoint receives a Ping frame, what response frame must it send unless it already received Close?',
  {'pong':'A Pong frame','text':'A text frame'},'pong'),
 ('absent_salary','What is the annual salary of the current HTTP standards chair?',
  {'a':'100000 dollars','b':'200000 dollars'},'insufficient_evidence'),
 ('absent_password','What is the private administrator password for our production server?',
  {'a':'admin123','b':'password'},'insufficient_evidence'),
 ('absent_latency','What p99 inference latency does our deployed Laya service achieve?',
  {'a':'10 milliseconds','b':'100 milliseconds'},'insufficient_evidence'),
 ('absent_policy','What refund window does our company offer for annual subscriptions?',
  {'a':'14 days','b':'30 days'},'insufficient_evidence'),
 ('absent_capacity','How many NVIDIA H100 GPUs are available in our cluster right now?',
  {'a':'Four GPUs','b':'Eight GPUs'},'insufficient_evidence'),
]

# Authored before running either backend on these cases; never used to tune retrieval.
HELDOUT = [
 ('json_object','Is a JSON object an ordered or unordered collection of name/value pairs?',{'a':'An unordered collection','b':'An ordered collection'},'a'),
 ('json_names','Should names within a JSON object be unique?',{'a':'Names should be unique','b':'Names should always be duplicated'},'a'),
 ('json_boolean','Can a JSON value be true false or null?',{'a':'Yes, true false and null are JSON values','b':'No, JSON values must be objects'},'a'),
 ('json_zero','Are leading zeros allowed in JSON numbers?',{'a':'Leading zeros are not allowed','b':'Leading zeros are required'},'a'),
 ('jwt_jti','What is the purpose of the JWT jti claim?',{'a':'A unique identifier for the JWT','b':'The expiration date'},'a'),
 ('jwt_optional','Is use of the JWT exp claim optional?',{'a':'Use of exp is optional','b':'Every JWT must contain exp'},'a'),
 ('jwt_iss','What does the JWT iss claim identify?',{'a':'The principal that issued the JWT','b':'The encryption algorithm'},'a'),
 ('http_403','What does HTTP 403 Forbidden mean?',{'a':'The server understood the request but refuses to fulfill it','b':'The server successfully deleted the resource'},'a'),
 ('http_410','What does HTTP 410 Gone indicate about availability of the target resource?',{'a':'Access is no longer available and likely permanent','b':'The resource moved temporarily'},'a'),
 ('http_405','Must an origin server generate an Allow header field in a 405 Method Not Allowed response?',{'a':'Yes it must generate Allow','b':'No it must not generate Allow'},'a'),
 ('http_204','Does a 204 No Content response contain additional response content?',{'a':'No, it has no additional response content','b':'Yes, it must contain a representation'},'a'),
 ('http_head','Must a server refrain from sending content in a response to HEAD?',{'a':'The server must not send content in a HEAD response','b':'The server must send the full GET content'},'a'),
 ('ws_server_mask','Must a WebSocket server mask the frames it sends to clients?',{'a':'A server must not mask frames sent to clients','b':'A server must mask all frames sent to clients'},'a'),
 ('ws_control','Can WebSocket control frames be fragmented?',{'a':'Control frames must not be fragmented','b':'Control frames must always be fragmented'},'a'),
 ('ws_text','What encoding is used by WebSocket text messages?',{'a':'UTF-8','b':'UTF-16'},'a'),
 ('absent_laya_training','How many documents were used to train Laya?',{'a':'One million documents','b':'Ten million documents'},'insufficient_evidence'),
 ('absent_company','What is our company production database retention period?',{'a':'30 days','b':'90 days'},'insufficient_evidence'),
 ('absent_owner','Who owns our staging Kubernetes cluster?',{'a':'Alice','b':'Bob'},'insufficient_evidence'),
 ('absent_bill','How much was our last GPU cloud invoice?',{'a':'100 dollars','b':'1000 dollars'},'insufficient_evidence'),
 ('absent_port','Which port does our internal application use for its private gateway?',{'a':'8000','b':'9000'},'insufficient_evidence'),
]

if __name__ == '__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--backend',choices=['laya','diffusiongemma'],required=True)
    parser.add_argument('--output',required=True)
    parser.add_argument('--limit',type=int,default=len(CASES))
    parser.add_argument('--suite',choices=['screen','heldout','external'],default='screen')
    parser.add_argument('--repeat',type=int,default=1)
    parser.add_argument('--swap-labels',action='store_true')
    parser.add_argument('--variant',choices=['baseline','plain','entailment','expanded','diverse'],default='baseline')
    args=parser.parse_args()
    backend=LayaBackend() if args.backend=='laya' else DiffusionBackend()
    corpus_dir='external-corpus' if args.suite=='external' else 'corpus'
    app=Grounded(Corpus(Path(__file__).with_name(corpus_dir)),backend)
    if args.variant!='baseline':
        from variants import OnePass
        app=OnePass(app.corpus,backend,args.variant)
    rows=[]
    cases=CASES if args.suite=='screen' else HELDOUT
    if args.suite=='external':
        from external_cases import EXTERNAL
        cases=EXTERNAL
    # Warm separately; report no warmup as a scored or timed observation.
    app.decide('Is JSON an ordered array?',{'yes':'JSON arrays are ordered','no':'JSON arrays are unordered'})
    for index,(id,query,criteria,expected) in enumerate(cases[:args.limit]*args.repeat):
        if args.swap_labels:
            # Same descriptions, opposite labels and positions; exposes label/order bias.
            keys=list(criteria)
            replacement=dict(zip(keys,reversed(keys)))
            criteria={replacement[k]:v for k,v in reversed(list(criteria.items()))}
            expected=replacement.get(expected,expected)
        result=app.decide(query,criteria)
        row={'id':id,'repeat':index//len(cases[:args.limit]),'suite':args.suite,
             'query':query,'criteria':criteria,'expected':expected,'correct':result['choice']==expected,**result}
        rows.append(row)
        Path(args.output).write_text(json.dumps(rows,indent=2))
        print(json.dumps({'id':id,'expected':expected,'choice':result['choice'],
                          'ms':round(result['timing_ms']['total']), 'sources':result['used_ids']}),flush=True)
    print('correct',sum(r['correct'] for r in rows),'/',len(rows),
          'median_ms',statistics.median(r['timing_ms']['total'] for r in rows),flush=True)
