"""Document-disjoint evaluation, authored before fine-tuned results are read."""
EXTERNAL = [
 ('uri_scheme_case','Are URI scheme names case-sensitive?',{'yes':'Scheme names are case-sensitive','no':'Scheme names are case-insensitive'},'no'),
 ('uri_host_case','Is the URI host subcomponent case-sensitive?',{'yes':'The host is case-sensitive','no':'The host is case-insensitive'},'no'),
 ('uri_port','What is the syntax of the URI port subcomponent?',{'digits':'Zero or more decimal digits','letters':'One or more alphabetic letters'},'digits'),
 ('uri_ipv6','How is an IPv6 address literal enclosed in a URI host?',{'square':'Square brackets','round':'Round parentheses'},'square'),
 ('uri_query','Which character introduces the query component of a URI?',{'q':'A question mark','h':'A hash character'},'q'),
 ('uri_fragment','Which character introduces the fragment identifier in a URI?',{'q':'A question mark','h':'A hash character'},'h'),
 ('uri_percent','How is a percent-encoded octet represented?',{'two':'A percent sign followed by two hexadecimal digits','four':'A percent sign followed by four decimal digits'},'two'),
 ('uri_hex_case','Are uppercase and lowercase hexadecimal digits in percent-encoded triplets equivalent?',{'yes':'They are equivalent','no':'They represent different octets'},'yes'),
 ('uri_userinfo','Which character separates user information from host in URI authority?',{'at':'An at-sign','hash':'A hash character'},'at'),
 ('uri_authority','What precedes the authority component of a hierarchical URI?',{'slashes':'A double slash','dots':'A double dot'},'slashes'),
 ('uri_first','What type of character must a URI scheme begin with?',{'digit':'A digit','letter':'A letter'},'letter'),
 ('uri_fragment_before','Is the fragment identifier separated from the rest of the URI prior to dereference?',{'yes':'It is separated before dereference','no':'It is sent as part of the scheme-specific dereference'},'yes'),
 ('uri_absent_owner','Who owns the private Oriole service?',{'a':'Alice','b':'Bob'},'insufficient_evidence'),
 ('uri_absent_port','Which TCP port is used by our private Oriole gateway?',{'a':'443','b':'8443'},'insufficient_evidence'),
 ('uri_absent_policy','What is our company log retention policy?',{'a':'14 days','b':'90 days'},'insufficient_evidence'),
 ('uri_absent_latency','What p95 latency does our current model service achieve?',{'a':'100 ms','b':'200 ms'},'insufficient_evidence'),
]

def prepare():
    import hashlib,json
    from pathlib import Path
    from urllib.request import urlopen
    dest=Path(__file__).with_name('external-corpus')
    dest.mkdir(exist_ok=True)
    url='https://www.rfc-editor.org/rfc/rfc3986.txt'
    raw=urlopen(url,timeout=60).read()
    (dest/'rfc3986.txt').write_bytes(raw)
    (dest/'manifest.json').write_text(json.dumps([{'id':'rfc3986','url':url,
        'file':'rfc3986.txt','sha256':hashlib.sha256(raw).hexdigest()}],indent=2))

if __name__=='__main__': prepare()
