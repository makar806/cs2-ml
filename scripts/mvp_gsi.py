"""Optional localhost GSI kill counter. No screen capture or input injection.

Only the local player's increasing kill count is used. It confirms a kill,
not a victim identity; controlled single-target bot tests are the intended use.
"""
import hmac
from http.server import BaseHTTPRequestHandler,ThreadingHTTPServer
import json
import threading
import time


class KillCounter:
    def __init__(self,clock=time.perf_counter):
        self.clock=clock;self.lock=threading.Lock();self.identity=None;self.kills=None
        self.sequence=0;self.last_update=None;self.last_kill=None

    def update(self,payload):
        if not isinstance(payload,dict):return False
        provider=payload.get('provider',{});player=payload.get('player',{});game_map=payload.get('map',{})
        if not all(isinstance(x,dict) for x in (provider,player,game_map)):return False
        owner=provider.get('steamid');observed=player.get('steamid')
        if provider.get('appid')!=730 or not isinstance(owner,str) or not owner or owner!=observed:return False
        if player.get('activity')!='playing':return False
        stats=player.get('match_stats',{})
        if not isinstance(stats,dict):return False
        kills=stats.get('kills')
        if type(kills) is not int or kills<0:return False
        identity=(owner,game_map.get('name'))
        with self.lock:
            now=self.clock();self.last_update=now
            # New match, initial payload and a reset/decrease establish baseline.
            increased=identity==self.identity and self.kills is not None and kills>self.kills
            self.identity=identity;self.kills=kills
            if increased:self.sequence+=1;self.last_kill=now
        return True

    def snapshot(self):
        with self.lock:
            age=self.clock()-self.last_update if self.last_update is not None else None
            return dict(sequence=self.sequence,kills=self.kills,age=age,last_kill=self.last_kill)


class KillReceiver:
    def __init__(self,token,port=8765):
        self.counter=KillCounter();counter=self.counter
        if not isinstance(token,str) or len(token)<24:raise ValueError('GSI token must have at least 24 characters')
        class Handler(BaseHTTPRequestHandler):
            def log_message(self,*args):pass # never log auth or player identity
            def reply(self,code):
                self.send_response(code);self.send_header('Content-Length','0');self.end_headers()
            def do_POST(self):
                self.connection.settimeout(1)
                try:
                    size=int(self.headers.get('Content-Length','0'))
                    if not 0<size<=65536:self.reply(413);return
                    payload=json.loads(self.rfile.read(size))
                    auth=payload.get('auth',{}) if isinstance(payload,dict) else {}
                    given=auth.get('token') if isinstance(auth,dict) else None
                    if not isinstance(given,str) or not hmac.compare_digest(given,token):self.reply(403);return
                    counter.update(payload);self.reply(200)
                except (ValueError,TypeError,TimeoutError,OSError):
                    try:self.reply(400)
                    except OSError:pass
            def do_GET(self):self.reply(404)
        self.server=ThreadingHTTPServer(('127.0.0.1',port),Handler)
        self.server.daemon_threads=True
        self.thread=threading.Thread(target=self.server.serve_forever,kwargs={'poll_interval':.1},daemon=True)
        self.thread.start()

    def snapshot(self):return self.counter.snapshot()

    def close(self):
        self.server.shutdown();self.server.server_close();self.thread.join(timeout=1)


def config_text(token,port=8765):
    if not token.isalnum() or len(token)<24:raise ValueError('Use an alphanumeric local token')
    if type(port) is not int or not 1024<=port<=65535:raise ValueError('Invalid local port')
    return f'''"CS2 ML local bot MVP"
{{
    "uri" "http://127.0.0.1:{port}/"
    "timeout" "0.5"
    "buffer" "0.0"
    "throttle" "0.1"
    "heartbeat" "1.0"
    "auth" {{ "token" "{token}" }}
    "data"
    {{
        "provider" "1"
        "map" "1"
        "player_id" "1"
        "player_match_stats" "1"
    }}
}}
'''
