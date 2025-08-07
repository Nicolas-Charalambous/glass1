import websocket
from constants import FINNHUB_API_KEY
import json

API_KEY = FINNHUB_API_KEY
TICKER = 'BINANCE:BTCUSDT'

def on_message(ws, message):
    data = json.loads(message)
    print((data['data'][0]['t'], data['data'][0]['p']))

def on_open(ws):
    print("Connection opened")
    sub_msg = json.dumps({"type":"subscribe","symbol": TICKER})
    ws.send(sub_msg)

def on_close(ws, close_status_code, close_msg):
    print("Connection closed")

ws = websocket.WebSocketApp(
    f"wss://ws.finnhub.io?token={API_KEY}",
    on_message=on_message,
    on_open=on_open,
    on_close=on_close
)

ws.run_forever()