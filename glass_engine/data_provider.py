import reactivex as rx
from reactivex.subject import Subject
import websocket
import json
from reactivex import operators as ops
from constants import FINNHUB_API_KEY
import logging

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

API_KEY = FINNHUB_API_KEY
TICKER = "BINANCE:BTCUSDT"



class WsDataProvider:
    def __init__(self,url, cb_on_message):
        self._ws = None
        self.url = url
        self.cb_on_message = cb_on_message
        

    def subscribe(self):
        self.ws.run_forever()

    @property
    def ws(self):
        if not self._ws:
            self._ws = websocket.WebSocketApp(
                self.url,
                on_message=self.on_message,
                on_open=self.on_open,
                on_close=self.on_close
            )
        return self._ws

    def prepare_message_fn(self, msg):
        return msg

    def on_message(self, ws, message):
        prep_msg = self.prepare_message_fn(message)
        return self.cb_on_message(prep_msg)
        
    def on_open(self, ws):
        logger.info("Connection opened")

    def on_close(self, ws, close_status_code, close_msg):
        logger.info(f"Connection closed: {close_status_code} - {close_msg}")


class FinnhubDataProvider(WsDataProvider):

    _URL = f"wss://ws.finnhub.io?token={API_KEY}"

    def __init__(self, ticker, cb_on_message):
        super().__init__(self._URL, cb_on_message)
        self.ticker = ticker

    def prepare_message_fn(self, msg):
        data = json.loads(msg)
        return (data["data"][0]["t"], data["data"][0]["p"])

    def on_open(self, ws):
        sub_msg = json.dumps({"type": "subscribe", "symbol": self.ticker})
        self.ws.send(sub_msg)
        self.ws.run_forever()





def print_message(message):
    print(f"Received message: {message}")

if __name__ == "__main__":
 

    # # Create a Subject to act as an observable stream
    message_stream = Subject()


    data_provider = FinnhubDataProvider(TICKER, message_stream.on_next)

    import datetime
    def format_timestamp_now():
        return datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f')
    
    # # Example RxPY pipeline: filter and map messages
    message_stream.pipe(
        ops.debounce(0.3),  # 10ms debounce
        # ops.filter(lambda msg: "data" in msg),
        ops.map(lambda msg: f'[{format_timestamp_now()}] :{msg}')
    ).subscribe(
        lambda x: print(f"{format_timestamp_now()} Received:", x),
        lambda e: print("Error:", e),
        lambda: print("Stream completed")
    )
    data_provider.subscribe()
