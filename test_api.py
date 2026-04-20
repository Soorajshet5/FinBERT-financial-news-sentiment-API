import urllib.request
import json

def test(text, ticker):
    data = json.dumps({"text": text, "ticker": ticker}).encode()
    req = urllib.request.Request(
        "http://127.0.0.1:8000/api/v1/analyze",
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST"
    )
    result = json.loads(urllib.request.urlopen(req).read())
    print(f"{ticker}: {result['label']} ({result['confidence']})")

test("Wipro wins 500 million dollar deal with European bank", "WIPRO")
test("Yes Bank faces RBI scrutiny over bad loans", "YESBANK")
test("Zomato reports first ever quarterly profit", "ZOMATO")
test("Adani Group shares crash after Hindenburg report", "ADANIENT")
test("TCS announces 18000 crore share buyback", "TCS")
test("Paytm loses payment aggregator license", "PAYTM")
test("Bajaj Finance raises interest rates on deposits", "BAJFINANCE")
test("Coal India declares highest ever dividend", "COALINDIA")