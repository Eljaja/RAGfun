# Honestly I keep this file cause we can have an alternative
# (for example only for put operations)
# Better have a little tech ZOO in the circus
#
#

from fastapi import FastAPI, Request, HTTPException
import json


app = FastAPI()


@app.post("/webhook")
async def main(request: Request):
    try:
        request_data = await request.json()
    except Exception as e:
        print(f"[WEBHOOK] Error processing webhook: {e}")
        raise HTTPException(
            status_code=500, detail=f"Failed to process webhook: {e}",)
    print(request.headers)
    # Log the incoming webhook
    print(
        f"[WEBHOOK] Received notification: {json.dumps(request_data, indent=2)}")

    return {
        "status": "received",
        "processed": True,
        "suspicious": "NO_WAY_TO_FIND_OUT",
        "reasons": "NOT_IMPLEMENTED",
    }
