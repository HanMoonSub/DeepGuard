from fastapi.responses import StreamingResponse

class PNGStreamingResponse(StreamingResponse):
    media_type = "image/png"