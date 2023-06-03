import asyncio
import websockets
import base64
import io
from PIL.Image import Image
from websockets.server import serve
from jss import initialize

def image_to_byte_array(image: Image) -> bytes:
  # BytesIO is a file-like buffer stored in memory
  imgByteArr = io.BytesIO()
  # image.save expects a file-like as a argument
  image.save(imgByteArr, format=image.format)
  # Turn the BytesIO object back into a bytes object
  imgByteArr = imgByteArr.getvalue()
  return imgByteArr

async def file_read(sock):
    async for message in sock:
        # BRUH

        decoded_msg = base64.b64decode(message)
        jss_res = initialize(decoded_msg)

        # jss_res_bytes = image_to_byte_array(jss_res)
        # jss_res_bytes = base64.b64encode(jss_res)
        print(str(jss_res))
        await sock.send(jss_res)

async def main():
    async with serve(file_read, 'localhost', 1243):
        await asyncio.Future()

asyncio.run(main())