import random
import time
from eden.client import Client
from eden.datatypes import Image


## set up client
c = Client(url="http://0.0.0.0:80", username="abraham")

def test_client():

    config = {
        "image_url": "https://generations.krea.ai/images/6fd9ab51-7825-4936-836a-7733983fa307.webp",
        "top_k": 3,
        "mode": "img2img2txt"
    }
    print(config)

    # start the task
    response = c.run(config)
    print(response)

    # check status of the task, returns the output too if the task is complete
    results = c.fetch(token=response["token"])
    print(results)

    while True:
        results = c.fetch(token=response["token"])
        if results["status"]["status"] == "complete":
            print(results["output"])
            break
        time.sleep(0.1)


if __name__ == "__main__":
    test_client()