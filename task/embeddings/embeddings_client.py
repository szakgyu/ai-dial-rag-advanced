import json

import requests

DIAL_EMBEDDINGS = 'https://ai-proxy.lab.epam.com/openai/deployments/{model}/embeddings'

class DialEmbeddingsClient:
    def __init__(self, deployment_name: str, api_key: str):
        self.deployment_name = deployment_name
        self.api_key = api_key

    def get_embeddings(self, input_list: list[str], dimensions: int) -> dict[int, list[float]]:
        url = DIAL_EMBEDDINGS.format(model=self.deployment_name)
        headers = {
            'Content-Type': 'application/json',
            'api-key': self.api_key
        }
        payload = {
            'input': input_list,
            'dimensions': dimensions
        }
        response = requests.post(url, headers=headers, data=json.dumps(payload))
        response.raise_for_status()
        data = response.json()
        embeddings_dict = {i: item['embedding'] for i, item in enumerate(data['data'])}
        return embeddings_dict
