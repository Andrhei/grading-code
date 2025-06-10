import requests

class RestApiClient:
    def __init__(self, base_url, auth_token=None):
        self.base_url = base_url
        self.auth_token = auth_token

    def _get_headers(self, headers):
        default_headers = headers or {}
        if self.auth_token:
            default_headers['Authorization'] = f"Bearer {self.auth_token}"
        return default_headers

    def get(self, endpoint, params=None, headers=None):
        url = f"{self.base_url}/{endpoint}"
        final_headers = self._get_headers(headers)
        response = requests.get(url, params=params, headers=final_headers)
        return self._handle_response(response)

    def post(self, endpoint, data=None, json_data=None, headers=None):
        url = f"{self.base_url}/{endpoint}"
        final_headers = self._get_headers(headers)
        response = requests.post(url, data=data, json=json_data, headers=final_headers)
        return self._handle_response(response)

    def _handle_response(self, response):
        try:
            response.raise_for_status()
            return response.json()
        except requests.exceptions.HTTPError as http_err:
            print(f"HTTP error occurred: {http_err}")
        except ValueError:
            print("Response content is not valid JSON")
        except Exception as err:
            print(f"An error occurred: {err}")
        return None