import requests

class MLAPIMiddleware:
    def __init__(self, api_url):
        self.api_url = api_url

    def process_request(self, income, duration):
        # Prepare the user information
        user_info = {
            'income': income,
            'duration': duration
        }

        # Make a POST request to the machine learning API
        try:
            response = requests.post(self.api_url, json=user_info)
            response.raise_for_status()

            # Process the API response
            processed_data = response.json()

            # Perform further actions or return the processed data
            return processed_data

        except requests.exceptions.RequestException as e:
            # Handle any request exceptions or errors
            print(f"Error processing the request: {e}")
            return None
