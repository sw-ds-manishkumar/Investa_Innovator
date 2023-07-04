from flask import Flask, request
from ml_api_middleware import MLAPIMiddleware

app = Flask(__name__)

# Initialize the ML API middleware
api_middleware = MLAPIMiddleware('http://192.168.4.126:8001/predict')  # Replace with the appropriate API URL

@app.route('/process_user_info', methods=['POST'])
def process_user_info():
    # Get the user information from the request
    user_info = request.get_json()

    # Extract the username and age from user_info
    income = user_info.get('income')
    duration = user_info.get('duration')

    # Process the user information using the ML API middleware
    processed_data = api_middleware.process_request(income, duration)

    if processed_data is not None:
        # Perform further actions with the processed data
        return processed_data
    else:
        return "Error processing user information."

if __name__ == '__main__':
    app.run(host="192.168.4.126", port=8002, debug=False)
