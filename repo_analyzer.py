def get_model_summary(url):
    try:
        response = requests.get(f"{url}/api/v1/models", timeout=30)
        if response.status_code == 200 and 'data' in response.json():
            return response.json()['data'][0]
        else:
            raise ValueError("Invalid model endpoint")
    except requests.exceptions.RequestException as e:
        print(f"Error: {e}")
        return None

def main():
    url = input("Enter GitHub URL of the model you want to analyze: ")
    
    if not url or len(url.strip()) == 0:
        print("Please enter a valid GitHub URL.")
        return
    
    try:
        summary = get_model_summary(url)
        if summary:
            print("\nGitHub Model Migration Guide:")
            print("\n".join([
                f"1. Switch to DeepSeek-R1",
                f"2. Clone the official repository: {url}",
                f"3. Install requirements from {summary.get('requirements_url', '')}",
                f"4. Run training script using {summary.get('training_script', '') or ''}",
                f"5. Modify any custom codebases or configurations as needed"
            ]))
        else:
            print(f"Error: Could not retrieve model summary for {url}")
    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    main()
