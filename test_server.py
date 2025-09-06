import requests
import json

def test_server():
    """Test if the Flask server is working properly"""
    base_url = "http://localhost:5000"
    
    print("Testing Flask server...")
    
    try:
        # Test basic endpoint
        response = requests.get(f"{base_url}/")
        print(f"✓ Main page: Status {response.status_code}")
        
        # Test text analysis
        test_text = "This is a simple test text to check if the analysis endpoints are working properly."
        
        print("\nTesting text analysis endpoint...")
        response = requests.post(f"{base_url}/check", 
                               json={"text": test_text},
                               headers={"Content-Type": "application/json"})
        
        if response.status_code == 200:
            result = response.json()
            print(f"✓ Text analysis: Status {response.status_code}")
            print(f"  Perplexity: {result.get('perplexity', 'N/A')}")
            print(f"  Burstiness: {result.get('burstiness', 'N/A')}")
        else:
            print(f"✗ Text analysis failed: Status {response.status_code}")
            print(f"  Error: {response.text}")
        
        print("\nTesting streamlit analysis endpoint...")
        response = requests.post(f"{base_url}/streamlit-analysis", 
                               json={"text": test_text},
                               headers={"Content-Type": "application/json"})
        
        if response.status_code == 200:
            result = response.json()
            print(f"✓ Streamlit analysis: Status {response.status_code}")
            print(f"  Word frequency data: {len(result.get('word_frequency', {}).get('words', []))} words")
        else:
            print(f"✗ Streamlit analysis failed: Status {response.status_code}")
            print(f"  Error: {response.text}")
            
        print("\n" + "="*50)
        print("Server test completed!")
        
    except requests.exceptions.ConnectionError:
        print("✗ Could not connect to server. Make sure it's running on localhost:5000")
    except Exception as e:
        print(f"✗ Test failed with error: {e}")

if __name__ == "__main__":
    test_server()
