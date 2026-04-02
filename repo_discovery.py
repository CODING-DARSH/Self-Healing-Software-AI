import requests
import json
from os import system

def get_top_repos(api_token, query, min_stars=1000, top_n=5):
    repos = []
    github_url = "https://api.github.com/search/repos"
    
    # Page through all results
    offset = 0
    while len(repos) < top_n:
        response = requests.get(
            f"{github_url}?q={query}&star_count>{min_stars}&page={offset}",
            headers={
                "Authorization": f"token {api_token}"
            }
        )
        if response.status_code == 200:
            data = response.json()
            total_repos = int(data.get("total search count", 0))
            
            # Limit results to top_n
            results = data.get("search", {}).get("items", [])
            chunk_size = len(results) // top_n
            
            for index, repo in enumerate(results):
                if repos and len(repos) >= top_n:
                    break
                
                name = repo["name"]
                description = repo.get("description", "")
                url = repo["url"]
                
                repos.append({
                    "name": name,
                    "url": url,
                    "description": description
                })
        else:
            print(f"Error: {response.status_code}")
            break
        
        offset += 1
    
    return repos

if __name__ == "__main__":
    from getpass import getpass
    api_token = input("Enter GitHub API token: ")
    
    # You can modify the query as needed (e.g., 'ml' for machine learning)
    search_query = "ml"
    
    top_repos = get_top_repos(api_token, search_query, 1000, 5)
    
    print("\nTop ML Repositories with >1000 Stars:")
    for repo in top_repos:
        print(f"\n{repo['name']}\nURL: {repo['url']}")
        if repo['description']:
            print(f"Description: {repo['description']}")
