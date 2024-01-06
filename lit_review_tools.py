import requests

# Define the paper search endpoint URL
search_url = 'https://api.semanticscholar.org/graph/v1/paper/search'
graph_url = 'https://api.semanticscholar.org/graph/v1/paper'

## retrieve papers based on keywords
def KeywordQuery(keyword):
    query_params = {
        'query': keyword,
        'limit': 50
    }
    response = requests.get(search_url, params=query_params)
    
    if response.status_code == 200:
        return response.json()
    else:
        return None

## get paper details based on paper id
def PaperDetails(paper_id):
    paper_data_query_params = {'fields': 'title,year,abstract,authors,citationCount,venue,citations,references'}
    response = requests.get(url = graph_url + '/' + paper_id, params = paper_data_query_params)

    if response.status_code == 200:
        return response.json()
    else:
        return None

if __name__ == "__main__":
    ## some unit tests
    # print (KeywordQuery("GPT-3"))
    print (PaperDetails("1b6e810ce0afd0dd093f789d2b2742d047e316d5"))