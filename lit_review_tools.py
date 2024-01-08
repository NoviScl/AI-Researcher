import requests
import re

# Define the paper search endpoint URL
search_url = 'https://api.semanticscholar.org/graph/v1/paper/search/'
graph_url = 'https://api.semanticscholar.org/graph/v1/paper/'
rec_url = "https://api.semanticscholar.org/recommendations/v1/papers/forpaper/"

def KeywordQuery(keyword):
    ## retrieve papers based on keywords
    query_params = {
        'query': keyword,
        'limit': 10,
        'fields': 'title,year,citationCount,abstract,tldr'
    }
    response = requests.get(search_url, params=query_params)
    
    if response.status_code == 200:
        return response.json()
    else:
        return None

def PaperQuery(paper_id):
    ## retrieve similar papers based on paper id
    query_params = {
        'paperId': paper_id,
        'limit': 50
    }
    response = requests.get(url = rec_url + paper_id, params = query_params)
    
    if response.status_code == 200:
        return response.json()
    else:
        return None

def PaperDetails(paper_id):
    ## get paper details based on paper id
    paper_data_query_params = {'fields': 'title,year,abstract,authors,citationCount,venue,citations,references'}
    response = requests.get(url = graph_url + paper_id, params = paper_data_query_params)

    if response.status_code == 200:
        return response.json()
    else:
        return None

def GetAbstract(paper_id):
    ## get the abstract of a paper based on paper id
    paper_details = PaperDetails(paper_id)
    
    if paper_details is not None:
        return paper_details["abstract"]
    else:
        return None

def GetCitationCount(paper_id):
    ## get the citation count of a paper based on paper id
    paper_details = PaperDetails(paper_id)
    
    if paper_details is not None:
        return int(paper_details["citationCount"])
    else:
        return None

def GetCitations(paper_id):
    ## get the citation list of a paper based on paper id
    paper_details = PaperDetails(paper_id)
    
    if paper_details is not None:
        return paper_details["citations"]
    else:
        return None

def GetReferences(paper_id):
    ## get the reference list of a paper based on paper id
    paper_details = PaperDetails(paper_id)
    
    if paper_details is not None:
        return paper_details["references"]
    else:
        return None

def paper_filter(response):
    ## filter out papers based on heuristics
    paper_lst = response["data"]
    filtered_lst = []
    for paper in paper_lst:
        abstract = paper["abstract"] if paper["abstract"] else paper["title"]
        if paper["year"] and int(paper["year"]) < 2022:
            continue 
        if paper["citationCount"] and int(paper["citationCount"]) <= 10:
            continue 
        if "survey" in abstract.lower() or "review" in abstract.lower() or "position paper" in abstract.lower():
            continue
        filtered_lst.append(paper)
    return filtered_lst

def parse_and_execute(output):
    ## parse gpt4 output and execute corresponding functions
    if output.startswith("KeywordQuery"):
        match = re.match(r'KeywordQuery\("([^"]+)"\)', output)
        keyword = match.group(1) if match else None
        if keyword:
            response = KeywordQuery(keyword)
            if response is not None:
                return paper_filter(response)
    elif output.startswith("PaperQuery"):
        match = re.match(r'PaperQuery\("([^"]+)"\)', output)
        paper_id = match.group(1) if match else None
        if paper_id:
            response = PaperQuery(paper_id)
            if response is not None:
                return paper_filter(response)
    elif output.startswith("GetAbstract"):
        match = re.match(r'GetAbstract\("([^"]+)"\)', output)
        paper_id = match.group(1) if match else None
        if paper_id:
            return GetAbstract(paper_id)
    elif output.startswith("GetCitationCount"):
        match = re.match(r'GetCitationCount\("([^"]+)"\)', output)
        paper_id = match.group(1) if match else None
        if paper_id:
            return GetCitationCount(paper_id)
    elif output.startswith("GetCitations"):
        match = re.match(r'GetCitations\("([^"]+)"\)', output)
        paper_id = match.group(1) if match else None
        if paper_id:
            return GetCitations(paper_id)
    
    return None 

def format_papers_for_printing(paper_lst):
    ## convert a list of papers to a string for printing or as part of a prompt 
    output_str = ""
    for paper in paper_lst:
        output_str += "paperId: " + paper["paperId"].strip() + "\n"
        output_str += "title: " + paper["title"].strip() + "\n"
        if paper["tldr"] and paper["tldr"]["text"]:
            output_str += "tldr: " + paper["tldr"]["text"].strip() + "\n"
        output_str += "\n"

    return output_str

if __name__ == "__main__":
    ## some unit tests
    # print (KeywordQuery("GPT-3"))
    # print (PaperDetails("1b6e810ce0afd0dd093f789d2b2742d047e316d5"))
    # print (PaperQuery("1b6e810ce0afd0dd093f789d2b2742d047e316d5"))
    # print (GetAbstract("1b6e810ce0afd0dd093f789d2b2742d047e316d5"))
    # print (GetCitationCount("1b6e810ce0afd0dd093f789d2b2742d047e316d5"))
    # print (GetCitations("1b6e810ce0afd0dd093f789d2b2742d047e316d5"))
    # print (GetReferences("1b6e810ce0afd0dd093f789d2b2742d047e316d5"))
    print (format_papers_for_printing(parse_and_execute("KeywordQuery(\"Prompting Strategies for Large Language Models\")")))