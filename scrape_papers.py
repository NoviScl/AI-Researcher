import numpy as np
import pandas as pd
from tqdm import tqdm 
import requests
import os
from pypdf import PdfReader
from io import BytesIO
import openreview 
import json 

def get_paper_ids():
    ## return a list of paper ids for rejected and accepted papers
    papers_accepted = []
    papers_rejected = []

    for year in [2024]:
        # print(year, end=': ')
        # for query in ['submission', 'Submission', 'Blind_Submission', 'Rejected_Submission', '']:
        for query in ['Rejected_Submission', '']:
            
            if year <= 2017:
                if query == '':
                    continue
                url = f'https://api.openreview.net/notes?invitation=ICLR.cc%2F{year}%2Fconference%2F-%2F{query}'
            elif year <= 2023:
                if query == '':
                    continue
                url = f'https://api.openreview.net/notes?invitation=ICLR.cc%2F{year}%2FConference%2F-%2F{query}'
            else:
                if query != '':
                    query = '/' + query
                url = f'https://api2.openreview.net/notes?content.venueid=ICLR.cc/{year}/Conference{query}'        
            
            print ("URL: ", url)
            for offset in range(0, 10000, 1000):
                ## the request is paginated, so we need to loop through the pages
                df = pd.DataFrame(requests.get(url + f'&offset={offset}').json()['notes'])
                for index, row in df.iterrows():
                    paper_dict = {}
                    
                    # print (index, row)
                    forum_id = row['forum']
                    content = row['content']
                    title = content['title']['value'].strip()
                    abstract = content['abstract']['value'].strip()
                    primary_area = content["primary_area"]["value"].strip()
                    keywords = content["keywords"]["value"]

                    paper_dict["forum_id"] = forum_id
                    paper_dict["title"] = title
                    paper_dict["abstract"] = abstract
                    paper_dict["primary_area"] = primary_area
                    paper_dict["keywords"] = keywords

                    ## keyword filtering 
                    all_fields = title + " " + abstract + " " + primary_area + " " + " ".join(keywords)
                    if "language model" not in all_fields.lower():
                        continue 

                    # if index % 500 == 0:
                    #     print (paper_dict)
                                        
                    if 'Rejected_Submission' in query:
                        papers_rejected.append(paper_dict)
                    else:
                        papers_accepted.append(paper_dict)
            
        return papers_accepted, papers_rejected


def get_reviews(client, forum_id):
    decision = None 
    meta_review = None 
    scores = [] 
    reviews = []
    all_replies = []

    # Fetch all replies (reviews, comments, etc.)
    all_reviews = client.get_notes(forum=forum_id)

    for review in all_reviews:
        if not isinstance(review, dict):
            review = review.to_json()
        
        content = review["content"]
        # print (content)
        # print ("\n\n")
        
        if "decision" in content:
            decision = content["decision"]["value"].strip()
        if "metareview" in content:
            meta_review = content["metareview"]["value"].strip() + "\n"
            meta_review += "Justification for why not higher score: " + content["justification_for_why_not_higher_score"]["value"].strip() + "\n"
            meta_review += "Justification for why not lower score: " + content["justification_for_why_not_lower_score"]["value"].strip()
        if "summary" in content:
            scores.append(content["rating"]["value"])
            reviews.append(content)
        all_replies.append(review)
    
    return decision, meta_review, scores, reviews, all_replies
    

def download_and_extract_text_from_pdf(pdf_url):
    # Step 1: Download the PDF
    response = requests.get(pdf_url)
    if response.status_code != 200:
        raise Exception(f"Failed to download PDF. Status code: {response.status_code}")
    
    # Step 2: Extract text from the PDF
    pdf_file = BytesIO(response.content)
    pdf_reader = PdfReader(pdf_file)
    text = ""
    
    for page_num in range(len(pdf_reader.pages)):
        page = pdf_reader.pages[page_num]
        text += page.extract_text()

    # Step 3: Return the extracted text
    return text.strip()


if __name__ == "__main__":
    with open("keys.json") as f:
        keys = json.load(f)

    client = openreview.api.OpenReviewClient(
        baseurl='https://api2.openreview.net',
        username=keys["openreview_name"],
        password=keys["openreview_password"]
    )

    papers_accepted, papers_rejected = get_paper_ids()
    print (f"Number of accepted papers: {len(papers_accepted)}")
    print (f"Number of rejected papers: {len(papers_rejected)}")

    for idx, paper in tqdm(enumerate(papers_accepted + papers_rejected)):
        forum_id = paper["forum_id"]
        decision, meta_review, scores, reviews, all_replies = get_reviews(client, forum_id)
        full_text = download_and_extract_text_from_pdf(f"https://openreview.net/pdf?id={forum_id}")
        if decision and meta_review and len( scores) > 1 and len(reviews) > 1 and len(all_replies) > 1 and full_text and len(full_text) > 500:
            paper["decision"] = decision
            paper["meta_review"] = meta_review
            paper["scores"] = scores
            paper["reviews"] = reviews
            paper["all_replies"] = all_replies
            paper["full_text"] = full_text
            with open(f"openreview_benchmark/paper_{idx}.json", "w") as f:
                json.dump(paper, f, indent=4)


