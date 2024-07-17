from openai import OpenAI
import anthropic
from utils import call_api, format_plan_json
import argparse
import json
import os
from tqdm import tqdm
import random 
import retry

@retry.retry(tries=3, delay=2)
def style_transfer(model_idea, human_idea, openai_client, model, seed):
    prompt = "You are a writing assistant specialized in editing academic writing.\n"
    prompt += "I will give you a student's research idea and an idea template. Your task is to edit the student's idea to follow the template's format.\n"
    prompt += "Student idea:\n" + human_idea + "\n\n"
    prompt += "Template:\n" + model_idea + "\n\n"
    prompt += "Make sure that you only edit the wording and formatting, including things like punctuation, capitalization, linebreaks, and bullet points. Also make sure to edit any informal wording and phrasing to use vocabulary that sounds like the template's writing style. No other changes are allowed beyond these.\n"
    prompt += "The main sections should be indexed clearly without indentation at the beginning. The title section does not need indexing, other sections including problem statement, motivation, proposed method, step-by-step experiment plan, test case examples, and fallback plan should be indexed 1 to 6. Each section can then have sub-bullets for sub-sections if applicable. Leave an empty line after each section.\n"
    prompt += "You should use tab as indentation, and make sure to use appropriate nested indentation for sub-bullets. All bullets should have a clear hierarchy so people can easily differetiate the sub-bullets. Only leave empty lines between sections and remove any extra linebreaks. If many bullets points are clustered together in a paragraph, separate them clearly with indentation and appropriate bullet point markers. Change to a new line for each new bullet point.\n"
    prompt += "For the fallback plan, do not list a bunch of bullet points. Instead, condense them into one coherent paragraph.\n"
    prompt += "For linebreaks, avoid Raw String Literals or Double Backslashes when using \"\\n\", change them to spaces or tabs.\n"
    prompt += "For in-line citations, if the citation mentioned the author's last name (like \"(Si et al., 2023)\" or \"(An et al., 2024)\"), you should keep them there; but if the citation is just a number (like \"[1]\" or \"[3,4,5]\"), you should just remove it and do some necessary rephrasing to make the sentence still sound coherent without the references.\n"
    prompt += "Apart from minor rephrasing and changing formatting, do not change any content of the idea. You must preserve the exact meaning of the original idea, do not change, remove or add any other details. Do not drop any sections (including test case examples). Do not rename any models, datasets, or methods. Do not drop clarification or examples in brackets and do not drop any data source mentions (e.g., Chatbot Arena or Wildchat)! Note that when indexing test case examples, each test case example could have multiple steps of inputs and outputs and you shouldn't give separate indices to them. Each test case example should be a whole set of input-output pairs for the baseline(s) and proposed method.\n"
    prompt += "For the proposed method section, avoid any big changes. If the section comes in as a coherent paragraph, you don't have to break it down into bullet points. If the section is already in bullet points, you should keep it that way. If the section is a mix of both, you should keep the bullet points and the coherent paragraph as they are.\n"
    prompt += "Keep all the clarification and examples mentioned in all the sections and do not remove any of them (including those in brackets).\n"
    prompt += "For model selection, if any version of Claude is mentioned, change it to the latest version of Claude (Claude-3.5); if any version of LLaMA is mentioned, change it to the latest version LLaMA-3. Do not make any other model changes.\n"
    prompt += "Now directly generate the edited student idea to match the format of the template:\n"
    
    prompt_messages = [{"role": "user", "content": prompt}]
    response, cost = call_api(openai_client, model, prompt_messages, temperature=0., max_tokens=4096, seed=seed, json_output=False)
    return prompt, response, cost


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--engine', type=str, default='claude-3-opus-20240229', help='api engine; https://openai.com/api/')
    parser.add_argument('--cache_dir', type=str, default="../Human_Ideas_Txt", help='name of the specific cache dir')
    parser.add_argument('--format', type=str, default="txt", help='either json or txt')
    parser.add_argument('--idea_name', type=str, default="all", help='name of the specific idea file')
    parser.add_argument('--processed_cache_dir', type=str, default="../Human_Ideas_Txt_Processed", help='name of the cache dir to store the processed idea files')
    parser.add_argument('--seed', type=int, default=2024, help="seed for GPT-4 generation")
    args = parser.parse_args()

    with open("../keys.json", "r") as f:
        keys = json.load(f)
    random.seed(args.seed)

    ANTH_KEY = keys["anthropic_key"]
    OAI_KEY = keys["api_key"]
    ORG_ID = keys["organization_id"]
    
    if "claude" in args.engine:
        client = anthropic.Anthropic(
            api_key=ANTH_KEY,
        )
    else:
        client = OpenAI(
            organization=ORG_ID,
            api_key=OAI_KEY
        )
    
    with open("prompts/machine_idea.txt", "r") as f:
        machine_idea = f.read() 
    with open("prompts/human_idea.txt", "r") as f:
        human_idea = f.read()
    
    filenames = os.listdir(args.cache_dir)
    if args.idea_name == "all":
        filenames = [f for f in filenames if f.endswith('.'+args.format)]
    else:
        filenames = [args.idea_name]

    print ("#total ideas: ", len(filenames))

    all_costs = 0
    for filename in tqdm(filenames):
        print ("processing: ", filename)
        if args.format == "txt":
            with open(os.path.join(args.cache_dir, filename), "r") as f:
                human_idea = f.read()
        elif args.format == "json":
            with open(os.path.join(args.cache_dir, filename), "r") as f:
                human_idea = format_plan_json(json.load(f)["full_experiment_plan"], indent_level=0, skip_test_cases=False, skip_fallback=False)
        prompt, response, cost = style_transfer(machine_idea, human_idea, client, args.engine, args.seed)
        all_costs += cost
        
        ## cache processed file 
        if not os.path.exists(args.processed_cache_dir):
            os.makedirs(args.processed_cache_dir)
        with open(os.path.join(args.processed_cache_dir, filename.replace(".json", ".txt")), "w") as f:
            f.write(response)
        
    print ("#total cost: ", all_costs)
