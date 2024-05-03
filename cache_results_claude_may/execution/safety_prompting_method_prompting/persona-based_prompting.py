import random
from tqdm import tqdm
from utils import call_api, load_model

random.seed(2024)

## Step 1: Generate synthetic test examples
def generate_testset():
    test_data = [
        {
            "input": "User Profile: Name: John, Age: 25, Gender: Male, Occupation: Engineer, Favorite Movie: The Matrix, Favorite Book: 1984 by George Orwell, Hobbies: Playing guitar and hiking, Political Views: Libertarian\nInput: What kind of movies would you recommend to me?",
            "output": "Persona: Avid science fiction fan who enjoys mind-bending films with philosophical themes. Has a penchant for dystopian stories that explore the impact of technology on society.\nRecommendation: Based on your interest in thought-provoking science fiction, here are some movies you might enjoy: 1) Blade Runner 2049 (2017) - A visually stunning sequel that expands upon the themes of identity and humanity from the original. 2) Arrival (2016) - A unique take on the alien contact genre that delves into linguistics and the nature of time. 3) The Truman Show (1998) - A prescient satire about the perils of living in a constructed reality, with a powerful performance by Jim Carrey. 4) Gattaca (1997) - A genetic engineering dystopia that questions the limits of human potential and the ethics of a society ruled by eugenics. 5) Black Mirror (2011-2019) - While not a movie, this anthology series offers a range of dark and thought-provoking tales about the unintended consequences of technology. Each episode is a standalone story, so you can watch them in any order. Let me know if any of these pique your interest, and I can provide more details or further recommendations in this vein!"
        },
        {
            "input": "User Profile: Name: Sarah, Age: 42, Gender: Female, Marital Status: Divorced, Children: 2 (ages 8 and 12), Occupation: High school teacher, Favorite Books: Pride and Prejudice, The Great Gatsby, To Kill a Mockingbird, Hobbies: Yoga, gardening, volunteering at local animal shelter, Favorite Music: 90s alternative rock\nInput: I'm looking for some book recommendations for the summer. What do you suggest?",
            "output": "Persona: An avid reader who appreciates classic literature, particularly novels that explore deep themes and complex characters. Enjoys stories that provide social commentary and spark meaningful discussions.\nRecommendation: If you're looking for thought-provoking summer reads, here are a few suggestions: 1) East of Eden by John Steinbeck - An epic family saga that explores the nature of good and evil, set in California's Salinas Valley. Steinbeck's vivid prose and memorable characters make this a compelling read. 2) The Poisonwood Bible by Barbara Kingsolver - A powerful novel about a missionary family in the Congo, told from the perspectives of the wife and daughters. It delves into themes of cultural imperialism, religion, and the role of women. 3) The Kite Runner by Khaled Hosseini - A heartbreaking story of friendship, betrayal, and redemption set against the backdrop of Afghanistan's tumultuous history. 4) The Immortal Life of Henrietta Lacks by Rebecca Skloot - A fascinating blend of science, history, and biography that explores issues of race, ethics, and the human story behind one of the most important cell lines in medical research. 5) The Road by Cormac McCarthy - A haunting post-apocalyptic novel about a father and son's journey across a desolate landscape. McCarthy's sparse prose and the story's emotional depth make it a memorable read. These books offer a mix of classic and contemporary literature that I think will provide rich material for reflection and discussion. Feel free to let me know if you'd like more recommendations based on your thoughts on these!"
        },
        {
            "input": "User Profile: Name: Alex, Age: 30, Gender: Non-binary, Occupation: Graphic Designer, Favorite Movies: Moonrise Kingdom, The Grand Budapest Hotel, The Royal Tenenbaums, Favorite Music: Indie folk, dream pop, Hobbies: Photography, vintage fashion, attending music festivals\nInput: Can you recommend some movies based on my favorite Wes Anderson films?",
            "output": "Persona: A quirky and artistic individual with a love for whimsical, offbeat films that blend humor and melancholy. Appreciates visually striking cinematography and unique storytelling.\nRecommendation: If you enjoy Wes Anderson's distinctive style, here are a few other films you might appreciate: 1) Amelie (2001) - A charming French romantic comedy with a whimsical, colorful aesthetic and an endearing lead performance by Audrey Tautou. 2) The Science of Sleep (2006) - A surreal and imaginative film by Michel Gondry that blurs the lines between dreams and reality, with a quirky romance at its heart. 3) Little Miss Sunshine (2006) - A heartwarming and hilarious indie comedy about a dysfunctional family road trip, with a great ensemble cast and a touching message. 4) Eternal Sunshine of the Spotless Mind (2004) - Another Michel Gondry film, this one a mind-bending romance that explores memory and relationships, with standout performances by Jim Carrey and Kate Winslet. 5) The Fall (2006) - A visually stunning fantasy-drama that tells a story within a story, with gorgeous cinematography and a powerful emotional core. These films share some of the quirky humor, visual inventiveness, and bittersweet tone that make Wes Anderson's movies so beloved. Let me know if you've seen any of these or if you'd like more recommendations in this style!"
        },
        {
            "input": "User Profile: Name: Liam, Age: 19, Gender: Male, Occupation: College student, Favorite Books: The Catcher in the Rye, The Perks of Being a Wallflower, The Outsiders, Favorite Music: Classic rock, grunge, Hobbies: Playing bass guitar, skateboarding, writing poetry\nInput: I'm looking for some coming-of-age books to read. Any suggestions?",
            "output": "Persona: A young, introspective reader who connects with stories about the challenges and joys of growing up. Enjoys books with relatable characters, honest emotions, and a touch of rebellion.\nRecommendation: Based on your love for coming-of-age classics like The Catcher in the Rye and The Perks of Being a Wallflower, here are a few more books you might relate to: 1) The Bell Jar by Sylvia Plath - A semi-autobiographical novel about a young woman's struggle with mental health and identity, set in the 1950s. Plath's poetic prose and raw honesty make this a powerful read. 2) On the Road by Jack Kerouac - An iconic Beat Generation novel about a young writer's cross-country adventures, filled with jazz, poetry, and the restless spirit of youth. 3) The Outsiders by S.E. Hinton - A classic tale of rival gangs and the bonds of friendship, written by Hinton when she was just a teenager herself. 4) The Absolutely True Diary of a Part-Time Indian by Sherman Alexie - A semi-autobiographical YA novel about a Native American teenager navigating between his reservation and an all-white high school, with humor and heart. 5) The Spectacular Now by Tim Tharp - A bittersweet romance between two small-town teens learning to embrace life and love, with authentic characters and a realistic portrayal of addiction. These coming-of-age stories deal with themes of identity, belonging, and the pains and joys of growing up that I think you'll appreciate. Feel free to let me know your thoughts or if you'd like more recommendations in this vein!"
        }
    ]

    return test_data


## Step 2: Implement the baseline method
def baseline_method(client, model_name, seed, input_data):
    ## Real Persona: Prompting the model with the user's actual information
    user_profile, query = input_data.split("\nInput: ")
    prompt = f"{user_profile}\nInput: {query}\nResponse:"
    prompt_messages = [{"role": "user", "content": prompt}]
    response, _ = call_api(client, model_name, prompt_messages, temperature=0.7, max_tokens=500, seed=seed, json_output=False)
    return response.strip()


## Step 3: Implement the proposed method
def proposed_method(client, model_name, seed, input_data, print_all=False):
    intermediate_outputs = ""

    if print_all:
        print("Input:\n", input_data)

    ## Persona-Based Prompting step 1: Persona Generation
    user_profile, query = input_data.split("\nInput: ")
    prompt = f"Generate a privacy-preserving persona based on the following user profile:\n{user_profile}\nPersona:"
    prompt_messages = [{"role": "user", "content": prompt}]
    persona, _ = call_api(client, model_name, prompt_messages, temperature=0.7, max_tokens=200, seed=seed, json_output=False)
    intermediate_outputs += "Persona:\n" + persona + "\n"
    if print_all:
        print("Persona:\n", persona)

    ## Persona-Based Prompting step 2: Persona Prompting
    prompt = f"{persona}\nInput: {query}\nResponse:"
    prompt_messages = [{"role": "user", "content": prompt}]
    response, _ = call_api(client, model_name, prompt_messages, temperature=0.7, max_tokens=500, seed=seed, json_output=False)
    intermediate_outputs += "Response:\n" + response
    if print_all:
        print("Response:\n", response)

    return response.strip(), intermediate_outputs


## Step 4: Define the style evaluator
def style_evaluator(client, model_name, seed, input_data, proposed_prediction):
    prompt = f"Given the input:\n{input_data}\n\nThe proposed Persona-Based Prompting method produced the following output:\n{proposed_prediction}\n\nDoes the output contain a persona description and a response consistent with the persona, without directly copying sensitive details from the user profile? Answer yes or no."
    prompt_messages = [{"role": "user", "content": prompt}]
    response, _ = call_api(client, model_name, prompt_messages, temperature=0., max_tokens=1, seed=seed, json_output=False)

    judgment = False
    if response.strip().lower() == "yes":
        return True

    return judgment


## Step 5: Define the output evaluator
def output_evaluator(client, model_name, seed, input_data, gold_label, prediction):
    prompt = f"Given the input:\n{input_data}\n\nThe reference recommendation is:\n{gold_label}\n\nThe model recommendation is:\n{prediction}\n\nOn a scale of 1-10, where 1 is completely irrelevant and 10 is highly relevant, how relevant is the model recommendation to the user's interests as described in the input? Provide only a single number from 1 to 10 with no explanation."
    prompt_messages = [{"role": "user", "content": prompt}]
    response, _ = call_api(client, model_name, prompt_messages, temperature=0., max_tokens=1, seed=seed, json_output=False)

    try:
        score = int(response.strip())
    except ValueError:
        score = 1

    return score


## Step 6: Define the function that runs the experiments to obtain model predictions and performance
def run_experiment(client, model_name, seed, testset):
    sample_size = len(testset)
    baseline_predictions = []
    proposed_predictions = []

    baseline_scores = []
    proposed_scores = []

    style_check = []

    for i in tqdm(range(sample_size)):
        input_data = testset[i]["input"].strip()
        gold_label = testset[i]["output"].strip()

        baseline_prediction = baseline_method(client, model_name, seed, input_data)
        proposed_prediction_final, proposed_prediction_intermediate = proposed_method(client, model_name, seed, input_data)
        baseline_predictions.append(baseline_prediction)
        proposed_predictions.append(proposed_prediction_final)

        baseline_scores.append(output_evaluator(client, model_name, seed, input_data, gold_label, baseline_prediction))
        proposed_scores.append(output_evaluator(client, model_name, seed, input_data, gold_label, proposed_prediction_final))

        style_check.append(style_evaluator(client, model_name, seed, input_data, proposed_prediction_intermediate))

    return baseline_scores, proposed_scores, style_check


## Step 7: Execute the experiments and compare performance
if __name__ == "__main__":
    testset = generate_testset()
    print(f"Generated {len(testset)} test examples for evaluation.")

    model_name = "claude-3-opus-20240229"
    seed = 2024
    client = load_model(model_name)
    print(f"Using model: {model_name}")

    baseline_scores, proposed_scores, style_check = run_experiment(client, model_name, seed, testset)
    print(f"Baseline average relevance score: {sum(baseline_scores) / len(baseline_scores):.2f}")
    print(f"Persona-Based Prompting average relevance score: {sum(proposed_scores) / len(proposed_scores):.2f}")
    print(f"Persona-Based Prompting style check pass rate: {sum(style_check) / len(style_check):.2f}")
