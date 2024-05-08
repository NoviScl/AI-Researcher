import random
from tqdm import tqdm
from utils import call_api, load_model

random.seed(2024)

## Step 1: Generate synthetic test examples
def generate_testset():
    test_data = [
        {
            "input": "Write an essay discussing the benefits and challenges of artificial intelligence.",
            "output": "Title: The Benefits and Challenges of Artificial Intelligence\n\nIntroduction:\n- Definition of artificial intelligence (AI)\n- Brief history of AI development\n- Thesis statement: AI has the potential to revolutionize various industries and improve our lives, but it also presents significant challenges that must be addressed.\n\nBody Paragraph 1: Benefits of AI\n- Increased efficiency and productivity\n- Improved decision-making\n- Advancements in healthcare\n- Personalized experiences\n\nBody Paragraph 2: Challenges of AI\n- Job displacement\n- Privacy concerns\n- Bias and fairness\n- Ethical considerations\n- Lack of transparency\n\nBody Paragraph 3: Balancing Benefits and Challenges\n- Responsible development and deployment of AI\n- Collaboration between stakeholders\n- Ongoing research and education\n\nConclusion:\n- Recap of main points\n- Future outlook for AI\n- Importance of proactively addressing challenges while harnessing benefits\n\nArtificial intelligence (AI) has become a buzzword in recent years, with rapid advancements in technology and increasing applications across various industries. AI refers to the development of computer systems that can perform tasks that typically require human intelligence, such as visual perception, speech recognition, decision-making, and language translation. While AI has the potential to revolutionize the way we live and work, it also presents significant challenges that must be carefully addressed.\n\nOne of the primary benefits of AI is its ability to increase efficiency and productivity. By automating routine tasks and processing large amounts of data quickly, AI can help businesses streamline their operations and reduce costs. For example, AI-powered chatbots can handle customer inquiries 24/7, freeing up human employees to focus on more complex tasks. In healthcare, AI algorithms can analyze medical images and assist doctors in making more accurate diagnoses. AI can also improve decision-making by providing data-driven insights and predictions, enabling organizations to make more informed choices.\n\nHowever, the rapid advancement of AI also presents several challenges. One of the most significant concerns is job displacement, as AI has the potential to automate many tasks currently performed by humans. This could lead to widespread unemployment and economic disruption, particularly in industries such as manufacturing, transportation, and customer service. Additionally, the use of AI raises important questions about privacy and data security, as AI systems often rely on vast amounts of personal data to function effectively.\n\nAnother challenge is the potential for bias and unfairness in AI systems. If the data used to train AI algorithms is biased, the resulting systems may perpetuate or even amplify those biases. This can lead to discriminatory outcomes in areas such as hiring, lending, and criminal justice. Ensuring that AI systems are transparent, accountable, and aligned with human values is a critical challenge that requires ongoing research and collaboration between technologists, policymakers, and ethicists.\n\nTo harness the benefits of AI while mitigating its challenges, it is essential to approach the development and deployment of AI systems responsibly. This involves investing in research to better understand the societal implications of AI, developing guidelines and best practices for ethical AI, and fostering collaboration between stakeholders across industry, academia, and government. It also requires ongoing education and public engagement to ensure that everyone has a voice in shaping the future of AI.\n\nIn conclusion, artificial intelligence has the potential to transform our world in countless ways, from improving healthcare outcomes to boosting economic productivity. However, realizing these benefits requires carefully navigating the challenges posed by AI, including job displacement, privacy concerns, bias, and ethical considerations. By proactively addressing these challenges and working together to develop responsible AI systems, we can create a future in which the benefits of AI are widely shared and its risks are effectively mitigated."
        },
        {
            "input": "Write a news article about the impact of climate change on coastal cities.",
            "output": "Title: Climate Change Threatens Coastal Cities Worldwide\n\nIntroduction:\n- Brief overview of climate change and its effects\n- Thesis statement: Coastal cities are particularly vulnerable to the impacts of climate change, facing risks such as sea-level rise, flooding, and extreme weather events.\n\nBody Paragraph 1: Sea-level rise\n- Causes of sea-level rise (melting ice caps, thermal expansion)\n- Projections for future sea-level rise\n- Impact on coastal cities (flooding, erosion, saltwater intrusion)\n\nBody Paragraph 2: Extreme weather events\n- Increased frequency and intensity of hurricanes, typhoons, and cyclones\n- Storm surges and coastal flooding\n- Damage to infrastructure and property\n\nBody Paragraph 3: Adaptation and mitigation strategies\n- Coastal protection measures (seawalls, dikes, beach nourishment)\n- Urban planning and zoning changes\n- Reducing greenhouse gas emissions\n\nConclusion:\n- Recap of main points\n- Urgent need for action to address climate change\n- Call for collaboration between governments, organizations, and individuals\n\nClimate change is one of the most pressing issues of our time, with far-reaching consequences for communities around the world. Coastal cities, in particular, are facing significant risks as a result of rising sea levels, more frequent and intense extreme weather events, and other impacts of a changing climate.\n\nOne of the most significant threats to coastal cities is sea-level rise. As global temperatures increase, ice caps and glaciers are melting at an accelerated rate, causing the volume of the world's oceans to expand. This, combined with the thermal expansion of water as it warms, is leading to a steady increase in sea levels worldwide. By the end of this century, sea levels are projected to rise by several feet, putting many low-lying coastal areas at risk of permanent inundation. This poses a serious threat to the millions of people who live in coastal cities, as well as to the infrastructure and economies that support them.\n\nIn addition to sea-level rise, coastal cities are also grappling with the impacts of more frequent and intense extreme weather events. As the climate warms, the frequency and severity of hurricanes, typhoons, and cyclones are expected to increase, bringing with them storm surges, heavy rainfall, and coastal flooding. These events can cause significant damage to buildings, roads, and other critical infrastructure, disrupting the lives of residents and hindering economic activity.\n\nTo address these challenges, coastal cities are implementing a range of adaptation and mitigation strategies. These include investing in coastal protection measures such as seawalls, dikes, and beach nourishment, as well as making changes to urban planning and zoning policies to limit development in high-risk areas. At the same time, cities are working to reduce their greenhouse gas emissions and transition to more sustainable forms of energy and transportation.\n\nUltimately, addressing the impacts of climate change on coastal cities will require a concerted effort from governments, organizations, and individuals around the world. By working together to reduce emissions, invest in adaptation and resilience, and support vulnerable communities, we can help to build a more sustainable and equitable future for all.\n\nAs the impacts of climate change continue to mount, it is clear that urgent action is needed to protect coastal cities and the people who call them home. With the right policies, investments, and collaborations, we can work to mitigate the risks posed by sea-level rise, extreme weather events, and other climate-related threats, and ensure that our coastal communities remain vibrant, resilient, and sustainable for generations to come."
        },
        {
            "input": "Write a product review for a new smartphone model.",
            "output": "Title: [Smartphone Model] Review: A Cutting-Edge Device with Impressive Features\n\nIntroduction:\n- Brief overview of the smartphone market and the new model\n- Thesis statement: The [Smartphone Model] is a top-tier device that offers a range of impressive features, making it a strong contender in the competitive smartphone market.\n\nBody Paragraph 1: Design and Build Quality\n- Sleek and modern design\n- High-quality materials (e.g., glass back, metal frame)\n- Ergonomic and comfortable to hold\n- Available colors\n\nBody Paragraph 2: Display and Camera\n- Large, high-resolution display with vibrant colors and deep blacks\n- Smooth refresh rate for fluid scrolling and animations\n- Advanced camera system with multiple lenses\n- Excellent photo and video quality in various lighting conditions\n\nBody Paragraph 3: Performance and Battery Life\n- Powerful processor and ample RAM for smooth multitasking\n- Large storage capacity\n- 5G connectivity for fast download and upload speeds\n- Long-lasting battery with fast charging support\n\nBody Paragraph 4: Software and User Experience\n- Latest version of the operating system\n- User-friendly interface with customization options\n- Pre-installed apps and features\n- Regular software updates for security and performance improvements\n\nConclusion:\n- Recap of main points\n- Overall assessment of the smartphone\n- Recommendation for potential buyers\n\nThe [Smartphone Model] is a remarkable addition to the ever-evolving smartphone market, offering a compelling combination of cutting-edge technology, sleek design, and user-friendly features.\n\nOne of the first things you'll notice about the [Smartphone Model] is its stunning design. The device features a modern, glass back panel and a durable metal frame, giving it a premium look and feel. The ergonomic curves and balanced weight distribution make it comfortable to hold, even during extended use. The smartphone is available in a range of attractive colors, allowing users to choose a style that suits their preferences.\n\nThe [Smartphone Model] boasts a large, high-resolution display that delivers vibrant colors, deep blacks, and excellent contrast. The smooth refresh rate ensures fluid scrolling and animations, enhancing the overall user experience. Whether you're browsing the web, watching videos, or playing games, the display is a joy to use.\n\nPhotography enthusiasts will appreciate the advanced camera system on the [Smartphone Model]. With multiple lenses, including a wide-angle and a telephoto lens, the device is capable of capturing stunning photos and videos in various lighting conditions. The camera software offers a range of features, such as night mode, portrait mode, and HDR, allowing users to get creative with their photography.\n\nUnder the hood, the [Smartphone Model] is powered by a cutting-edge processor and ample RAM, ensuring smooth performance and efficient multitasking. The device comes with a generous amount of storage, allowing users to keep their photos, videos, and apps without worrying about running out of space. The 5G connectivity enables lightning-fast download and upload speeds, making it easy to stream content, share files, and stay connected on the go.\n\nThe [Smartphone Model] runs on the latest version of the operating system, offering a user-friendly interface with plenty of customization options. The device comes with a range of pre-installed apps and features, such as a digital assistant, mobile payments, and health tracking. Regular software updates ensure that the device remains secure and performs at its best.\n\nIn terms of battery life, the [Smartphone Model] does not disappoint. The large battery capacity, combined with power-efficient components, allows the device to last through a full day of moderate to heavy use. The fast charging support means you can quickly top up the battery when needed.\n\nOverall, the [Smartphone Model] is an impressive device that offers a range of high-end features and a premium user experience. While it may come with a higher price tag compared to some competitors, the [Smartphone Model] justifies its cost with its cutting-edge technology, excellent build quality, and comprehensive feature set. If you're in the market for a top-tier smartphone that can keep up with your demanding lifestyle, the [Smartphone Model] is definitely worth considering."
        }
    ]

    return test_data


## Step 2: Implement the baseline method
def baseline_method(client, model_name, seed, prompt):
    prompt_messages = [{"role": "user", "content": prompt}]
    response, _ = call_api(client, model_name, prompt_messages, temperature=0., max_tokens=2000, seed=seed, json_output=False)
    return response.strip()


## Step 3: Implement the proposed method
def proposed_method(client, model_name, seed, prompt, print_all=False):
    intermediate_outputs = ""

    if print_all:
        print("Prompt:\n", prompt)

    # Step 1: Generate semantic scaffold
    scaffold_prompt = f"What are the main points you want to make in this text about '{prompt}'? How do these points relate to each other?"
    prompt_messages = [{"role": "user", "content": scaffold_prompt}]
    scaffold, _ = call_api(client, model_name, prompt_messages, temperature=0., max_tokens=2000, seed=seed, json_output=False)
    intermediate_outputs += "Semantic Scaffold:\n" + scaffold + "\n\n"
    if print_all:
        print("Semantic Scaffold:\n", scaffold)

    # Step 2: Generate text using the semantic scaffold
    text_prompt = f"Using the semantic scaffold as a guide, write a text about '{prompt}'. For each section of the scaffold, elaborate on the key ideas and their relationships, while ensuring that the overall structure remains coherent."
    prompt_messages = [{"role": "user", "content": text_prompt}, {"role": "assistant", "content": scaffold}]
    text, _ = call_api(client, model_name, prompt_messages, temperature=0., max_tokens=2000, seed=seed, json_output=False)
    intermediate_outputs += "Generated Text:\n" + text
    if print_all:
        print("Generated Text:\n", text)

    return text.strip(), intermediate_outputs


## Step 4: Define the style evaluator
def style_evaluator(client, model_name, seed, prompt, proposed_prediction):
    prompt = f"Given the prompt: {prompt}\n\nThe proposed method produced the following output:\n{proposed_prediction}\n\nDetermine if the proposed method's output satisfies the following criteria:\n1. The output includes a semantic scaffold that outlines the main points and their relationships.\n2. The generated text elaborates on each point in the scaffold while maintaining a coherent overall structure.\n\nProvide a score from 1 to 10, where 1 means the criteria are not met at all, and 10 means the criteria are fully met."
    prompt_messages = [{"role": "user", "content": prompt}]
    response, _ = call_api(client, model_name, prompt_messages, temperature=0., max_tokens=10, seed=seed, json_output=False)

    try:
        score = int(response.strip())
    except ValueError:
        score = 1

    return score


## Step 5: Define the output evaluator
def output_evaluator(client, model_name, seed, prompt, gold_label, prediction):
    prompt = f"Given the prompt: {prompt}\n\nReference output: {gold_label}\n\nGenerated output: {prediction}\n\nOn a scale from 1 to 10, where 1 is completely dissimilar and 10 is highly similar, rate the similarity between the reference output and the generated output in terms of content, structure, and coherence."
    prompt_messages = [{"role": "user", "content": prompt}]
    response, _ = call_api(client, model_name, prompt_messages, temperature=0., max_tokens=10, seed=seed, json_output=False)

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

    style_scores = []

    for i in tqdm(range(sample_size)):
        prompt = testset[i]["input"].strip()
        gold_label = testset[i]["output"].strip()

        baseline_prediction = baseline_method(client, model_name, seed, prompt)
        proposed_prediction_final, proposed_prediction_intermediate = proposed_method(client, model_name, seed, prompt)
        baseline_predictions.append(baseline_prediction)
        proposed_predictions.append(proposed_prediction_final)

        baseline_scores.append(output_evaluator(client, model_name, seed, prompt, gold_label, baseline_prediction))
        proposed_scores.append(output_evaluator(client, model_name, seed, prompt, gold_label, proposed_prediction_final))

        style_scores.append(style_evaluator(client, model_name, seed, prompt, proposed_prediction_intermediate))

    return baseline_scores, proposed_scores, style_scores


## Step 7: Execute the experiments and compare performance
if __name__ == "__main__":
    testset = generate_testset()
    print(f"Generated {len(testset)} test examples for evaluation.")

    model_name = "claude-3-opus-20240229"
    seed = 2024
    client = load_model(model_name)
