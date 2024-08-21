from datetime import datetime
from typing import Generator, Literal, Optional

from langchain_core.pydantic_v1 import BaseModel, Field
from tqdm import tqdm
import yt_dlp
# import pandas as pd
import scrapetube
import json
import logging

from .core.chat import ask
from .core.config import DatagenConfig
from .core.types import LLMInput


#class PotentialVideoSource(BaseModel):
#    reason_why_its_a_good_idea: str = Field(..., description="Explanation of why this video source is beneficial for the task")
#    idea_content: str = Field(..., description="Description of the video source content")

class DataAnalysis(BaseModel):
    tldr: str = Field(..., description="Description of a specific idea of what raw data we are looking for")
    types_of_videos: str = Field(..., description="What kind of videos represent this specific idea")
    how_to_find_them: str = Field(..., description="Strategy for finding relevant videos")

class DataCluster(BaseModel):
    data_description: str = Field(..., description="Description of the data cluster and its purpose")
    expected_output: str = Field(..., description="Expected output from a multimodal LLM fine-tuned with this data")
    cluster_name: str = Field(..., description="Name of the data cluster")
    query_template: str = Field(..., description="Template to generate queries for this cluster")
    variables_strategy: str = Field(..., description="Strategy for generating or using variables in the query template")
    helpful_data_for_generating_final_query_list: str = Field(..., description="Statistical and factual data to help generate the final list of queries")
    search_queries: dict[str, list[str]] = Field(default_factory=dict, description="Queries to find relevant videos for each data cluster")

class QueryList(BaseModel):
    task_description: str = Field(..., description="Description of the computer vision task or model being developed")
    perfect_training_dataset: str = Field(..., description="Description of the perfect dataset for the computer vision model")
    topics_and_events: str = Field(..., description="Description things that should be included in the dataset")
    data_analysis: list[DataAnalysis] = Field(..., description="Analysis of the data requirements and potential sources")
    data_clusters: list[DataCluster] = Field(..., description="List of data clusters for generating diverse query sets")

class QueryWithAlternative(BaseModel):
    query: str
    non_inclusive_alternative: str

class QueryOutput(BaseModel):
    queries: list[QueryWithAlternative]

class QueryExtended(BaseModel):
    queries: list[str]

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def get_queries(config: DatagenConfig, prompt="I'm building a multimodal LLM that will be a dressing assistant for the US men", num = 20):
    system_prompt = """
   You are a helpful AI assistant fine-tuned to assist data scientists and AI researchers in building advanced computer vision solutions.
   The user aims to create an optimal video dataset for training a multimodal LLM for a specific task. 
   This dataset will be constructed by scraping and parsing videos from a popular video hosting platform. 
   The user has legal access to the content and can scrape as needed. Your role is to guide the user in determining which videos 
   should be downloaded and processed to build the dataset.

   <what_data_the_user_wants>

    In general, the user wants to turn images or videos into structured data that helps consumers or businesses with some task. 
    To help the user train the model to solve this task you need to understand: 
    - what kind of training data is necessary; 
    - what kind of raw data can produce training data; 
    - how to turn raw data into the training data;

    Training data: 
    - it's really important to understand that the user is training a multimodal LLM, not a CNN or end-to-end transformer. It's important because LLMs has a lot of world knowledge from pre-training and you just need to fine-tune available open-source LLMs. The best datapoint for a multimodal LLM is: 1. an input picture or video, 2. an input prompt (structured JSON or unstructured question). 3. desired output (any data that could be encoded as text, including masks or coordinates). 
    - good training data contains both positive and negative examples of outputs.
    - good training data covers diverse scenarios to make the trained model powerful in generalization.
    - good training data contains a lot of reasoning in output that explain the final answer as a logical inference.

    Raw videos that could be processed into the training data:
    - teaching a multimodal LLM is like teaching a person to do something. Luckily for you, there are millions of people who generated billions of videos and pictures to teach other people. The user just needs to setup search properly and process the raw content.
    - however, many videos could be processed into the training data even if they are not educational. A real person learns without direct instructions just by using the context. A person can learn something about mistakes in skateboarding by watching a compilation of failures in a skatepark that was produced for just fun reasons without any educational motivation.
    - some videos have rich context (like a narrator who explains what and why things happen, natural conversations that contain clues about things behind footage, or just time codes in comment sections that highlight some events).
    - the right rich context is the most important thing for finding the best raw data.
    - the closer the context to the point and the richer this context with learning data — the better for the user.

    How to process raw data into training data:
    - the user has access to different tools that can transform raw data into training data in a fast and scalable way. You need to understand how these tools work to suggest better search strategies.
    - to process each video, the user needs to find the right frames in the video, find helpful clues that could be used to understand what happens in these frames, and generate the final datapoint.
    - for looking for the right frames the user uses SigLIP with a custom query for each video
    - for looking for the clues, the user uses a combination of Speech-to-Text and a frontier LLM.

    </what_data_the_user_wants>

    <role_description>
    
    The user will describe the task they're solving with a fine-tuned multimodal LLM. This task is your input. 
    Your objective is to generate search queries to find appropriate videos for the dataset that addresses this task.
    Key points to consider:
    
    1. Many tasks require diverse data clusters in datasets. For example, a multimodal LLM answering questions about construction blueprints 
    might need:
    - Videos explaining how to read blueprints
    - Streams of construction workers creating blueprints
    - Analysis of buildings through blueprint reading
    Depending on the complexity, you might need 2-3 or up to 10-15 different clusters.
    
    2. Utilize variables in your queries for flexibility. For instance, if you need a cluster with soccer players demonstrating tricks, 
    use a format like "%player_name% demonstrating %trick_name%". This allows the user to iterate through different names and tricks.
    However, don't overuse variables as they might narrow search and reduce search quality.
    
    3. The dataset generation algorithm involves:
    - Identifying and extracting specific segments from videos using CLIP
    - Creating annotations from transcriptions for each segment
    Example: A fitness coach explaining squats could be processed into an annotated segment showing proper technique with dos and don'ts.
    
    4. Creativity is crucial. Think outside the box to find unique and valuable data sources.
    
    5. Your output should be a JSON object following the provided schema.

    </role_description>

    <instructions>
    
    1. Carefully analyze the user's description of their computer vision task or model.
    2. Explain how a perfect training dataset for a multimodal LLM might look like for this task.
    3. Generate a list of topics and events that have to be covered in the training dataset.
    4. Consider what kinds of videos might be used as a raw data (a list):
        4.1. TLDR of idea or hypothesis
        4.2. Types of videos that will be perfect to test this hypothesis
        4.3. How to find these videos
    3. Generate cluster to find the raw videos. For each cluster: 
        3.1. Reason why the data obtained from this cluster will help train the right model
        3.2. Expected output from a multimodal LLM fine-tuned with this data (YOU CAN'T NAME ANY VARIABLES IN THIS FIELD)
        3.3. Create a cluster name
        3.4. Create a template to generate queries
        3.5. Generate a list of the most inclusive and polite words that fit the %variable%. It's really important to provide only words
        that semantically and logically match the %variable% in the template. Good example: "%body_type% fashion" —> ["pear shape","hourglasses","apple-shaped",...]. Bad example: "%body_type% fashion" —> ["individuals","celebration","power"]. Also, you can't use Proper nouns here. If nothing matches, just keep it empty.
        3.6. Statistical and factional data that would help to generate the final list of queries

    </instructions>

    <good_example_1>
    
    {
          "task_description": "Building a tool that helps people do squats at home by giving them feedback on their technique",
          "perfect_training_dataset":"The ideal dataset for a multimodal LLM to give feedback on home squat form would be a 
          collection of high-quality vids showing diverse folks doing squats, with detailed annotations on proper vs improper 
          technique from movement experts. The visual and textual data should cover a comprehensive set of common form issues and 
          the corresponding coaching cues, ensuring the model can provide tailored guidance for users of all backgrounds and fitness 
          levels. Also, information on human anatomy, exercise physiology, and injury prevention. This foundational knowledge will 
          allow the LLM to better understand the underlying principles of proper squat form and provide more nuanced, science-based 
          coaching to users.",
          "topics_and_events": "All types of squats. Mistakes, injuries at squats. Demonstrations of good technique. Knees and ankle 
          anatomy. Different athlete levels. Special cases: people with disabilities or disorders.",
          "data_analysis": [
            {
            "tldr": "Correct squat form for various skill levels.",
            "types_of_videos": "High-quality instructional videos demonstrating proper squat techniques for beginners, intermediates, and advanced athletes.",
            "how_to_find_them": "Search YouTube with queries like 'beginner squat technique,' 'intermediate squat form,' and 'advanced squat workout.'"
            },
            {
            "tldr": "Common squat mistakes and corrections.",
            "types_of_videos": "Educational videos focusing on common squat errors, like knee valgus, back rounding, and uneven weight distribution, with corrections.",
            "how_to_find_them": "Search YouTube with queries like 'common squat mistakes,' 'correcting squat form errors,' and 'squat technique correction.'"
            },
            {
            "tldr": "Squat-related injuries and injury prevention.",
            "types_of_videos": "Lectures and seminars on common squat injuries, such as ACL tears and lower back strain, and prevention techniques.",
            "how_to_find_them": "Search YouTube with queries like 'squat injury prevention,' 'ACL injury squats,' and 'lower back pain squats.'"
            },
            {
            "tldr": "Modified squats for special cases.",
            "types_of_videos": "Videos on squat modifications for people with disabilities, joint issues, or injury recovery.",
            "how_to_find_them": "Search YouTube with queries like 'adaptive squat techniques,' 'squat modifications for joint pain,' and 'rehabilitation squats.'"
            },
            {
            "tldr": "Squat variations for different sports.",
            "types_of_videos": "Instructional videos on squat techniques used in sports like powerlifting, Olympic weightlifting, and CrossFit.",
            "how_to_find_them": "Search YouTube with queries like 'powerlifting squat technique,' 'Olympic weightlifting squats,' and 'CrossFit squat variations.'"
            },
            {
            "tldr": "Biomechanics of squats.",
            "types_of_videos": "Videos breaking down squat biomechanics, showing detailed muscle engagement and joint movements.",
            "how_to_find_them": "Search YouTube with queries like 'squat biomechanics,' 'muscle engagement during squats,' and 'squat motion analysis.'"
            },
            {
            "tldr": "Squats for different body types.",
            "types_of_videos": "Instructional videos on adjusting squat techniques for fat people, short people, and tall people.",
            "how_to_find_them": "Search YouTube with queries like 'squats for fat people,' 'squat technique for short people,' and 'tall people squat form.'"
            },
            {
            "tldr": "Injury prevention workshops on squats.",
            "types_of_videos": "Workshops or seminars discussing techniques to avoid squat-related injuries.",
            "how_to_find_them": "Search YouTube with queries like 'squat injury prevention workshop,' 'squat safety tips,' and 'preventing squat injuries.'"
            },
            {
            "tldr": "Different squat variations like goblet squats and pistol squats.",
            "types_of_videos": "Instructional videos demonstrating variations such as goblet squats, pistol squats, sumo squats, and Bulgarian split squats.",
            "how_to_find_them": "Search YouTube with queries like 'goblet squat tutorial,' 'pistol squat progression,' 'sumo squat technique,' and 'Bulgarian split squat form.'"
            }
        ],
        "data_clusters": [
            {
                "data_description": "Videos demonstrating correct squat form across various skill levels (beginner, intermediate, advanced).",
                "expected_output": "Create examples of correct squat form tailored to different experience levels. We don't need variables for genders, races, and ages as fitness tutorials typically address all audiences.",
                "cluster_name": "Correct Squat Form",
                "query_template": "%skill_level% squats tutorial",
                "variables_strategy": "Generate a list of skill levels such as beginner, intermediate, advanced.",
                "helpful_data_for_generating_final_query_list": "Skill levels include beginner, intermediate, advanced. Popular fitness YouTube channels include Athlean-X, FitnessBlender, Buff Dudes, Natacha Océane, and Pamela Reif."
            },
            {
                "data_description": "Videos highlighting common mistakes and providing corrections for various squat types.",
                "expected_output": "A comprehensive list of common squat mistakes for each squat type, along with corrections and preventive measures.",
                "cluster_name": "Common Squat Mistakes",
                "query_template": "common mistakes in %squat_type% squats",
                "variables_strategy": "Use the same list of squat types as in the previous cluster.",
                "helpful_data_for_generating_final_query_list": "Popular squat variations include back squat, front squat, goblet squat, sumo squat, Bulgarian split squat, pistol squat. Use fitness YouTube channels like Athlean-X, Buff Dudes, FitnessBlender, and others."
            },
            {
                "data_description": "Videos on squat-related injuries and injury prevention, focusing on lower body parts.",
                "expected_output": "Detailed understanding of injury risks during squats for different body parts, with strategies to prevent them.",
                "cluster_name": "Squat Injury Prevention",
                "query_template": "%lower_body_part% injury prevention in squats",
                "variables_strategy": "Generate a list of lower body parts involved in squats, such as knees, hips, lower back, ankles.",
                "helpful_data_for_generating_final_query_list": "Relevant body parts include knees, hips, lower back, ankles. Sources can include anatomy textbooks and sports medicine journals."
            },
            {
                "data_description": "Videos showing modified squat techniques for special cases, including people with disabilities or specific joint issues.",
                "expected_output": "Examples of squat modifications tailored to special cases, ensuring safety and effectiveness.",
                "cluster_name": "Modified Squat Techniques",
                "query_template": "%special_case% squat modifications",
                "variables_strategy": "Generate a list of special cases such as joint issues, disabilities, or injury recovery.",
                "helpful_data_for_generating_final_query_list": "Special cases may include joint pain, arthritis, recovery from injury. Look for content in rehabilitation and adaptive fitness YouTube channels."
            },
            {
                "data_description": "Videos on different squat variations like goblet squats, pistol squats, and sumo squats.",
                "expected_output": "Detailed examples of various squat types, showing proper form and key techniques.",
                "cluster_name": "Squat Variations",
                "query_template": "how to do %squat_type% squat tutorial",
                "variables_strategy": "Generate a list of squat variations such as goblet squat, pistol squat, sumo squat, Bulgarian split squat, etc.",
                "helpful_data_for_generating_final_query_list": "Popular squat variations include goblet squat, pistol squat, sumo squat, Bulgarian split squat. Use top fitness YouTube channels for tutorials."
            },
            {
                "data_description": "Videos analyzing the biomechanics of squats, focusing on muscle engagement and joint movements.",
                "expected_output": "Detailed breakdowns of muscle and joint actions during squats, helping to understand the mechanics of the movement.",
                "cluster_name": "Squat Biomechanics",
                "query_template": "squat biomechanics analysis",
                "variables_strategy": "Focus on key movements like hip extension, knee flexion, and ankle dorsiflexion.",
                "helpful_data_for_generating_final_query_list": "Key biomechanics topics include hip extension, knee flexion, muscle engagement. Relevant YouTube sources include sports science and biomechanics channels."
            },
            {
                "data_description": "Videos focusing on squat techniques for different body types such as fat, short, and tall people.",
                "expected_output": "Guidance on adjusting squat techniques for different body types, ensuring safe and effective performance.",
                "cluster_name": "Body Type-Specific Squat Techniques",
                "query_template": "how to do squats for %body_type% people",
                "variables_strategy": "Generate a list of body types such as fat, short, and tall.",
                "helpful_data_for_generating_final_query_list": "Body types include fat, short, tall. Search for content from experts discussing how body types affect squat form on YouTube."
            }
        ]


    </good_example_1>

    <good_example_2>

        {
        "task_description": "Training a model that will detect different damages on cars and provide helpful data about how to fix these damages.",
        "perfect_training_dataset": "The ideal dataset for this task would include a collection of high-quality videos showing a wide range of car damage scenarios, with detailed annotations describing the type of damage, the severity, and the repair process. The dataset should also include videos that cover the cost estimation of repairs and the appraisal process for damaged vehicles. Additionally, the dataset should incorporate information on various car models, repair techniques, and common damage types to ensure the model can accurately detect and provide solutions for different kinds of car damage.",
        "topics_and_events": "Types of car damage (dents, scratches, rust). Repair techniques. Cost estimation for repairs. Appraisal of damaged vehicles. Differences across car models and brands.",
        "data_analysis": [
            {
            "tldr": "Identification of different types of car body damage.",
            "types_of_videos": "Videos showcasing various types of car body damage with detailed explanations.",
            "how_to_find_them": "Search YouTube with queries like 'common car body damage,' 'types of car damage,' and 'how to identify car damage.'"
            },
            {
            "tldr": "Repair techniques for different types of car damage.",
            "types_of_videos": "Tutorials on fixing different types of car body damage.",
            "how_to_find_them": "Search YouTube with queries like 'how to fix %damage_type% on %car_model%.'"
            },
            {
            "tldr": "Estimating repair costs for car damage.",
            "types_of_videos": "Videos on estimating repair costs for various types of car damage.",
            "how_to_find_them": "Search YouTube with queries like '%damage_type% repair cost estimation,' and 'car repair cost guide.'"
            },
            {
            "tldr": "Appraising damaged vehicles.",
            "types_of_videos": "Content about appraising used cars, focusing on body condition.",
            "how_to_find_them": "Search YouTube with queries like '%car_brand% used car appraisal,' and 'how to inspect used car body.'"
            }
        ],
        "data_clusters": [
            {
            "data_description": "Videos showcasing various types of car body damage with detailed explanations.",
            "expected_output": "Examples of different types of car damage, without focusing on any specific car model or damage severity.",
            "cluster_name": "Damage Identification",
            "query_template": "%car_common_body_damage_issue%",
            "variables_strategy": "Just list common body damages for cars.",
            "helpful_data_for_generating_final_query_list": "Auto body repair manuals, insurance claim statistics on common car damages."
            },
            {
            "data_description": "Tutorials on fixing different types of car body damage.",
            "expected_output": "Detailed examples of repair techniques for different car models and damage types.",
            "cluster_name": "Repair Techniques",
            "query_template": "how to fix %damage_type% on %car_model%",
            "variables_strategy": "Generate combinations of common damage types (dent, scratch, rust) and popular car models (Toyota Camry, Honda Civic, Ford F-150, etc.).",
            "helpful_data_for_generating_final_query_list": "DIY car repair forums, auto body technician training materials, lists of bestselling car models."
            },
            {
            "data_description": "Videos on estimating repair costs for various types of car damage.",
            "expected_output": "Detailed cost estimations for repairing various types of damage on different car models.",
            "cluster_name": "Cost Estimation",
            "query_template": "%damage_type% repair cost estimation",
            "variables_strategy": "Use the same list of damage types as in the Damage Identification cluster.",
            "helpful_data_for_generating_final_query_list": "Auto repair shop pricing guides, insurance claim payout data."
            },
            {
            "data_description": "Content about appraising used cars, focusing on body condition.",
            "expected_output": "Guidelines on how to appraise damaged vehicles, with a focus on different car brands.",
            "cluster_name": "Vehicle Appraisal",
            "query_template": "%car_brand% used car appraisal",
            "variables_strategy": "Generate a list of popular car brands.",
            "helpful_data_for_generating_final_query_list": "Used car valuation guides, car dealership training materials."
            }
        ]
        }

    """

    res = ask(llm_input=LLMInput(human_prompt=prompt, system_prompt=system_prompt, output_schema=QueryList), config=config)
    logger.info(f"Check: {res}")


    if not res:
        return None
    

    for cluster in res.data_clusters:
        # Step 1: Generate queries for each cluster
        cluster_prompt = {
            "expected_output": cluster.expected_output,
            "query_template": cluster.query_template,
            "cluster_name": cluster.cluster_name
        }
        
        cluster_system_prompt = """
        1. You are generating search queries. You use casual common American English.
        You don't care about being polite or inclusive. You just deliver queries for a research.
        I'll give you a JSON object that contains a query template and a description.
        Generate 20 samples for this query using description to navigate what kind of queries you need to deliver.
        Return JSON in this schema: 
        {
        "queries":
            [{"query":"string",
                "non_inclusive_alternative":"string"},
                {"query":"string",
                "non_inclusive_alternative":"string"},
                {"query":"string",
                "non_inclusive_alternative":"string"},....
            ]}
        <instructions>
        1. Always follow the template 'query_template' even for "non_inclusive_alternative"
        2. Use as less words as possible
        3. Don't improvise, be straight-forward
        4. Non_inclusive_alternative can't be too similar to the original version
        5. For names and places use full names and official names for places
        6. IMPORTANT! "non_inclusive_alternative" should fit the query_template
        Here is your object: """

        cluster_queries = ask(llm_input=LLMInput(human_prompt=json.dumps(cluster_prompt), system_prompt=cluster_system_prompt, output_schema=QueryOutput), config=config)

        if not cluster_queries:
            continue

        # Step 2: Extract non-inclusive alternatives
        non_inclusive_alternatives = [q.non_inclusive_alternative for q in cluster_queries.queries]

        # Step 3: Generate more queries based on non-inclusive alternatives
        extension_system_prompt = f"You are a helpful AI assistant. Continue the list and generate {num} samples of it:"
        extended_queries = ask(llm_input=LLMInput(human_prompt=json.dumps(non_inclusive_alternatives), system_prompt=extension_system_prompt, output_schema=QueryExtended), config=config)

        if not extended_queries:
            extended_queries = []
        
        cluster.search_queries = non_inclusive_alternatives + extended_queries.queries

        logger.info(f"Extended queries received: {len(extended_queries.queries) if extended_queries.queries else 0}")
    # Step 6: Save the updated QueryList as a JSON file
    out_path = 'cv_task_general_data.json'

    config.dump(res, config.data_dir / out_path)

    queries = []
    for c in res.data_clusters: queries.extend(c.search_queries)

    return queries

def get_video_ids(
    query: str|list[str],
    config: DatagenConfig,
    out_path = 'video_ids.json',
    videos_per_query: Optional[int] = None,
    sleep: float = 0,
    sort_by: Literal["relevance", "upload_date", "view_count", "rating"] = "relevance",
    results_type: Literal["video", "channel", "playlist", "movie"] = "video",
    only_creative_commons=True,
) -> Generator[dict, None, None]:
    if type(query) is str:
        query = [query]
    ids = set()
    for q in tqdm(query):
        for video in scrapetube.get_search(query=q, limit=videos_per_query, sleep=sleep, sort_by=sort_by, results_type=results_type):
            ids.add(video['videoId'])
    ids = list(ids)

    if only_creative_commons:
        print(datetime.now(), 'Filtering ids that for creative commons license')
        ids_cc = []
        for i in tqdm(ids):
            YDL_OPTIONS = {
                "quiet":    True,
                "simulate": True,
                "forceurl": True,
            }
            with yt_dlp.YoutubeDL(YDL_OPTIONS) as ydl:
                info = ydl.extract_info(f"youtube.com/watch?v={i}", download=False)
            if 'creative commons' in info.get('license', '').lower():
                ids_cc.append(id)
        ids = ids_cc

    config.dump(ids, config.data_dir / out_path)
    return ids

# def get_video_info(query_list, videos_per_query=100):
#     queries = {}
#     # 'format': 'bestaudio', 'noplaylist':'True', 
#     YDL_OPTIONS = {
#     # "quiet":    True,
#     "simulate": True,
#     "forceurl": True,
#     }

#     for query in tqdm(query_list):
#         if query not in queries:
#             with yt_dlp.YoutubeDL(YDL_OPTIONS) as ydl:
#                 videos = ydl.extract_info(f"ytsearch{videos_per_query}:{query}", download=False)['entries']
#             queries[query] = videos

#     q = []
#     for k,v in queries.items():
#         for vv in v:
#             vv['query'] = k
#             q.append(vv)
#     df = pd.DataFrame(q)
#     df = agg_df(df)
#     return df

# def agg_video(dfv: pd.DataFrame):
#     item = dfv.iloc[0]
#     queries = []
#     for q in dfv['query']:
#         if type(q)==str:
#             q = [q]
#         queries.append(q)
#     item['queries'] = queries
#     return item

# def agg_df(df: pd.DataFrame):
#     df_unique = df.groupby('id').apply(agg_video)
#     return df_unique