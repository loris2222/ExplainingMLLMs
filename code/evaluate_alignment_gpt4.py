# Load data
import pickle

with open("../data/gpt_evaluation/selected_images.pickle", "rb") as f:
    coco_image_data = pickle.load(f)

CAPTION_SOURCE = "owlvit_llava_deepermlp_includecls"  # "gt", "original_llava", "owlvit_llava", "owlvit_llava_coco", "owlvit_llava_includecls", "owlvit_llava_deepermlp_includecls"

if CAPTION_SOURCE == "gt":
    caption_key = "caption"
elif CAPTION_SOURCE == "original_llava":
    caption_key = "caption_original_llava"
elif CAPTION_SOURCE == "owlvit_llava":
    caption_key = "caption_owlvit_llava"
elif CAPTION_SOURCE == "owlvit_llava_coco":
    caption_key = "caption_owlvit_llava_coco"
elif CAPTION_SOURCE == "owlvit_llava_includecls":
    caption_key = "caption_owlvit_llava_includecls"
elif CAPTION_SOURCE == "owlvit_llava_deepermlp_includecls":
    caption_key = "caption_owlvit_llava_deepermlp_includecls"  # NOTE this actually does not include CLS
else:
    raise ValueError(f"Invalid CAPTION_SOURCE {CAPTION_SOURCE}")

# Run GPT4 on captions
from openai import OpenAI
from tqdm import tqdm

with open("../auth/openai.txt", "r") as f:
    openai_key = f.read().strip()
with open("../auth/openai_org.txt", "r") as f:
    openai_org = f.read().strip()

client = OpenAI(
    api_key=openai_key,
    organization=openai_org
)

responses = []

for data in tqdm(coco_image_data):
    system_prompt = "Please act as an impartial judge and evaluate the quality of the caption provided by an" \
                    " AI assistant for the provided image. Your evaluation should consider factors" \
                    " such as the helpfulness, relevance, accuracy, depth, and level of detail of" \
                    " the caption. Be as objective as possible." \

    user_prompt = f"From 1 to 10, score this caption for the image: \"{data[caption_key]}\". Only answer with a number from 1 to 10."
    response = client.chat.completions.create(
        model="gpt-4-vision-preview",
        temperature=0,
        messages=[
            {
                "role": "system",
                "content": [
                    {
                        "type": "text", "text": system_prompt
                    },
                ],
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "text", "text": user_prompt
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": data["url"],
                        }
                    },
                ],
            },
        ],
        max_tokens=1000,
    )
    responses.append(response)

with open(f"../data/gpt_evaluation/responses_coco_{CAPTION_SOURCE}.pickle", "wb") as f:
    pickle.dump(responses, f)

with open(f"../data/gpt_evaluation/responses_coco_{CAPTION_SOURCE}.pickle", "rb") as f:
    reloaded_responses = pickle.load(f)

for response in reloaded_responses:
    print(response.choices[0].message.content, end=", ")

# Compute average score
num_scores = 0
sum_scores = 0
for response in reloaded_responses:
    try:
        score = int(response.choices[0].message.content)
        num_scores += 1
        sum_scores += score
    except Exception as e:
        print(response)

print(f"Computed average over {num_scores} (skipped {len(reloaded_responses) - num_scores} invalid scores) scores: {sum_scores/num_scores}")

# Print low scoring ones
for idx, response in enumerate(reloaded_responses):
    try:
        score = int(response.choices[0].message.content)
    except Exception as e:
        continue

    if score < 4:
        print("*"*30)
        print(coco_image_data[idx]["caption"])
        print(coco_image_data[idx]["caption_original_llava"])
        print(coco_image_data[idx][caption_key])


