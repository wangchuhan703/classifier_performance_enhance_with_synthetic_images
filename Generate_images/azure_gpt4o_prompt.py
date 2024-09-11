import json
import os

import openai
import time
from joblib import Parallel, delayed

from openai import AzureOpenAI

# domainNet_classnames = ['aircraft carrier', 'airplane', 'alarm clock', 'ambulance', 'angel',
#                         'animal migration', 'ant', 'anvil', 'apple', 'arm',
#                         'asparagus', 'axe', 'backpack', 'banana', 'bandage',
#                         'barn', 'baseball', 'baseball bat', 'basket', 'basketball',
#                         'bat', 'bathtub', 'beach', 'bear', 'beard',
#                         'bed', 'bee', 'belt', 'bench', 'bicycle',
#                         'binoculars', 'bird', 'birthday cake', 'blackberry', 'blueberry',
#                         'book', 'boomerang', 'bottlecap', 'bowtie', 'bracelet',
#                         'brain', 'bread', 'bridge', 'broccoli', 'broom',
#                         'bucket', 'bulldozer', 'bus', 'bush', 'butterfly',
#                         'cactus', 'cake', 'calculator', 'calendar', 'camel',
#                         'camera', 'camouflage', 'campfire', 'candle', 'cannon',
#                         'canoe', 'car', 'carrot', 'castle', 'cat',
#                         'ceiling fan', 'cello', 'cell phone', 'chair', 'chandelier',
#                         'church', 'circle', 'clarinet', 'clock', 'cloud',
#                         'coffee cup', 'compass', 'computer', 'cookie', 'cooler',
#                         'couch', 'cow', 'crab', 'crayon', 'crocodile',
#                         'crown', 'cruise ship', 'cup', 'diamond', 'dishwasher',
#                         'diving board', 'dog', 'dolphin', 'donut', 'door',
#                         'dragon', 'dresser', 'drill', 'drums', 'duck',
#                         'dumbbell', 'ear', 'elbow', 'elephant', 'envelope',
#                         'eraser', 'eye', 'eyeglasses', 'face', 'fan',
#                         'feather', 'fence', 'finger', 'fireplace', 'firetruck',
#                         'fire hydrant', 'fish', 'flamingo', 'flashlight', 'flip flops',
#                         'floor lamp', 'flower', 'flying saucer', 'foot', 'fork',
#                         'frog', 'frying pan', 'garden', 'garden hose', 'giraffe',
#                         'goatee', 'golf club', 'grapes', 'grass', 'guitar',
#                         'hamburger', 'hammer', 'hand', 'harp', 'hat',
#                         'headphones', 'hedgehog', 'helicopter', 'helmet', 'hexagon',
#                         'hockey puck', 'hockey stick', 'horse', 'hospital', 'hot air balloon',
#                         'hot dog', 'hot tub', 'hourglass', 'house', 'house plant',
#                         'hurricane', 'ice cream', 'jacket', 'jail', 'kangaroo',
#                         'key', 'keyboard', 'knee', 'knife', 'ladder',
#                         'lantern', 'laptop', 'leaf', 'leg', 'lighter',
#                         'lighthouse', 'lightning', 'light bulb', 'line', 'lion',
#                         'lipstick', 'lobster', 'lollipop', 'mailbox', 'map',
#                         'marker', 'matches', 'megaphone', 'mermaid', 'microphone',
#                         'microwave', 'monkey', 'moon', 'mosquito', 'motorbike',
#                         'mountain', 'mouse', 'moustache', 'mouth', 'mug',
#                         'mushroom', 'nail', 'necklace', 'nose', 'ocean',
#                         'octagon', 'octopus', 'onion', 'oven', 'owl',
#                         'paintbrush', 'paint can', 'palm tree', 'panda', 'pants',
#                         'paper clip', 'parachute', 'parrot', 'passport', 'peanut',
#                         'pear', 'peas', 'pencil', 'penguin', 'piano',
#                         'pickup truck', 'picture frame', 'pig', 'pillow', 'pineapple',
#                         'pizza', 'pliers', 'police car', 'pond', 'pool',
#                         'popsicle', 'postcard', 'potato', 'power outlet', 'purse',
#                         'rabbit', 'raccoon', 'radio', 'rain', 'rainbow',
#                         'rake', 'remote control', 'rhinoceros', 'rifle', 'river',
#                         'rollerskates', 'roller coaster', 'sailboat', 'sandwich', 'saw',
#                         'saxophone', 'school bus', 'scissors', 'scorpion', 'screwdriver',
#                         'sea turtle', 'see saw', 'shark', 'sheep', 'shoe',
#                         'shorts', 'shovel', 'sink', 'skateboard', 'skull',
#                         'skyscraper', 'sleeping bag', 'smiley face', 'snail', 'snake',
#                         'snorkel', 'snowflake', 'snowman', 'soccer ball', 'sock',
#                         'speedboat', 'spider', 'spoon', 'spreadsheet', 'square',
#                         'squiggle', 'squirrel', 'stairs', 'star', 'steak',
#                         'stereo', 'stethoscope', 'stitches', 'stop sign', 'stove',
#                         'strawberry', 'streetlight', 'string bean', 'submarine', 'suitcase',
#                         'sun', 'swan', 'sweater', 'swing set', 'sword',
#                         'syringe', 't-shirt', 'table', 'teapot', 'teddy-bear',
#                         'telephone', 'television', 'tennis racquet', 'tent', 'The Eiffel Tower',
#                         'The Great Wall of China', 'The Mona Lisa', 'tiger', 'toaster', 'toe',
#                         'toilet', 'tooth', 'toothbrush', 'toothpaste', 'tornado',
#                         'tractor', 'traffic light', 'train', 'tree', 'triangle',
#                         'trombone', 'truck', 'trumpet', 'umbrella', 'underwear',
#                         'van', 'vase', 'violin', 'washing machine', 'watermelon',
#                         'waterslide', 'whale', 'wheel', 'windmill', 'wine bottle',
#                         'wine glass', 'wristwatch', 'yoga', 'zebra', 'zigzag']

domainNet_classnames = ['aircraft carrier', 'airplane', 'alarm clock']

# domainNet_domains = ['clipart', 'infograph', 'quickdraw', 'painting', 'sketch', 'real']

domainNet_domains = ['clipart']

print(len(domainNet_classnames))


client = AzureOpenAI(
  azure_endpoint = "https://openai-mlsp-1.openai.azure.com/",
  api_key="5dafc4526c224ac1bc5ce915b5d200a7",
  api_version="2024-05-01-preview"
)

def get_completion_with_retry(prompt, seed, max_retries=5, delay=10):
    retries = 0
    while retries < max_retries:
        try:
            response = client.chat.completions.create(
                model="gpt4o", # model = "deployment_name".
                messages=[
                    {"role": "system", "content": "You are an AI assistant that helps people find information."},
                    {"role": "user", "content": prompt}
                ],
                seed=seed,
                max_tokens=60,
                top_p=0.8,
                temperature=1.5,
                logprobs=True
            )
            #return response.choices[0].message.content
            return response
        except openai.RateLimitError as e:
            retries += 1
            print(f"Rate limit exceeded. Retrying {retries}/{max_retries} in {delay} seconds...")
            time.sleep(delay)
        except Exception as e:
            return f"Error: {str(e)}"
    return f"Failed after {max_retries} retries."




for domain in domainNet_domains:
    responses_dict = {}

    for classname in domainNet_classnames:
        # prompts = [f"The definition of {classname} in the format of: a {domain} of {classname}, definition"] * 10
        prompts = [f"The definition of {classname}, max_tokens=60"] * 10

        responses = Parallel(n_jobs=-1, prefer="threads")(
            delayed(get_completion_with_retry)(prompt, seed)
            for seed, prompt in enumerate(prompts)
        )

        responses_dict[classname] = {"prompt": []}

        for idx, response in enumerate(responses):
            if isinstance(response, str):
                print(f"Failed to get response for seed {idx}: {response}")
                responses_dict[classname]["prompt"].append(f"Error: {response}")
            else:
                try:
                    # 提取内容并保存到字典
                    content = response.choices[0].message.content
                    responses_dict[classname]["prompt"].append(content)
                except Exception as e:
                    print(f"Error processing response for seed {idx}: {str(e)}")
                    responses_dict[classname]["prompt"].append("Error processing response")


    # Saving the file
    directory = "gpt4o_domainNet_prompts"
    if not os.path.exists(directory):
        os.makedirs(directory)
    file_path = os.path.join(directory, f"{domain}_domainNet_prompts.json")

    with open(file_path, 'w') as file:
        json.dump(responses_dict, file, indent=4)
    print(f"JSON file '{file_path}' has been created with the domainNet prompts.")