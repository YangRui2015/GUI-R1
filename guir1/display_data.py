# display image, instruction, response, and action from the output json file
import json
import os
from PIL import Image, ImageDraw
from io import BytesIO
from datasets import load_dataset

def load_json(json_file):
    results = []
    with open(json_file, 'r') as f:
        # each line is a json object
        for line in f.readlines():
            data = json.loads(line)
            results.append(data)
    return results


def display_data(datas, original_datas, idx):
    # display image, instruction, response, and action from the output json file
    #  The example data format is as follows:
    # {"image": "", "history": " \nStep 1: PRESS_HOME\nStep 2: SCROLL [RIGHT]\nStep 3: SCROLL [RIGHT]\nStep 4: CLICK <point>[[634,113]]</point>\nStep 5: CLICK <point>[[365,153]]</point>\nStep 6: TYPE [a travel guide to visit Hong Kong]\nStep 7: CLICK <point>[[909,78]]</point>\nStep 8: SCROLL [UP]\nStep 9: SCROLL [UP]\nStep 10: SCROLL [UP]\nStep 11: SCROLL [UP]\nStep 12: CLICK <point>[[372,599]]</point>\nStep 13: CLICK <point>[[352,71]]</point>\nStep 14: CLICK <point>[[464,157]]</point>\nStep 15: PRESS_HOME\nStep 16: SCROLL [LEFT]\nStep 17: SCROLL [LEFT]\nStep 18: CLICK <point>[[605,242]]</point>\nStep 19: CLICK <point>[[715,925]]</point>\nStep 20: TYPE [a travel guide to visit HK]\nStep 21: CLICK <point>[[190,148]]</point>\nStep 22: CLICK <point>[[488,638]]</point>\nStep 23: CLICK <point>[[559,596]]</point>\nStep 24: PRESS_HOME\nStep 25: SCROLL [UP]\nStep 26: SCROLL [UP]\nStep 27: CLICK <point>[[627,610]]</point>\nStep 28: CLICK <point>[[78,78]]</point>\nStep 29: CLICK <point>[[270,284]]</point>\nStep 30: CLICK <point>[[284,151]]</point>\nStep 31: TYPE [Hong Kong]\nStep 32: CLICK <point>[[185,254]]</point>\nStep 33: SCROLL [UP]\nStep 34: SCROLL [UP]\nStep 35: SCROLL [UP]\nStep 36: SCROLL [LEFT]\nStep 37: SCROLL [LEFT]\nStep 38: CLICK <point>[[477,415]]</point>\nStep 39: SCROLL [LEFT]\nStep 40: CLICK <point>[[649,417]]</point>\nStep 41: PRESS_HOME\nStep 42: CLICK <point>[[863,518]]</point>\nStep 43: CLICK <point>[[107,76]]</point>\nStep 44: CLICK <point>[[284,324]]</point>\nStep 45: TYPE [Hong Kong]\nStep 46: CLICK <point>[[345,208]]</point>\nStep 47: CLICK <point>[[399,395]]</point>\nStep 48: CLICK <point>[[506,355]]</point>\nStep 49: CLICK <point>[[637,934]]</point>\nStep 50: CLICK <point>[[428,597]]</point>\n\n", "instruction": "Search for a travel guide to visit Hong Kong, record the resource website, then select a day without rain, finally book a flight.", "gt_action": "click", "gt_bbox": [552, 1422], "gt_input_text": "no input text", "group": "web", "ui_type": "click", "orig_w": 1440, "orig_h": 3120, "scale": [0.7583333333333333, 0.7628205128205128], "image_size": [1092, 2380], "pred": "<think>\nI observe that I am on a flight booking interface showing search results for flights from San Francisco (SFO) to Hong Kong (HKG) for May 15. The list displays several flight options with airlines, departure/arrival times, durations, and prices. According to the instruction, after searching for a travel guide and selecting a day without rain, I should now \"book a flight.\" \n\nThe interaction history shows that I have already searched for a travel guide, selected Hong Kong as my destination, and navigated through the flight search results. Now, I need to proceed with booking a flight. The first step in booking is typically to select a specific flight option from the list. The first flight listed departs at 13:00 and arrives at 18:45+1, operated by United Airlines, and is labeled as \"Fastest.\" This is a logical choice if I want to minimize travel time.\n\nTo proceed with booking, I should click on this first flight option to view more details or begin the booking process.\n</think>\n<answer>\n{\n\"action_description\": \"Click on the first flight option departing at 13:00 from SFO to HKG to proceed with booking.\",\n\"action_type\": \"CLICK\",\n\"action_target\": \"First flight option row, showing 13:00 SFO to 18:45+1 HKG, United Airlines, labeled 'Fastest'\",\n\"value\": \"None\"\n}\n</answer>", "pred_action": "click", "pred_input_text": "None", "pred_action_description": "Click on the first flight option departing at 13:00 from SFO to HKG to proceed with booking.", "pred_action_target": "First flight option row, showing 13:00 SFO to 18:45+1 HKG, United Airlines, labeled 'Fastest'", "pred_coord": [546, 1118]}
    # the image binary data is stored in the original data, but the idx is not the same as the output json file
    data = datas[idx]
    history = data['history']
    instruction = data['instruction']
    pred = data['pred']
    pred_action = data['pred_action']
    pred_input_text = data['pred_input_text']
    # pred_action_description = data['pred_action_description']
    # pred_action_target = data['pred_action_target']
    pred_coord = data['pred_coord']
    pred_coord[0] = pred_coord[0]  / data['image_size'][0] * data['orig_w']
    pred_coord[1] = pred_coord[1]  / data['image_size'][1] * data['orig_h']
    ground_truth_action = data['gt_action']
    ground_truth_bbox = data['gt_bbox']
    ground_truth_input_text = data['gt_input_text']

    # find the corresponding original data by matching the instruction using instruction and step number in history to match
    step_num = history.count('Step ') + 1
    original_data = None
    for od in original_datas:
        od['step_num'] = od['history'].count('Step ') + 1
        if od['instruction'] == instruction and od['step_num'] == step_num:
            original_data = od
            break
    if original_data is None:
        print(f'Cannot find the original data for idx {idx}')
        return

    img = Image.open(BytesIO(original_data["image"]["bytes"]))

    # Draw predicted coordinate using PIL
    if pred_coord != "None" and pred_coord != []:
        x, y = pred_coord
        if x >= 0 and y >= 0 and x < img.width and y < img.height:
            r = 10  # radius
            draw = ImageDraw.Draw(img)
            draw.ellipse((x - r, y - r, x + r, y + r), outline='red', width=5)

    # Draw ground truth bbox using PIL
    if ground_truth_bbox != "None" and ground_truth_bbox != []:
        x, y = ground_truth_bbox
        if x >= 0 and y >= 0 and x < img.width and y < img.height:
            r = 10  # radius
            draw = ImageDraw.Draw(img)
            draw.ellipse((x - r, y - r, x + r, y + r), outline='green', width=5)
    
    # save image to temp.png
    img.save('temp.png')


# main function to test
if __name__ == '__main__':
    # load the output json file
    # json_file = '/projects/illinois/eng/ece/huanz/ry21/GUI-R1/outputs/gui-planner-7B-v2/guiodyssey_test.json'
    # original_json_file = '/projects/illinois/eng/ece/huanz/ry21/GUI-R1/datasets/GUI-R1/guiodyssey_test.parquet'
    json_file = '/projects/illinois/eng/ece/huanz/ry21/GUI-R1/outputs/qwen2_5_vl_7b_guir1_grpo_1080p_4gpu_cpt155_maxres1024_pngv2/androidcontrol_high_test.json'
    original_json_file = '/projects/illinois/eng/ece/huanz/ry21/GUI-R1/datasets/GUI-R1/androidcontrol_high_test.parquet'
    datas = load_json(json_file)
    original_datas = load_dataset("parquet", data_files=original_json_file, split="train")
    print(f'Loaded {len(datas)} data from {json_file}')
    print(f'Loaded {len(original_datas)} original data from {original_json_file}')

    # display the first data
    # idx = 0
    for idx in range(1, 20):
        print(f'Displaying data idx {idx}')
        display_data(datas, original_datas, idx)
        # better organize the output information, e.g., action, input text, action description, action target put in a single line
        data = datas[idx]
        print(f'Instruction: {data["instruction"]}\n')
        print(f'History: {data["history"]}\n')
        print(f'Prediction: {data["pred"]}\n')
        print(f'Predicted Coordinate (red circle): {data["pred_coord"]}\n')
        print(f'Ground Truth Action (green circle): {data["gt_action"]}, Ground Truth Input Text: {data["gt_input_text"]}, Ground Truth BBox: {data["gt_bbox"]}\n')
        breakpoint()