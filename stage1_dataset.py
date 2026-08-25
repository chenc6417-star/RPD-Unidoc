import json
import ast
import random
import re

TEXT_TO_BOX_PROMPTS = [
    "Please provide the bounding box for the content '{text}' in this document page.",
    "Where is the '{label}' located that starts with '{text}'? Please output the coordinates.",
    "Please locate the region containing the following content: '{text}'",
    "Identify the coordinates for the text that reads '{text}'.",
    "Find the bounding box for the {label} section containing '{text}'."
]

BOX_TO_TEXT_PROMPTS = [
    "Please extract the label and content corresponding to the bounding box {box}.",
    "What is the content and label inside the region {box}?",
    "Please tell me the text information enclosed by the coordinates {box}.",
    "Identify and output the category label and content within the bounding box {box}."
]

def create_swift_message(image_path, query, response):
    return {
        "messages": [
            {
                "role": "user",
                "content": [{"type": "image", "image": image_path}, {"type": "text", "text": query}]
            },
            {
                "role": "assistant",
                "content": [{"type": "text", "text": response}]
            }
        ]
    }
# key: "messages" -> list of dicts
def generate_vqa_dataset(input_file, output_file):
    new_dataset = []
    error_count = 0
    success_count = 0
    
    with open(input_file, 'r', encoding='utf-8') as fin:
        for line_idx, line in enumerate(fin):
            line = line.strip()
            if not line: continue

            try:
                data = json.loads(line)
                image_path = ""
                for item in data["messages"][0]["content"]:
                    if item["type"] == "image":
                        image_path = item["image"]
                        break
                
                content_str = data["messages"][1]["content"][0]["text"].strip()

                try:
                    elements = ast.literal_eval(content_str)
                except (SyntaxError, ValueError):
                    try:
                        fixed_str = content_str.replace('\n', '\\n')
                        elements = ast.literal_eval(fixed_str)
                    except:
                        error_count += 1
                        continue

                for item in elements:
                    if len(item) < 3: continue
                    
                    box = item[0]
                    label = item[1]
                    # original text
                    text = str(item[2])

                    if label == 'fig':
                        text = ""
                    # ---------------------------------------------
                    
                    box_str = str(box)

                    if text.strip():
                        short_text = text[:40].replace("\n", " ").strip()
                        if len(text) > 40: short_text += "..."
                        
                        q_t2b = random.choice(TEXT_TO_BOX_PROMPTS).format(text=short_text, label=label)
                        new_dataset.append(create_swift_message(image_path, q_t2b, box_str))
                    q_b2t = random.choice(BOX_TO_TEXT_PROMPTS).format(box=box_str)
                    r_b2t = f"[{label}] {text}".strip()
                    
                    new_dataset.append(create_swift_message(image_path, q_b2t, r_b2t))
                
                success_count += 1

            except Exception:
                error_count += 1
    random.shuffle(new_dataset)
    with open(output_file, 'w', encoding='utf-8') as fout:
        for item in new_dataset:
            fout.write(json.dumps(item, ensure_ascii=False) + '\n')
            
    print("-" * 30)
    print(f"success_count: {success_count}")
    print(f"error_count: {error_count}")
    print("-" * 30)

if __name__ == "__main__":
    generate_vqa_dataset("qwen35_unified_full_v2.jsonl", "qwen35_vqa_short_format.jsonl")