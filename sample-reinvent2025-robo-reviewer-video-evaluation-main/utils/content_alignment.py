"""
Copyright 2025 Amazon.com, Inc. or its affiliates. All Rights Reserved.

This is AWS Content subject to the terms of the Customer Agreement
----------------------------------------------------------------------
Package content:
    Content alignment evaluation utilities using Q&A generation and video analysis
"""

import time
import json
import html
import re
from botocore.exceptions import ClientError
import boto3
from tqdm import tqdm

qa_gen_prompt = """
## Task
Your task is to assess the alignment of video content with textual descriptions by formulating structured questions and answers.

## Video Description
{{video_prompt}}

## Focus Area
{{focus_area}}

## Instructions
1. Decompose the given video description into atomic tuples, where each tuple represents the smallest unit of meaning that accurately captures an aspect of the video related to {{focus_area}}.
    - Each atomic tuple consists of two elements: 
        1) The first element indicates a global or local object in the video.
        2) The second element specifies an attribute, detail, activity, color, material, spatial information, location, shape, OCR text, or any other relevant information about the object.
    - Avoid including resolution or non-content-related details in the atomic tuples, such as "4k" or "photorealistic"

2. Only keep the atomic tuples that are directly related to {{focus_area}}. Removed other unrelated tuples

3. Generate 5 questions based on the atomic tuples, targeting the specific video aspect indicated by the tuple with emphasis on {{focus_area}}. Ensure that the questions reflect the atomicity principle, avoiding over-fragmentation or excessive aggregation of concepts.

4. Provide direct and relevant answer choices for each question based on the corresponding atomic tuple. Include "NONE" as an option if the existence of an entity or attribute is uncertain.

5. Position the correct answer immediately after each question for clarity.

6. Start your response with a list of the derived atomic tuples from the video description.

7. For questions related to entities or subjects, use the format "If there is a [entity], ..." to account for the possibility that the entity may not be present in the video.

### Response Format
Provide a valid JSON response in exactly this format (all strings must be in double quotes):
{
    "atomic_tuples": [
        [first element, second element],
        [first element, second element],
        ...
    ],
    "questions": [
        {
            "question": "Question 1",
            "answer_choices": [
                Answer Choice 1,
                Answer Choice 2,
                ...
            ],
            "correct_answer": [Correct Answer]
        },
        {
            "question": "Question 2",
            "answer_choices": [
                Answer Choice 1,
                Answer Choice 2,
                ...
            ],
            "correct_answer": [Correct Answer]
        },
        ...
    ]
}

IMPORTANT: 
- All strings must be enclosed in double quotes
- Use "NONE" (with quotes) as an option, NOT ""NONE""
- Provide valid JSON only, no additional text
- Do not use double quotes around already quoted strings
"""


def generate_qa_alignment(
        boto3_session, 
        video_prompt, 
        focus_area,
        model_id="us.amazon.nova-premier-v1:0"
    ):
    """Generate Q&A pairs for video evaluation based on a specific focus area.
    
    Args:
        boto3_session: AWS boto3 session for Bedrock access
        video_prompt (str): Text description of the video content
        focus_area (str): Evaluation focus area (e.g., 'subject alignment')
        model_id (str): Bedrock model ID for Q&A generation
        
    Returns:
        dict: JSON object with atomic tuples and 5 Q&A pairs, or empty dict on error
    """

    bedrock_runtime = boto3_session.client("bedrock-runtime")

    input_prompt = qa_gen_prompt.replace("{{video_prompt}}", video_prompt).replace("{{focus_area}}", focus_area)

    retry_delays = [1, 2, 4, 8, 16]
    messages = [
    {"role": "user",
        "content": [{"text": input_prompt}]}
    ]
    for attempt, delay in enumerate(retry_delays + [None]):
        print(f"invoke LLM - {attempt}" + " "*10)#, end = "\r")
        try:
            response = bedrock_runtime.converse(
                modelId=model_id,
                messages=messages,
                inferenceConfig={"temperature": 0.4},
            )
            generated_text = response['output']['message']['content'][0]["text"]
            break
        except (ClientError, Exception) as e:
            print(f"ERROR: Can't invoke '{model_id}'. Reason: {e}")
            if delay is not None:
                time.sleep(delay)  # Required: Exponential backoff retry strategy for API resilience
            else:
                return {}

    try:
        # Extract JSON from the generated text
        start_idx = generated_text.find('{')
        end_idx = generated_text.rfind('}') + 1
        if start_idx != -1 and end_idx > start_idx:
            json_str = generated_text[start_idx:end_idx]
            # Decode HTML entities
            json_str = html.unescape(json_str)
            
            # Try to fix common JSON formatting issues
            # Fix double quotes around NONE and other strings
            json_str = re.sub(r'""([^"]+)""', r'"\1"', json_str)
            # Replace NONE without quotes (if any remain)
            json_str = re.sub(r'(?<!")\bNONE\b(?!")', '"NONE"', json_str)
            
            print(json_str)
            return json.loads(json_str)
        else:
            return {}
    except json.JSONDecodeError as e:
        print(f"JSON decode error: {e}")
        if 'json_str' in locals():
            print(f"Problematic JSON: {json_str[:200]}...")
        return {}


def evaluate_video_qa(
        boto3_session,
        s3_video_uri,
        question,
        answer_choices,
        model_id="us.amazon.nova-premier-v1:0"
    ):
    """Evaluate a video against a specific question using multimodal AI.
    
    Args:
        boto3_session: AWS boto3 session for Bedrock access
        s3_video_uri (str): S3 URI of the video file to evaluate
        question (str): Question to ask about the video
        answer_choices (list): List of possible answer choices
        model_id (str): Bedrock model ID for video analysis
        
    Returns:
        str: Selected answer from the choices, or empty string on error
    """
    
    bedrock_runtime = boto3_session.client("bedrock-runtime")
    
    choices_text = "\n".join([f"- {choice}" for choice in answer_choices])
    
    prompt = f"""Here is the question: \n{question}

Here is the answer choices:
{choices_text}

Please select the best answer from the options above. 
If you can not answer the question, select "None" as your final answer.
Respond with only the exact text of your chosen answer."""
    
    messages = [{
        "role": "user",
        "content": [
            {"video": {
                "source": {"s3Location": {"uri": s3_video_uri}},
                "format": "mp4"
                }},
            {"text": prompt}
        ]
    }]
    
    retry_delays = [1, 2, 4, 8, 16]
    for attempt, delay in enumerate(retry_delays + [None]):
        print(f"invoke LLM - attempt {attempt}" + " "*10)#, end = "\r")
        try:
            response = bedrock_runtime.converse(
                modelId=model_id,
                messages=messages,
                inferenceConfig={"temperature": 0.1}
            )
            answer = response['output']['message']['content'][0]["text"]
            return answer
        except (ClientError, Exception) as e:
            print(f"ERROR: Can't invoke '{model_id}'. Reason: {e}")
            if delay is not None:
                time.sleep(delay)  # Required: Exponential backoff retry strategy for API resilience
            else:
                return ""


def evaluation_pipeline(
        s3_video_uri, 
        boto3_session=None, 
        model_id="us.amazon.nova-premier-v1:0",
        focus_areas= [
            "subject_alignment",
            "background_alignment",
            "color_accuracy",
            "activity_alignment",
            "spatial_relationships"
        ]
    ):
    """Complete video evaluation pipeline across multiple focus areas.
    
    Evaluates a video against its text prompt across 5 focus areas, generating
    Q&A pairs and scoring alignment. Saves results and Q&A data to S3.
    
    Args:
        s3_video_uri (str): S3 URI of video file (expects corresponding _prompt.txt)
        boto3_session: AWS boto3 session, creates default if None
        model_id (str): Bedrock model ID for evaluation
        focus_areas (list): List of evaluation focus areas
        
    Returns:
        dict: Evaluation results with video URI as key and scores per focus area
    """
    
    if boto3_session is None:
        boto3_session = boto3.Session()
    
    # Extract prompt file path from video URI
    prompt_uri = s3_video_uri.replace('.mp4', '_prompt.txt')
    
    # Read prompt from S3
    s3 = boto3_session.client('s3')
    bucket = s3_video_uri.split('/')[2]
    video_key = '/'.join(s3_video_uri.split('/')[3:])
    prompt_key = '/'.join(prompt_uri.split('/')[3:])
    
    try:
        response = s3.get_object(Bucket=bucket, Key=prompt_key)
        video_prompt = response['Body'].read().decode('utf-8')
    except Exception as e:
        print(f"Error reading prompt file: {e}")
        return {}

    
    results = {s3_video_uri: {}}
    all_qa_data = {}
    
    for focus_area in tqdm(focus_areas, desc="Evaluating focus areas"):
        print(f"Evaluating {focus_area}...")
        
        # Generate Q&A for this focus area
        qa_data = generate_qa_alignment(boto3_session, video_prompt, focus_area, model_id)
        all_qa_data[focus_area] = qa_data

        if not qa_data or "questions" not in qa_data:
            results[s3_video_uri][focus_area] = 0
            continue
            
        score = 0
        questions = qa_data["questions"]
        
        for i, q_data in enumerate(questions):
            print(f"Answer Question {i+1}/{len(questions)}" + " "*10, end = "\r")
            question = q_data["question"]
            answer_choices = q_data["answer_choices"]
            correct_answer = q_data["correct_answer"][0]
            
            # Get model's answer
            model_answer = evaluate_video_qa(boto3_session, s3_video_uri, question, answer_choices, model_id)
            
            # Check if answer is correct
            if model_answer.strip().lower() == correct_answer.lower():
                score += 1
                
        results[s3_video_uri][focus_area] = score
    
    # Save results to S3
    video_filename = video_key.replace('.mp4', '')
    
    # Save alignment scores
    alignment_key = f"{video_filename}_alignment.json"
    s3.put_object(
        Bucket=bucket,
        Key=alignment_key,
        Body=json.dumps(results, indent=2),
        ContentType='application/json'
    )
    
    # Save Q&A data
    qa_key = f"{video_filename}_alignment_qa.json"
    s3.put_object(
        Bucket=bucket,
        Key=qa_key,
        Body=json.dumps(all_qa_data, indent=2),
        ContentType='application/json'
    )
    
    print(f"Saved alignment scores to s3://{bucket}/{alignment_key}")
    print(f"Saved Q&A data to s3://{bucket}/{qa_key}")
    
    return results


if __name__ == "__main__":
    boto3_session = boto3.Session(profile_name="default")
    '''
    video_prompt = "A mushroom drinking a cup of coffee while sitting on a couch, photorealistic."
    focus_area = "action_alignment"
    result = generate_qa_alignment(boto3_session, video_prompt, focus_area)
    print(json.dumps(result, indent=4))

    question = result["questions"][0]["question"]
    choices = result["questions"][0]["answer_choices"]
    print(question)
    print(choices)
    print(evaluate_video_qa(
        boto3_session,
        "s3://haochen-dev3/generated_videos/video_1_2_5.mp4",
        question,
        choices))
    '''

    result = evaluation_pipeline(
        "s3://haochen-dev3/generated_videos/uiu8csfrd108.mp4",
        boto3_session=boto3_session,
        model_id="us.amazon.nova-premier-v1:0"
    )
    print(result)