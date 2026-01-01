"""
Copyright 2025 Amazon.com, Inc. or its affiliates. All Rights Reserved.

This is AWS Content subject to the terms of the Customer Agreement
----------------------------------------------------------------------
Package content:
    Video quality evaluation utilities using LLM as judge across multiple dimensions
"""

import time
import json
from botocore.exceptions import ClientError
import boto3
from tqdm import tqdm

# Evaluation prompts for different quality metrics
TEMPORAL_CONSISTENCY_PROMPT = """
Evaluate the temporal consistency of this video on a scale of 1-5.

Temporal consistency measures frame-to-frame coherence and smoothness:
• Smooth transitions between consecutive frames without flickering or jarring jumps
• Consistent object appearance and positioning across time
• Natural motion flow without abrupt changes or discontinuities

Rate the video from 1 (severe flickering, jarring jumps, inconsistent objects) to 5 (perfectly smooth transitions, stable objects, natural motion flow).

Provide your response in this format:
<score>X</score>
<justification>Brief explanation of why this score was assigned based on the criteria above.</justification>
"""

AESTHETIC_QUALITY_PROMPT = """
Evaluate the aesthetic quality of this video on a scale of 1-5.

Aesthetic quality assesses visual appeal and artistic composition:
• Visual composition including framing, balance, and rule of thirds
• Color harmony, lighting quality, and overall mood
• Artistic merit and creative visual elements

Rate the video from 1 (poor framing, harsh lighting, unappealing colors) to 5 (excellent composition, harmonious colors, artistic lighting).

Provide your response in this format:
<score>X</score>
<justification>Brief explanation of why this score was assigned based on the criteria above.</justification>
"""

TECHNICAL_QUALITY_PROMPT = """
Evaluate the technical quality of this video on a scale of 1-5.

Technical quality evaluates technical aspects including clarity and sharpness:
• Image sharpness and detail clarity throughout the video
• Absence of visual artifacts, noise, or compression distortions
• Proper exposure, contrast, and color accuracy

Rate the video from 1 (blurry, artifacts, poor exposure) to 5 (crystal clear, no artifacts, perfect exposure).

Provide your response in this format:
<score>X</score>
<justification>Brief explanation of why this score was assigned based on the criteria above.</justification>
"""

MOTION_EFFECTS_PROMPT = """
Evaluate the motion effects of this video on a scale of 1-5.

Motion effects evaluates quality of movement and dynamics:
• Realistic and natural movement patterns of objects and subjects
• Smooth camera movements and transitions without jerkiness
• Dynamic visual elements that enhance the overall viewing experience

Rate the video from 1 (unnatural motion, jerky camera, static visuals) to 5 (realistic movement, smooth camera work, engaging dynamics).

Provide your response in this format:
<score>X</score>
<justification>Brief explanation of why this score was assigned based on the criteria above.</justification>
"""

def evaluate_video_quality_metric(
        boto3_session,
        s3_video_uri,
        metric_prompt,
        model_id="us.amazon.nova-premier-v1:0"
    ):
    """Evaluate a video against a specific quality metric using multimodal AI.
    
    Args:
        boto3_session: AWS boto3 session for Bedrock access
        s3_video_uri (str): S3 URI of the video file to evaluate
        metric_prompt (str): Evaluation prompt for the specific metric
        model_id (str): Bedrock model ID for video analysis
        
    Returns:
        tuple: (score, justification) or (0, "") on error
    """
    
    bedrock_runtime = boto3_session.client("bedrock-runtime")
    
    messages = [{
        "role": "user",
        "content": [
            {"video": {
                "source": {"s3Location": {"uri": s3_video_uri}},
                "format": "mp4"
                }},
            {"text": metric_prompt}
        ]
    }]
    
    retry_delays = [1, 2, 4, 8, 16]
    for attempt, delay in enumerate(retry_delays + [None]):
        print(f"invoke LLM - attempt {attempt}", end="\r")
        try:
            response = bedrock_runtime.converse(
                modelId=model_id,
                messages=messages,
                inferenceConfig={"temperature": 0}
            )
            answer = response['output']['message']['content'][0]["text"].strip()
            
            # Extract score and justification
            import re
            score_match = re.search(r'<score>(\d+)</score>', answer)
            justification_match = re.search(r'<justification>(.*?)</justification>', answer, re.DOTALL)
            
            if score_match:
                score = int(score_match.group(1))
                score = max(1, min(5, score))  # Clamp to 1-5 range
                justification = justification_match.group(1).strip() if justification_match else ""
                return score, justification
            else:
                # Fallback: try to extract first number
                numbers = re.findall(r'\d+', answer)
                if numbers:
                    score = max(1, min(5, int(numbers[0])))
                    return score, ""
                return 0, ""
                
        except (ClientError, Exception) as e:
            print(f"ERROR: Can't invoke '{model_id}'. Reason: {e}")
            if delay is not None:
                time.sleep(delay)  # Required: Exponential backoff retry strategy for API resilience
            else:
                return 0, ""

def video_quality_evaluation_pipeline(
        s3_video_uri,
        boto3_session=None,
        model_id="us.amazon.nova-premier-v1:0",
        temporal_consistency_flag = True,
        aesthetic_quality_flag = True,
        technical_quality_flag = True,
        motion_effects_flag = True
    ):
    """Complete video quality evaluation pipeline across multiple metrics.
    
    Evaluates a video across temporal consistency, aesthetic quality, and 
    technical quality. Saves results to S3.
    
    Args:
        s3_video_uri (str): S3 URI of video file to evaluate
        boto3_session: AWS boto3 session, creates default if None
        model_id (str): Bedrock model ID for evaluation
        
    Returns:
        dict: Quality scores for each metric
    """
    
    if boto3_session is None:
        boto3_session = boto3.Session()
    
    s3_client = boto3_session.client('s3')
    bucket = s3_video_uri.split('/')[2]
    video_key = '/'.join(s3_video_uri.split('/')[3:])
    
    print(f"Evaluating video quality for: {s3_video_uri}")
    
    # Define quality metrics and their prompts
    quality_metrics = {}
    if temporal_consistency_flag:
        quality_metrics["temporal_consistency"] = TEMPORAL_CONSISTENCY_PROMPT
    if aesthetic_quality_flag:
        quality_metrics["aesthetic_quality"] = AESTHETIC_QUALITY_PROMPT
    if technical_quality_flag:
        quality_metrics["technical_quality"] = TECHNICAL_QUALITY_PROMPT
    if motion_effects_flag:
        quality_metrics["motion_effects"] = MOTION_EFFECTS_PROMPT
    
    results = {}
    
    for metric_name, prompt in tqdm(quality_metrics.items(), desc="Evaluating quality metrics"):
        print(f"Evaluating {metric_name}...")
        
        score, justification = evaluate_video_quality_metric(
            boto3_session=boto3_session,
            s3_video_uri=s3_video_uri,
            metric_prompt=prompt,
            model_id=model_id
        )
        
        results[metric_name] = {
            "score": score,
            "justification": justification
        }
        print(f"{metric_name}: {score}/5")
    

    # Save results to S3
    video_filename = video_key.replace('.mp4', '')
    quality_key = f"{video_filename}_quality.json"
    
    s3_client.put_object(
        Bucket=bucket,
        Key=quality_key,
        Body=json.dumps(results, indent=2),
        ContentType='application/json'
    )
    
    print(f"Saved quality scores to s3://{bucket}/{quality_key}")
    
    return results

if __name__ == "__main__":
    # Example usage
    boto3_session = boto3.Session()
    
    result = video_quality_evaluation_pipeline(
        "s3://your-bucket/generated_videos/video.mp4",
        boto3_session=boto3_session
    )
    print(json.dumps(result, indent=2))