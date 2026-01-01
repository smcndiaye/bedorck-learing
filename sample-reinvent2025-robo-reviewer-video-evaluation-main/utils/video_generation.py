"""
Copyright 2025 Amazon.com, Inc. or its affiliates. All Rights Reserved.

This is AWS Content subject to the terms of the Customer Agreement
----------------------------------------------------------------------
Package content:
    Video generation utilities using Amazon Nova Reel for text-to-video generation
"""

import time
import boto3
import json
import time
from botocore.exceptions import ClientError
import re

def get_delay():
    return 0.5

def start_invocation_t2v(
    bedrock_runtime,
    s3_bucket:str,
    text_prompts:list,
    duration_seconds:int = 6,
    fps:int = 24,
    dimension:str = "1280x720",
    seed:int = 42,
    model_id:str = "amazon.nova-reel-v1:1"
    ):
    """
    Start invocations to generate videos from text prompts.

    Parameters:
    -----------
    bedrock_runtime: boto3.client
        The Bedrock Runtime client.

    s3_bucket: str
        The S3 bucket where the video will be stored.

    text_prompt: list
        The text prompt(s) to generate video(s).

    duration_seconds: int
        The duration of the video in seconds. Default is 6.

    fps: int
        The frames per second of the video. Default is 24.

    dimension: str
        The dimension of the video. Default is "1280x720".

    seed: int
        The seed to use for the video generation. Default is 0.

    model_id: str
        The model ID to use for the video generation. Default is 
        "amazon.nova-reel-v1:1".

    Returns:
    --------
    str or list
        The invocation ARN(s).
    
    
    """
    
    # Handle both single prompt and list of prompts
    prompts = text_prompts
    
    invocation_arns = []

    if duration_seconds <= 6:

        for i, prompt in enumerate(prompts):
            skip_flag = False
            model_input = {
                "taskType": "TEXT_VIDEO",
                "textToVideoParams": {
                    "text": prompt
                },
                "videoGenerationConfig": {
                    "durationSeconds": duration_seconds,
                    "fps": fps,
                    "dimension": dimension,
                    "seed": seed + i,  # Increment seed for each prompt
                },
            }

            retry_delays = [32, 32, 32, 32, 32]
            for attempt, delay in enumerate(retry_delays + [None]):
                print(f"inovke Nova Reel - attempt {attempt}")
                try:
                    invocation = bedrock_runtime.start_async_invoke(
                        modelId = model_id,
                        modelInput=model_input,
                        outputDataConfig={
                            "s3OutputDataConfig": {
                                "s3Uri": "s3://" + s3_bucket
                            }
                        }
                    )
                    break
                except (ClientError, Exception) as e:
                    print(f"ERROR: Reason: {e}")
                    if delay is not None:
                        print(f"Retryinng in {delay} seconds.")
                        time.sleep(delay)
                    else:
                        skip_flag = True

            if not skip_flag:
                invocation_arns.append(invocation["invocationArn"])
    else:

        for i, prompt in enumerate(prompts):
            skip_flag = False
            model_input = {
                "taskType": "MULTI_SHOT_AUTOMATED",
                "multiShotAutomatedParams": {
                    "text": prompt
                },
                "videoGenerationConfig": {
                    "durationSeconds": duration_seconds,
                    "fps": fps,
                    "dimension": dimension,
                    "seed": seed + i,  # Increment seed for each prompt
                },
            }

            retry_delays = [32, 32, 32, 32, 32]
            for attempt, delay in enumerate(retry_delays + [None]):
                print(f"inovke Nova Reel - attempt {attempt}")
                try:
                    invocation = bedrock_runtime.start_async_invoke(
                        modelId = model_id,
                        modelInput=model_input,
                        outputDataConfig={
                            "s3OutputDataConfig": {
                                "s3Uri": "s3://" + s3_bucket
                            }
                        }
                    )
                    break
                except (ClientError, Exception) as e:
                    print(f"ERROR: Reason: {e}")
                    if delay is not None:
                        print(f"Retryinng in {delay} seconds.")
                        time.sleep(delay)
                    else:
                        skip_flag = True

            if not skip_flag:
                invocation_arns.append(invocation["invocationArn"])
    
    return invocation_arns

def invocation_status_check(
        bedrock_runtime,
        invocation_arns
    ):
    """
    Check the status of invocation(s).

    Parameters:
    -----------
    bedrock_runtime: boto3.client
        The Bedrock Runtime client.

    invocation_arn: list
        The invocation ARN(s).

    Returns:
    --------
    str or list
        The S3 URI(s) of the generated video(s). If generation failed,
        return None for that video.
    
    """
    
    # Handle both single ARN and list of ARNs
    arns = invocation_arns
    
    # Track status for each ARN
    arn_status = {arn: "InProgress" for arn in arns}
    results = {}
    update_count = 0
    arn_in_progress_flag = {arn: False for arn in arns}
    
    while any(status not in ["Completed", "Failed"] for status in arn_status.values()):
        # INTENTIONAL: Sleep required for AWS Bedrock API rate limiting to prevent throttling
        # This prevents overwhelming the AWS service with too many status check requests
        time.sleep(get_delay())
        update_count += 1
        
        for arn in arns:
            if arn_status[arn] in ["Completed", "Failed"]:
                continue
            try:    
                invocation = bedrock_runtime.get_async_invoke(invocationArn=arn)
            except:
                continue
                
            status = invocation["status"]
            arn_status[arn] = status
            
            if status == "InProgress":
                if update_count > 25000 and update_count % 5 == 0:
                    start_time = invocation["submitTime"]
                    if not arn_in_progress_flag[arn]:
                        print(f"Job {arn} is still in progress. Started at: {start_time}")
                        arn_in_progress_flag[arn] = True
                elif update_count % 5 == 0:
                    print(f"Jobs in progress: {sum(1 for s in arn_status.values() if s == 'InProgress')}   " + "." * (update_count%30//5) + " " * (5- update_count%30//5), end = "\r")
                    
            elif status == "Failed":
                failure_message = invocation["failureMessage"]
                print(f"Job {arn} failed. Failure message: {failure_message}")
                results[arn] = None
            
            elif status == "Completed":
                bucket_uri = invocation["outputDataConfig"]["s3OutputDataConfig"]["s3Uri"]
                video_uri = bucket_uri + "/output.mp4"
                results[arn] = video_uri
    
    # Return results in same order as input
    result_list = [results.get(arn) for arn in arns]
    return result_list

def generate_prompts(
    bedrock_runtime,
    user_request,
    num_prompts = 2,
    video_duration = 6,
    model_id="us.amazon.nova-premier-v1:0"
    ):
    """
    Generate n video prompts for Amazon Nova Reel based on a given scenario.
    
    Args:
        bedrock_runtime: AWS Bedrock runtime client
        user_request (str): The video generation request from the user about what types of video to be generated.
        num_prompts: the number of prompts to be generated.
        video_duration: the video duration in seconds
        model_id (str): Model ID for prompt generation
        
    Returns:
        list: List of generated video prompts
    """
    long_video_flag = False
    num_scenes = 1
    if video_duration > 6:
        long_video_flag = True
        num_scenes = video_duration // 6
    print(f"num_scenes: {num_scenes}")
    
    # Camera motion dictionary
    camera_motions = {
        'aerial shot': ['Aerial shot', 'Drone shot', 'FPV drone shot', 'First person view drone shot'],
        'arc shot': ['Arc shot', '360 degree shot', '360 tracking shot', 'Orbit shot'],
        'clockwise rotation': ['Clockwise rotating shot', 'Camera rotates clockwise', 'Camera rolls clockwise'],
        'counterclockwise rotation': ['Counterclockwise rotating shot', 'Camera rotates counterclockwise', 'Camera rolls counterclockwise'],
        'dolly in': ['Dolly in shot', 'Dolly in', 'Camera moves forward', 'Camera moving forward'],
        'dolly out': ['Dolly out shot', 'Dolly out', 'Camera moves backward', 'Camera moving backward'],
        'pan left': ['Pan left shot', 'Pan left', 'Camera pans left', 'Camera moves left'],
        'pan right': ['Pan right shot', 'Pan right', 'Camera pans right', 'Camera moves right'],
        'whip pan': ['Whip pan left', 'Whip pan right'],
        'pedestal down': ['Pedestal down shot', 'Pedestal down', 'Camera moves down', 'Camera moving down'],
        'pedestal up': ['Pedestal up shot', 'Pedestal up', 'Camera moves up', 'Camera moving up'],
        'static shot': ['Static shot', 'Fixed shot'],
        'tilt down': ['Tilt down shot', 'Tilt down', 'Camera tilts down', 'Camera moving down'],
        'tilt up': ['Tilt up shot', 'Tilt up', 'Camera tilts up', 'Camera moving up'],
        'whip tilt': ['Whip tilt up', 'Whip tilt down'],
        'track left': ['Track left', 'Truck left', 'Camera tracks left', 'Camera moving to the left'],
        'track right': ['Track right', 'Truck right', 'Camera tracks right', 'Camera moving to the right'],
        'zoom in': ['Zoom in', 'Camera zooms in', 'Camera moves forward'],
        'zoom out': ['Zoom out', 'Camera zooms out', 'Camera moves backward'],
        'whip zoom': ['Whip zoom in', 'Whip zoom out'],
        'dolly zoom': ['Dolly zoom', 'Dolly zoom shot', 'Dolly zoom effect'],
        'following shot': ['Following shot']
    }
    
    # Refined prompt template for Claude
    prompt_template_short = f"""Generate {num_prompts} different prompts for the Amazon Nova Reel video generation model based on the given request. Nova Reel generates video clips from text descriptions.

<scenario>
{user_request}
</scenario>

<guidelines>
Video generation prompts should be descriptive captions, not commands. Include details about subject, action, environment, lighting, style, and camera motion.

Requirements:
- Prompts must be no longer than 512 characters.
- Place camera movement descriptions at start or end of prompt
- Avoid negation words (no, not, without)
- Write as scene descriptions, not instructions

Camera motions available: {json.dumps(camera_motions, indent=2)}
</guidelines>

<examples>
<example_1>
Cinematic dolly shot of a juicy cheeseburger with melting cheese, fries, and a condensation-covered cola on a worn diner table. Natural lighting, visible steam and droplets. 4k, photorealistic, shallow depth of field.
</example_1>

<example_2>
Arc shot on a salad with dressing, olives and other vegetables; 4k; Cinematic.
</example_2>

<example_3>
First person view of a motorcycle riding through the forest road.
</example_3>

<example_4>
Closeup of a large seashell in the sand. Gentle waves flow around the shell. Camera zoom in.
</example_4>

<example_5>
Clothes hanging on a thread to dry, windy; sunny day; 4k; Cinematic; highest quality;
</example_5>

<example_6>
Slow cam of a man middle age; 4k; Cinematic; in a sunny day; peaceful; highest quality; dolly in;
</example_6>

<example_7>
A mushroom drinking a cup of coffee while sitting on a couch, photorealistic.
</example_7>
</examples>

Generate {num_prompts} unique prompts for the scenario. Each prompt should:
1. Be clear and concise (under 512 characters)
2. Include appropriate camera motion at beginning or end
3. Describe the scene naturally (avoid starting with verbs)
4. Provide sufficient context for video generation

Return only a Python list of the {num_prompts} prompts, no additional text.
"""

    prompt_template_long = f"""Generate {num_prompts} different prompts for the Amazon Nova Reel video generation model based on the given request. Nova Reel generates video clips from text descriptions.

<scenario>
{user_request}
</scenario>

<guidelines>
Video generation prompts should be descriptive captions, not commands. Include details about subject, action, environment, lighting, style, and camera motion.

Requirements:
- Each prompt must be no longer than 4000 characters.
- Each prompt must contain {num_scenes + 1} sentences.
- Each sentence except the last one in the prompt represented a scene in the generated video. The last sentence contains the technical specifications, such as 4k, photorealistic, Cinematic.
- Place camera movement descriptions at start or end of the sentence. But this is optional.
- Avoid negation words (no, not, without)
- Write as scene descriptions, not instructions

Camera motions available: {json.dumps(camera_motions, indent=2)}
</guidelines>

<examples>
Norwegian fjord with still water reflecting mountains in perfect symmetry. Uninhabited wilderness of Giant sequoia forest with sunlight filtering between massive trunks. Sahara desert sand dunes with perfect ripple patterns. Alpine lake with crystal clear water and mountain reflection. Ancient redwood tree with detailed bark texture. Arctic ice cave with blue ice walls and ceiling. Bioluminescent plankton on beach shore at night. Bolivian salt flats with perfect sky reflection. Bamboo forest with tall stalks in filtered light. Cherry blossom grove against blue sky. Lavender field with purple rows to horizon. Autumn forest with red and gold leaves. Tropical coral reef with fish and colorful coral. Antelope Canyon with light beams through narrow passages. Banff lake with turquoise water and mountain backdrop. Joshua Tree desert at sunset with silhouetted trees. Iceland moss- covered lava field. Amazon lily pads with perfect symmetry. Hawaiian volcanic landscape with lava rock. New Zealand glowworm cave with blue ceiling lights. 8K nature photography, professional landscape lighting, no movement transitions, perfect exposure for each environment, natural color grading.

Explosion of colored powder against black background. Start with slow-motion closeup of single purple powder burst. Dolly out revealing multiple powder clouds in vibrant hues colliding mid-air. Track across spectrum of colors mixing: magenta, yellow, cyan, orange. Zoom in on particles illuminated by sunbeams. Arc shot capturing complete color field. 4K, festival celebration, high-contrast lighting
</examples>

Generate {num_prompts} unique prompts for the scenario. Each prompt should:
1. Be clear and concise (under 4000 characters)
2. Include appropriate camera motion at beginning or end
3. Describe the scene naturally (avoid starting with verbs)
4. Provide sufficient context for video generation

Return only a Python list of the {num_prompts} prompts, no additional text.
"""

    if long_video_flag:
        input_prompt = prompt_template_long
    else:
        input_prompt = prompt_template_short


    retry_delays = [1, 2, 4, 8, 16]
    messages = [
    {"role": "user",
        "content": [{"text": input_prompt}]}
    ]
    for attempt, delay in enumerate(retry_delays + [None]):
        print(f"inovke LLM - attempt {attempt}")
        try:
            response = bedrock_runtime.converse(
                modelId=model_id,
                messages=messages,
                inferenceConfig={"temperature": 0},
            )
            generated_text = response['output']['message']['content'][0]["text"]
            break
        except (ClientError, Exception) as e:
            print(f"ERROR: Can't invoke '{model_id}'. Reason: {e}")
            if delay is not None:
                time.sleep(delay)  # Required: Exponential backoff retry strategy for API resilience
            else:
                return ""



    
    # Extract the Python list from the response
    try:
        # Find the list in the response text
        start_idx = generated_text.find('[')
        end_idx = generated_text.rfind(']') + 1
        list_str = generated_text[start_idx:end_idx]
        return_prompts = json.loads(list_str)  # Validate JSON
        return return_prompts[0:num_prompts]
    # raise error if failed to generate prompts
    except Exception as e:
        raise ValueError(f"Error parsing generated text: {e}" + "Failed to generate prompts. Please try again.")

def move_videos_to_centralized_location(s3_client, bucket_name, video_uris, prompts):
    """
    Move videos from session-specific locations to centralized generated_videos/ prefix.
    
    Parameters:
    -----------
    s3_client: boto3.client
        The S3 client.
    bucket_name: str
        The S3 bucket name.
    video_uris: list
        List of S3 URIs of generated videos.
    prompts: list
        List of prompts used to generate the videos.
    
    Returns:
    --------
    list
        List of new S3 URIs in centralized location.
    """
    centralized_uris = []
    
    for i, uri in enumerate(video_uris):
        if uri is None:
            centralized_uris.append(None)
            continue
            
        # Extract session_id from URI like s3://bucket/session_id/output.mp4
        match = re.search(r's3://[^/]+/([^/]+)/output\.mp4', uri)
        if not match:
            print(f"Warning: Could not extract session_id from URI: {uri}")
            centralized_uris.append(uri)
            continue
            
        session_id = match.group(1)
        source_key = f"{session_id}/output.mp4"
        dest_key = f"generated_videos/{session_id}.mp4"
        prompt_key = f"generated_videos/{session_id}_prompt.txt"
        
        try:
            # Copy video to new location
            s3_client.copy_object(
                CopySource={'Bucket': bucket_name, 'Key': source_key},
                Bucket=bucket_name,
                Key=dest_key
            )
            
            # Save prompt as text file
            s3_client.put_object(
                Bucket=bucket_name,
                Key=prompt_key,
                Body=prompts[i].encode('utf-8'),
                ContentType='text/plain'
            )
            
            # Delete original
            s3_client.delete_object(Bucket=bucket_name, Key=source_key)
            
            new_uri = f"s3://{bucket_name}/{dest_key}"
            centralized_uris.append(new_uri)
            print(f"Moved video: {uri} -> {new_uri}")
            print(f"Saved prompt: s3://{bucket_name}/{prompt_key}")
            
        except Exception as e:
            print(f"Error moving video {uri}: {e}")
            centralized_uris.append(uri)
    
    return centralized_uris

def video_generation_pipeline(
        boto3_session,
        s3_bucket,
        user_request,
        prompt_optimization_flag = False,
        num_videos = 2,
        duration_seconds = 6,
        fps = 24,
        dimension = "1280x720",
        seed = 42
    ):
    """
    wrapper function to generate video from text prompt.

    Parameters:
    -----------
    bedrock_runtime: boto3.client
        The Bedrock Runtime client.

    s3_bucket: str
        The S3 bucket where the video will be stored.

    text_prompt: list
        The text prompt(s) to generate video(s).


    duration_seconds: int
        The duration of the video in seconds. Default is 6.

    fps: int
        The frames per second of the video. Default is 24.

    dimension: str
        The dimension of the video. Default is "1280x720".

    seed: int
        The seed to use for the video generation. Default is 42.

    model_id: str
        The model ID to use for the video generation. Default is 
        "amazon.nova-reel-v1:0".

    Returns:
    --------
    str
        The S3 URI of the generated video.

    """
    bedrock_runtime= boto3_session.client("bedrock-runtime")


    if duration_seconds < 6:
        duration_seconds = 6
    elif duration_seconds < 12 and duration_seconds > 6:
        duration_seconds = 12
    elif duration_seconds > 120:
        duration_seconds = 120
    else:
        duration_seconds = (duration_seconds // 6) * 6

    if prompt_optimization_flag:
        prompts = generate_prompts(
            bedrock_runtime,
            user_request,
            num_prompts = num_videos,
            video_duration = duration_seconds,
            model_id="us.amazon.nova-premier-v1:0"
        )

        print(f"\nGenerated {len(prompts)} prompts:")
        for i, prompt in enumerate(prompts, 1):
            print(f"{i}. {prompt}")
        print("\n")
    else:
        prompts = [user_request]

    invocation_arns = start_invocation_t2v(bedrock_runtime,
                                      s3_bucket,
                                      prompts,
                                      duration_seconds = duration_seconds,
                                      fps = fps,
                                      dimension = dimension,
                                      seed = seed)
    
    video_uris = invocation_status_check(bedrock_runtime,
                                        invocation_arns)
    
    # Move videos to centralized location
    s3_client = boto3_session.client('s3')
    centralized_uris = move_videos_to_centralized_location(s3_client, s3_bucket, video_uris, prompts)
    
    return centralized_uris

