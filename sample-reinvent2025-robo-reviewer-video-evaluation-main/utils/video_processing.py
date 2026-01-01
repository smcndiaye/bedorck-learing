"""
Copyright 2025 Amazon.com, Inc. or its affiliates. All Rights Reserved.

This is AWS Content subject to the terms of the Customer Agreement
----------------------------------------------------------------------
Package content:
    Video processing utilities for frame extraction and semantic segmentation
"""

import random
import base64
import time
import json
import os
import tempfile
from urllib.parse import urlparse
from concurrent.futures import ThreadPoolExecutor
from io import BytesIO

import numpy as np
import cv2

from PIL import Image
from botocore.exceptions import ClientError
import matplotlib.pyplot as plt

# ---------------------------------------------------------------------------- #
#                          Utils Tools                                         #
# ---------------------------------------------------------------------------- #

def visualize_frames(all_frames, method_name="", max_frames_per_row=8):
    """
    Visualize extracted frames in a grid layout.
    
    Args:
        frame_results (dict): Results from process_video function
        method_name (str): Name to display (e.g., "Middle Sampling")
        max_frames_per_row (int): Maximum frames per row in display
    """

    # Calculate grid dimensions
    num_frames = len(all_frames)
    cols = min(max_frames_per_row, num_frames)
    rows = (num_frames + cols - 1) // cols
    
    # Create subplot grid
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 3, rows * 2))
    if rows == 1 and cols == 1:
        axes = [axes]
    else:
        axes = axes.flatten()
    
    # Display each frame
    for i, frame in enumerate(all_frames):
        # Convert base64 to PIL Image
        img = Image.fromarray(frame)
        
        # Display image
        axes[i].imshow(img)
        #axes[i].set_title(f'Frame {i+1}')
        axes[i].axis('off')
    
    # Hide unused subplots
    for i in range(num_frames, len(axes)):
        axes[i].axis('off')
    
    plt.suptitle(f'{method_name} - {num_frames} Frames', fontsize=14)
    plt.tight_layout()
    plt.show()

    return

def frames_to_video(frames, s3_uri, fps=24, boto3_session=None):
    """ Convert a list of frames back to an MP4 video file and save to S3.
    Args:
        frames (list): List of frame arrays (numpy arrays)
        s3_uri (str): S3 URI to save the output video (e.g., 's3://bucket/path/video.mp4')
        fps (int): Frames per second for the output video (default: 24)
        boto3_session: AWS boto3 session for S3 upload
    Returns:
        bool: True if successful, False otherwise
    """
    if not frames:
        print("No frames to convert")
        return False
    
    # Parse S3 URI
    parsed = urlparse(s3_uri)
    bucket = parsed.netloc
    key = parsed.path.lstrip('/')
    
    # Create temporary file for video creation
    temp_fd, temp_file_path = tempfile.mkstemp(suffix='.mp4')
    os.close(temp_fd)  # Close file descriptor immediately
    
    try:
        # Get dimensions from the first frame
        height, width, _ = frames[0].shape
        
        # Define the codec and create VideoWriter object
        fourcc = cv2.VideoWriter_fourcc(*'avc1')
        out = cv2.VideoWriter(temp_file_path, fourcc, fps, (width, height))
        
        # Write frames to the video file
        for frame in frames:
            out.write(frame)
        
        # Release the VideoWriter
        out.release()
        
        # Upload to S3
        s3_client = boto3_session.client('s3')
        s3_client.upload_file(temp_file_path, bucket, key)
        
        print(f"Video saved to {s3_uri}")
        return True
        
    except Exception as e:
        print(f"Error saving video to S3: {e}")
        return False
    finally:
        os.unlink(temp_file_path)
    return False

# ---------------------------------------------------------------------------- #
#                          Embedding Geneartion                                #
# ---------------------------------------------------------------------------- #
def get_titan_embedding_image(
        boto3_session, 
        image_base64
    ):
    """
    Generate image embedding using Amazon Titan Embed Image model.
    
    Args:
        boto3_session: AWS boto3 session for API calls
        image_base64 (str): Base64 encoded image string
        
    Returns:
        np.ndarray: 1024-dimensional embedding vector reshaped to (1, 1024)
        None: If all retry attempts fail
    """
    # Initialize Bedrock runtime client
    bedrock_client = boto3_session.client(service_name='bedrock-runtime')
    
    # Create request body for Titan embedding model
    body = json.dumps({
        "inputImage": image_base64,
        "embeddingConfig": {
            "outputEmbeddingLength": 1024
        }
    })

    # Exponential backoff retry strategy
    retry_delays = [1, 2, 4, 8, 16]

    for attempt, delay in enumerate(retry_delays + [None]):
        print(f"inovke Embedding - attempt {attempt}", end = "\r")
        try:
            # Invoke Titan embedding model
            response = bedrock_client.invoke_model(
                body=body,
                modelId="amazon.titan-embed-image-v1",
                accept = "application/json",
                contentType = "application/json"
            )

            # Extract embedding from response
            temp = json.loads(response['body'].read())['embedding']

            return np.array(temp).reshape(1, -1)
        except (ClientError, Exception) as e:
            print(f"ERROR: Can't invoke amazon.titan-embed-image-v1. Reason: {e}")
            if delay is not None:
                time.sleep(delay)  # Wait before retry
            else:
                return None  # All retries exhausted
    return None

def get_image_embeddings_parallel(
        boto3_session, 
        image_base64_list, 
        max_workers=10
    ):
    """
    Generate embeddings for multiple images in parallel using ThreadPoolExecutor.
    
    Args:
        boto3_session: AWS boto3 session for API calls
        image_base64_list (list): List of base64 encoded image strings
        max_workers (int): Maximum number of parallel threads (default: 10)
        
    Returns:
        np.ndarray: Stacked embeddings array of shape (num_images, 1024)
    """
    # Process images in parallel using thread pool
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        embeddings = list(
            executor.map(
                lambda image_base64: get_titan_embedding_image(
                    boto3_session, 
                    image_base64
                ), 
                image_base64_list
            )
        )

    # Stack all embeddings into single array
    embeddings = np.vstack(embeddings)
    
    return embeddings.astype(np.float32)


# ---------------------------------------------------------------------------- #
#                          Frame Sampling Function                             #
# ---------------------------------------------------------------------------- #
def cosine_sim(a, b):
    """
    Calculate cosine similarity between two vectors.
    
    Args:
        a (np.ndarray): First vector
        b (np.ndarray): Second vector
        
    Returns:
        float: Cosine similarity value between -1 and 1
    """
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

def get_frame_indices(
        num_frames, 
        vlen, 
        sample='rand',
        embedd_array=None,
        threshold=0.8
    ):
    """
    Sample frame indices from video using different strategies.
    
    Args:
        num_frames (int): Number of frames to sample
        vlen (int): Total video length in frames
        sample (str): Sampling method - 'rand', 'uniform', or 'semantic'
        embedd_array (np.ndarray, optional): Frame embeddings for semantic sampling
        threshold (float): Cosine similarity threshold for semantic segmentation (default: 0.8)
        
    Returns:
        dict: Dictionary with segment indices as keys and frame indices as values
              For non-semantic modes: {1: [frame_indices]}
              For semantic mode: {0: [frames], 1: [frames], ...}
    """
    if sample == "semantic":
        if embedd_array is None:
            raise ValueError("embedd_array is required for semantic sampling")
        
        # Calculate cosine similarities between consecutive frames
        similarities = []
        for i in range(len(embedd_array) - 1):
            sim = cosine_sim(embedd_array[i], embedd_array[i + 1])
            similarities.append(sim)
        
        # Find segment boundaries where similarity drops below threshold
        segments = []
        start_idx = 0
        
        for i, sim in enumerate(similarities):
            if sim < threshold:  # Scene change detected
                segments.append((start_idx, i + 1))
                start_idx = i + 1
        
        # Add the final segment
        if start_idx < len(embedd_array):
            segments.append((start_idx, len(embedd_array)))
        
        # Uniformly sample frames within each detected segment
        result = {}
        for seg_idx, (start, end) in enumerate(segments):
            seg_len = end - start
            # Distribute frames evenly across segments
            seg_samples = max(1, min(num_frames // len(segments), seg_len))
            
            # Create uniform intervals within segment
            intervals = np.linspace(start, end, seg_samples + 1).astype(int)
            # Select middle frame from each interval
            seg_indices = [(intervals[i] + intervals[i + 1] - 1) // 2 for i in range(len(intervals) - 1)]
            
            result[str(seg_idx)] = seg_indices
        
        return result
    
    elif sample in ["rand", "uniform"]:  # Traditional uniform sampling
        acc_samples = min(num_frames, vlen)
        # Split video into equal intervals for sampling
        intervals = np.linspace(
            start=0, 
            stop=vlen, 
            num=acc_samples + 1
        ).astype(int)

        # Create frame ranges for each interval
        ranges = []
        for idx, interv in enumerate(intervals[:-1]):
            ranges.append((interv, intervals[idx + 1] - 1))

        if sample == 'rand':
            # Randomly select one frame from each interval
            try:
                frame_indices = [random.choice(range(x[0], x[1])) for x in ranges]
            except:
                # Fallback: random permutation if ranges are invalid
                frame_indices = np.random.permutation(vlen)[:acc_samples]
                frame_indices.sort()
                frame_indices = list(frame_indices)

        elif sample == 'uniform':
            # Select uniform frame from each interval
            frame_indices = [(x[0] + x[1]) // 2 for x in ranges]

        else:
            raise NotImplementedError

        # Pad with last frame if needed
        if len(frame_indices) < num_frames:
            padded_frame_indices = [frame_indices[-1]] * num_frames
            padded_frame_indices[:len(frame_indices)] = frame_indices
            frame_indices = padded_frame_indices

        # Return as single segment for consistency
        return {"0": frame_indices}

    else:
        raise ValueError
    

# ---------------------------------------------------------------------------- #
#                          Video Load and Process                              #
# ---------------------------------------------------------------------------- #
def process_video(
        boto3_session,
        video_path,
        num_frames = 16,
        threshold = 0.9
    ):
    """
    Load video, extract frames, and return selected frames as base64 images organized by segments.
    
    Args:
        boto3_session: AWS boto3 session for embedding generation (required for semantic sampling)
        video_path (str): Path to video file (must be .mp4)
        num_frames (int, optional): Number of frames to sample
        sample (str): Sampling method - 'rand', 'middle', or 'semantic' (default: 'middle')
        threshold (float): Cosine similarity threshold for semantic segmentation (default: 0.8)
        
    Returns:
        dict: Dictionary with segment indices as keys and lists of base64 image strings as values
              Example: {0: ['base64_img1', 'base64_img2'], 1: ['base64_img3']}
              
    Raises:
        NotImplementedError: If video format is not .mp4
    """
    # Load video file (currently supports only MP4)
    if video_path.endswith('.mp4'):
        #decord.bridge.set_bridge('native')
        #video_reader = VideoReader(video_path, num_threads=1)
        cap = cv2.VideoCapture(video_path)
        frame_indices_len = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        # Extract all frames from video
        frames = []
        for i in range(frame_indices_len):
            cap.set(cv2.CAP_PROP_POS_FRAMES, i)
            ret, frame = cap.read()
            if not ret:
                continue
            frames.append(frame)
    else:
        raise NotImplementedError("Only MP4 format is currently supported")
    
    if len(frames) < 24*7:
        return frames, {}
    
    # Convert all frames to base64 format for processing
    base64_frames = []
    for frame in frames:
        # Convert numpy array to PIL Image
        pil_img = Image.fromarray(frame)
        
        # Save image to memory buffer as PNG
        img_buffer = BytesIO()
        pil_img.save(img_buffer, format="PNG")
        img_buffer.seek(0)
        
        # Encode as base64 string
        img_str = base64.b64encode(img_buffer.read()).decode('utf-8')
        base64_frames.append(img_str)
    
    # Generate frame indices based on sampling method

    # Generate embeddings for all frames (required for semantic segmentation)
    embedd_array = get_image_embeddings_parallel(
        boto3_session=boto3_session,
        image_base64_list=base64_frames
    )

    # Perform semantic segmentation and sampling
    semantic_frame_indices = get_frame_indices(
            num_frames, 
            frame_indices_len, 
            sample = "semantic",
            embedd_array = embedd_array,
            threshold = threshold
        )
    # Use traditional sampling methods (random or middle)
    #uniform_frame_indices = get_frame_indices(
    #        num_frames,
    #        frame_indices_len,
    #        sample="uniform"
    #    )
    
    #rand_frame_indices = get_frame_indices(
    #        num_frames,
    #        frame_indices_len,
    #        sample="rand"
    #    )
    
    
    return frames, semantic_frame_indices#uniform_frame_indices, rand_frame_indices




def process_videos_pipeline(
        boto3_session,
        s3_uris,
        num_frames=16,
        threshold=0.9
    ):
    """
    Download videos from S3 URIs, process them, and upload results back to S3.
    
    Args:
        boto3_session: AWS boto3 session
        s3_uris (list): List of S3 URIs to video files
        num_frames (int): Number of frames to sample (default: 16)
        sample (str): Sampling method (default: "middle")
        threshold (float): Cosine similarity threshold (default: 0.8)
        
    Returns:
        dict: Results mapping original URI to output S3 URI
    """
    s3_client = boto3_session.client('s3')
    
    for s3_uri in s3_uris:
        if not s3_uri.endswith('.mp4'):
            continue
            
        # Parse S3 URI
        parsed = urlparse(s3_uri)
        bucket = parsed.netloc
        key = parsed.path.lstrip('/')
        video_name = os.path.splitext(os.path.basename(key))[0]
        
        # Download video to temp file
        temp_fd, temp_file_path = tempfile.mkstemp(suffix='.mp4')
        os.close(temp_fd)  # Close file descriptor immediately
        
        try:
            s3_client.download_file(bucket, key, temp_file_path)
            
            # Process video
            frames, semantic_frame_indices = process_video(
                boto3_session=boto3_session,
                video_path=temp_file_path,
                num_frames=num_frames,
                threshold=threshold
            )

            for segment_index, semantic_frame_indices in semantic_frame_indices.items():
                full_frames = [frames[idx] for idx in range(semantic_frame_indices[0], semantic_frame_indices[-1]+1)]
                output_full_key = f"{os.path.dirname(key)}/{video_name}/semantic/{segment_index}_video.mp4"

                frames_to_video(
                    full_frames,
                    f"s3://{bucket}/{output_full_key}",
                    fps=24,
                    boto3_session=boto3_session
                )
                
        except Exception as e:
            print(f"Error processing {s3_uri}: {e}")
        finally:
            os.unlink(temp_file_path)
    
    return
