"""
Copyright 2025 Amazon.com, Inc. or its affiliates. All Rights Reserved.

This is AWS Content subject to the terms of the Customer Agreement
----------------------------------------------------------------------
Package content:
    S3 bucket configuration utilities for automatic detection from CloudFormation
"""

import boto3
import json

def get_s3_bucket_from_cloudformation(session=None):
    """Get S3 bucket name from CloudFormation stack outputs"""
    if session is None:
        session = boto3.Session()
    
    try:
        # Try SSM parameter first
        ssm_client = session.client('ssm')
        try:
            response = ssm_client.get_parameter(Name='/robo-reviewer/s3-bucket-name')
            bucket_name = response['Parameter']['Value']
            print(f"Found S3 bucket from SSM parameter: {bucket_name}")
            return bucket_name
        except:
            pass
        
        # Skip S3 bucket pattern matching (requires ListAllMyBuckets permission)
        
        # Try CloudFormation as last resort
        try:
            cf_client = session.client('cloudformation')
            stacks = cf_client.list_stacks(StackStatusFilter=['CREATE_COMPLETE', 'UPDATE_COMPLETE'])
            
            for stack in stacks['StackSummaries']:
                try:
                    response = cf_client.describe_stacks(StackName=stack['StackName'])
                    outputs = response['Stacks'][0].get('Outputs', [])
                    
                    for output in outputs:
                        if output['OutputKey'] == 'S3BucketName':
                            bucket_name = output['OutputValue']
                            print(f"Found S3 bucket from CloudFormation stack '{stack['StackName']}': {bucket_name}")
                            return bucket_name
                except:
                    continue
        except Exception as cf_error:
            # Silently skip CloudFormation if no permissions
            if 'AccessDenied' not in str(cf_error):
                print(f"Warning: CloudFormation access failed: {cf_error}")
                
    except Exception as e:
        print(f"Warning: Could not get S3 bucket: {e}")
    
    return None

def get_s3_bucket(session=None):
    """Get S3 bucket name with CloudFormation detection and config fallback"""
    # Try to get S3 bucket from CloudFormation/SSM first
    bucket_name = get_s3_bucket_from_cloudformation(session)
    
    if not bucket_name:
        # Load configuration as fallback
        try:
            with open('config.json', 'r') as f:
                config = json.load(f)
            bucket_name = config['s3_bucket']
            
            # Check if it's an SSM parameter reference
            if bucket_name.startswith('{{resolve:ssm:'):
                # Extract parameter name from {{resolve:ssm:/parameter-name}}
                param_name = bucket_name.split(':')[2].rstrip('}}')
                try:
                    if session is None:
                        session = boto3.Session()
                    ssm_client = session.client('ssm')
                    response = ssm_client.get_parameter(Name=param_name)
                    bucket_name = response['Parameter']['Value']
                    print(f"Resolved SSM parameter {param_name}: {bucket_name}")
                except Exception as e:
                    print(f"Failed to resolve SSM parameter {param_name}: {e}")
                    return None
            else:
                print(f"Using S3 bucket from config: {bucket_name}")
        except Exception as e:
            print(f"Error loading config: {e}")
            return None
    
    return bucket_name

def discover_video_files(session, s3_bucket, video_prefix):
    """Discover available video files in S3"""
    s3_client = session.client('s3')
    try:
        response = s3_client.list_objects_v2(Bucket=s3_bucket, Prefix=video_prefix)
        if 'Contents' not in response:
            return []
        
        video_files = []
        for obj in response['Contents']:
            key = obj['Key']
            if key.endswith('.mp4'):
                # Check if corresponding prompt file exists
                prompt_key = key.replace('.mp4', '_prompt.txt')
                try:
                    s3_client.head_object(Bucket=s3_bucket, Key=prompt_key)
                    video_files.append(key.split('/')[-1])
                except:
                    pass
        return video_files
    except Exception as e:
        print(f"Error discovering videos: {e}")
        return []