import boto3
import json

prompt_data = """

Act as Sr manager and provide a detailed explanation of the following:
how to manage a team of 10 people, how to handle a difficult team member, how to handle a difficult situation, how to handle a difficult client, how to handle a difficult project
"""

bedrock = boto3.client(service_name='bedrock-runtime')

#prompt = "Hello"
claude_prompt = f"\n\nHuman:{prompt_data}\n\nAssistant:"
body = json.dumps({
                "prompt": claude_prompt,
                "temperature": 0.5,
                "top_p": 1,
                "top_k": 250,
                "max_tokens_to_sample": 200,
                "stop_sequences": ["\n\nHuman:"]
                })


model_id = "anthropic.claude-v2:1"

response = bedrock.invoke_model(
    body=body,
    modelId = model_id,
    accept= "*/*",
    contentType="application/json" )
response_body = json.loads(response.get("body").read())
print(response_body['completion'])
