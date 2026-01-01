# re:Invent 2025 Workshop: Robo-Reviewer: Building AI Video Evaluators with Bedrock

This repository contains the code samples for the AWS re:Invent 2025 workshop that will let participants learn how to use [Amazon Nova Reel](https://aws.amazon.com/bedrock/nova/) with [Amazon Bedrock](https://aws.amazon.com/bedrock/) to generate videos from text prompts and perform advanced video evaluation for quality assessment. Participants will also learn how to extract and process video frames using semantic segmentation and various sampling techniques.

### Overview

[Amazon Bedrock](https://aws.amazon.com/bedrock/) is a fully managed service that offers a choice of high-performing Foundation Models (FMs) from leading AI companies accessible through a single API, along with a broad set of capabilities you need to build generative AI applications, simplifying development while maintaining privacy and security.

[Amazon Nova Reel](https://aws.amazon.com/bedrock/nova/) is a state-of-the-art video generation model that can create high-quality videos from text descriptions. It supports various video durations, resolutions, and camera movements, making it ideal for content creation and evaluation workflows.

This repository provides a complete toolkit for video generation, analysis, and evaluation, including:
- **Text-to-video generation** using Amazon Nova Reel with intelligent prompt generation
- **Advanced frame extraction** with semantic segmentation capabilities  
- **Multiple sampling strategies** including uniform, random, and semantic-based sampling
- **Video processing utilities** for analysis and evaluation workflows
- **Content alignment evaluation** with structured Q&A pairs across multiple focus areas
- **Video quality evaluation** with LLM-as-judge scoring across technical and aesthetic metrics
- **Automated video organization** with centralized S3 storage management

### To get started

1. Choose an AWS Account to use and make sure to create all resources in that Account.
2. Identify an AWS Region that has [Amazon Bedrock with Nova Reel and Titan Embeddings](https://docs.aws.amazon.com/bedrock/latest/userguide/models-regions.html) models (us-east-1 recommended).
3. In that Region, create a new or use an existing [Amazon S3 bucket](https://docs.aws.amazon.com/AmazonS3/latest/userguide/UsingBucket.html) for video storage.
4. Configure AWS credentials using `aws configure` or environment variables.
5. Create your virtual Python environment with [conda](https://anaconda.org/anaconda/conda) or [uv](https://github.com/astral-sh/uv) and activate it.
6. Clone this repository and go into it.
   ```bash
   git clone git@ssh.gitlab.aws.dev:genaiic-reusable-assets/engagement-artifacts/riv-robo-reviewer-video-evaluation.git
   cd riv-robo-reviewer-video-evaluation/
   ```
7. Install required dependencies:
   ```bash
   pip install -r requirements.txt
   ```
8. Open the Jupyter notebooks to get started:
   - `1_2_video_generation_pipeline_walkthrough.ipynb` - Video generation with Nova Reel
   - `2_1_video_frame_processing_walkthrough.ipynb` - Frame extraction and analysis
   - `3_1_content_alignment_through_qa.ipynb` - Content alignment evaluation through Q&A
   - `3_2_video_quality_evaluation_using_llm_judge.ipynb` - Video quality evaluation using LLM as judge
   - `4_end_to_end_robo_reviewer` - Full automated pipeline for video generation and evaluation

### Repository structure

This repository contains:

* [Video generation walkthrough notebook](1_2_video_generation_pipeline_walkthrough.ipynb) to learn text-to-video generation with Amazon Nova Reel.

* [Video frame processing walkthrough notebook](2_1_video_frame_processing_walkthrough.ipynb) to learn advanced frame extraction and semantic analysis.

* [Video evaluation walkthrough notebook](3_1_content_alignment_through_qa.ipynb) to learn content alignment evaluation through Q&A pairs.

* [Video quality evaluation walkthrough notebook](3_2_video_quality_evaluation_using_llm_judge.ipynb) to learn video quality assessment using LLM as judge.

* [Video generation and evaluation end-to-end notebook](4_end_to_end_robo_reviewer.ipynb) to learn the end-to-end video generation and evaluation pipeline.

* [Video generation utilities](utils/video_generation.py) with functions for prompt generation, video creation, and S3 management.

* [Video processing utilities](utils/video_processing.py) with frame extraction, semantic segmentation, and analysis tools.

* [Video evaluation utilities](utils/content_alignment.py) with content alignment evaluation functions and Q&A generation.

* [Video quality evaluation utilities](utils/quality_assessment.py) with quality assessment functions across multiple metrics.

* [S3 bucket configuration utilities](utils/config.py) for automatic CloudFormation and SSM parameter detection.

## Security

See [CONTRIBUTING](CONTRIBUTING.md#security-issue-notifications) for more information.

## License

This library is licensed under the MIT-0 License. See the LICENSE file.