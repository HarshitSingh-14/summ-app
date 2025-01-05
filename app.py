import boto3
import streamlit as st
import os
import json
import requests
from bs4 import BeautifulSoup
from dotenv import load_dotenv
from urllib.parse import urlparse
from uuid import uuid4
from io import BytesIO

# Load environment variables from .env file
load_dotenv()

# Retrieve AWS configurations from environment variables
aws_region = os.getenv("AWS_REGION")
s3_bucket_name = os.getenv("S3_BUCKET_NAME")

# Validate essential environment variables
if not aws_region:
    st.error("AWS_REGION environment variable is not set.")
    st.stop()

if not s3_bucket_name:
    st.error("S3_BUCKET_NAME environment variable is not set.")
    st.stop()

# Initialize AWS Bedrock and Polly clients
try:
    bedrock_client = boto3.client('bedrock-runtime', region_name=aws_region)
except Exception as e:
    st.error(f"Error initializing Bedrock client: {e}")
    st.stop()

try:
    polly_client = boto3.client('polly', region_name=aws_region)
except Exception as e:
    st.error(f"Error initializing Polly client: {e}")
    st.stop()

# Initialize AWS S3 client
try:
    s3_client = boto3.client('s3', region_name=aws_region)
except Exception as e:
    st.error(f"Error initializing S3 client: {e}")
    st.stop()

# Initialize LangChain Bedrock LLM
from langchain_community.embeddings import BedrockEmbeddings
from langchain.llms.bedrock import Bedrock

bedrock_embeddings = BedrockEmbeddings(model_id="amazon.titan-embed-text-v1", client=bedrock_client)

llm = Bedrock(
    model_id="anthropic.claude-v2:1",
    client=bedrock_client,
    model_kwargs={
        'max_tokens_to_sample': 5000,  # Adjust as needed
        'temperature': 0.5  # Adjust as needed
    }
)

# --- Helper Functions ---

def extract_text_from_url(url):
    """
    Extracts and returns the text content from a given URL.
    """
    try:
        response = requests.get(url)
        response.raise_for_status()
    except requests.exceptions.RequestException as e:
        return None, f"Request error: {e}"

    soup = BeautifulSoup(response.text, 'html.parser')
    body_text = soup.get_text(separator=' ', strip=True)
    return body_text, None

def summarize_text_bedrock(text, max_words=50):
    """
    Summarizes the given text using AWS Bedrock's Claude-3 Sonnet model via LangChain's Bedrock LLM.
    """
    # Prepare the prompt with the required structure
    prompt = (
        f"Human: Summarize the following text in {max_words} words with the first line as the topic of the summary:\n\n{text}\n\nAssistant: "
    )
    
    try:
        summary = llm(prompt)
        if not summary:
            return None, "No summary returned from Bedrock."
        return summary.strip(), None
    except Exception as e:
        return None, f"Error generating summary: {e}"

def split_text_into_chunks(text, max_chars=3000):
    """
    Splits the text into chunks each with a maximum of `max_chars` characters.
    Ensures that sentences are not broken in the middle.
    """
    import textwrap
    sentences = text.split('. ')
    chunks = []
    current_chunk = ""
    
    for sentence in sentences:
        if len(current_chunk) + len(sentence) + 1 <= max_chars:
            current_chunk += sentence + '. '
        else:
            chunks.append(current_chunk.strip())
            current_chunk = sentence + '. '
    
    if current_chunk:
        chunks.append(current_chunk.strip())
    
    return chunks

def text_to_speech_polly(text, voice_id="Joanna"):
    """
    Converts the given text to speech using AWS Polly and returns the audio bytes.
    Handles text longer than Polly's limit by splitting into chunks.
    """
    max_polly_chars = 3000  # AWS Polly's limit per request
    chunks = split_text_into_chunks(text, max_chars=max_polly_chars)
    audio_stream = BytesIO()
    
    for idx, chunk in enumerate(chunks):
        try:
            # response = polly_client.synthesize_speech(
            #     Text=chunk,
            #     OutputFormat='mp3',
            #     VoiceId=voice_id
            # )
            response = polly_client.synthesize_speech(
                Text=chunk,
                OutputFormat='mp3',
                VoiceId= voice_id,
                Engine='neural'  
            )
            
            if "AudioStream" in response:
                audio_stream.write(response["AudioStream"].read())
            else:
                return None, "No audio stream returned from Polly."
        except Exception as e:
            return None, f"Error generating speech for chunk {idx + 1}: {e}"
    
    return audio_stream.getvalue(), None

def upload_audio_to_s3(audio_bytes, file_name):
    """
    Uploads the given audio bytes to the specified S3 bucket with the provided file name.
    Returns the S3 object key.
    """
    try:
        s3_client.put_object(
            Bucket=s3_bucket_name,
            Key=file_name,
            Body=audio_bytes,
            ContentType='audio/mpeg'
        )
        return file_name, None
    except Exception as e:
        return None, f"Error uploading to S3: {e}"

def list_audio_files():
    """
    Lists all audio files in the specified S3 bucket.
    Returns a list of object keys.
    """
    try:
        paginator = s3_client.get_paginator('list_objects_v2')
        pages = paginator.paginate(Bucket=s3_bucket_name)
        audio_files = []
        for page in pages:
            if 'Contents' in page:
                audio_files.extend([obj['Key'] for obj in page['Contents'] if obj['Key'].lower().endswith('.mp3')])
        return audio_files, None
    except Exception as e:
        return None, f"Error listing S3 bucket contents: {e}"

def generate_presigned_url(object_key, expiration=3600):
    """
    Generates a presigned URL for the given S3 object key.
    """
    try:
        url = s3_client.generate_presigned_url(
            'get_object',
            Params={'Bucket': s3_bucket_name, 'Key': object_key},
            ExpiresIn=expiration
        )
        return url, None
    except Exception as e:
        return None, f"Error generating presigned URL: {e}"

def delete_audio_from_s3(object_key):
    """
    Deletes the specified audio file from the S3 bucket.
    """
    try:
        s3_client.delete_object(Bucket=s3_bucket_name, Key=object_key)
        return True, None
    except Exception as e:
        return False, f"Error deleting audio file: {e}"

# --- Page 1: Summarize and Generate Audio / Generate Audio Without Summarizing ---

def summarize_and_generate_audio():
    st.header("📄 Summarize and Generate Audio")
    st.markdown("---")
    
    # Option Selection
    operation_mode = st.radio(
        "📝 Choose Operation:",
        options=["Summarize and Generate Audio", "Generate Audio Without Summarizing"],
        index=0,
        horizontal=True
    )
    
    # Input Mode Selection
    input_mode = st.radio(
        "🔗 Choose Input Method:",
        options=["Provide a URL", "Paste Raw Text"],
        index=0,
        horizontal=True
    )
    
    # Form for inputs
    with st.form(key='audio_form'):
        if input_mode == "Provide a URL":
            url_input = st.text_input("🔗 Enter the URL:", placeholder="https://example.com")
            raw_text = None  # Ensure raw_text is None when URL is used
        else:
            raw_text = st.text_area("📝 Paste your raw text here:", height=200, placeholder="Enter your text here...")
            url_input = None  # Ensure url_input is None when raw text is used
        
        if operation_mode == "Summarize and Generate Audio":
            max_words = st.number_input(
                "✂️ Max words for summary:",
                min_value=10,
                max_value=1000,  # Reduced max to prevent exceeding Polly's limits
                value=100,  # Adjusted default
                step=10,
                help="Ensure that the summary does not exceed AWS Polly's maximum character limit."
            )
        
        file_name_input = st.text_input(
            "🎼 Optional: Enter a custom name for the audio file (without .mp3):",
            placeholder="my_summary_audio"
        )
        submit_button = st.form_submit_button(label='Process')

    if submit_button:
        # Input Validation
        if input_mode == "Provide a URL" and not url_input:
            st.error("❗ Please enter a valid URL.")
        elif input_mode == "Paste Raw Text" and not raw_text.strip():
            st.error("❗ Please paste some text to process.")
        else:
            # Extract or Use Raw Text
            if input_mode == "Provide a URL":
                with st.spinner("🕒 Fetching webpage content..."):
                    page_text, error = extract_text_from_url(url_input)

                if error:
                    st.error(f"❌ Error fetching page: {error}")
                    page_text = None
            else:
                page_text = raw_text
                error = None  # No error when using raw text

            if error:
                st.error(f"❌ {error}")
            elif not page_text:
                st.error("❌ No text found to process.")
            else:
                # Depending on operation mode, summarize or use raw text
                if operation_mode == "Summarize and Generate Audio":
                    with st.spinner("📝 Summarizing with Claude-3 Sonnet..."):
                        summary, error = summarize_text_bedrock(page_text, max_words)

                    if error:
                        st.error(f"❌ {error}")
                        summary = None
                    elif not summary:
                        st.error("❌ No summary returned from Bedrock.")
                    else:
                        st.subheader("📑 Summary")
                        st.write(summary)
                        text_to_convert = summary
                else:
                    text_to_convert = page_text  # Use full text without summarizing

                if text_to_convert:
                    # Check the length of the text
                    if len(text_to_convert) > 3000:
                        st.warning(
                            "⚠️ The text is longer than AWS Polly's maximum limit of 3000 characters. "
                            "It will be split into smaller chunks for processing."
                        )
                    
                    # Generate Audio
                    with st.spinner("🎤 Generating audio with AWS Polly..."):
                        audio_bytes, error = text_to_speech_polly(text_to_convert)

                    if error:
                        st.error(f"❌ {error}")
                    elif not audio_bytes:
                        st.error("❌ Failed to generate audio.")
                    else:
                        # Handle custom file name
                        if file_name_input:
                            # Sanitize the file name
                            sanitized_file_name = "".join(c for c in file_name_input if c.isalnum() or c in (' ', '_', '-')).rstrip()
                            if not sanitized_file_name:
                                st.warning("⚠️ Provided file name is invalid. Using a generated name instead.")
                                sanitized_file_name = None
                            else:
                                # Ensure the file name ends with .mp3
                                if not sanitized_file_name.lower().endswith('.mp3'):
                                    sanitized_file_name += '.mp3'
                        else:
                            sanitized_file_name = None
                        
                        if sanitized_file_name:
                            file_name = sanitized_file_name
                        else:
                            # Generate a unique file name using UUID and domain or a generic identifier
                            if input_mode == "Provide a URL":
                                parsed_url = urlparse(url_input)
                                domain = parsed_url.netloc.replace('.', '_')
                                unique_id = str(uuid4())
                                if operation_mode == "Summarize and Generate Audio":
                                    file_name = f"{domain}_summary_{unique_id}.mp3"
                                else:
                                    file_name = f"{domain}_{unique_id}.mp3"
                            else:
                                unique_id = str(uuid4())
                                if operation_mode == "Summarize and Generate Audio":
                                    file_name = f"rawtext_summary_{unique_id}.mp3"
                                else:
                                    file_name = f"rawtext_{unique_id}.mp3"

                        with st.spinner("💾 Uploading audio to Amazon S3..."):
                            uploaded_file_key, error = upload_audio_to_s3(audio_bytes, file_name)

                        if error:
                            st.error(f"❌ {error}")
                        else:
                            st.success("✅ Audio successfully generated and uploaded to S3.")
                            if file_name_input:
                                st.info(f"📁 File Name: `{file_name}`")

                            # Generate a presigned URL for the uploaded audio
                            audio_url, error = generate_presigned_url(uploaded_file_key)

                            if error:
                                st.error(f"❌ {error}")
                            else:
                                st.subheader("🔊 Generated Audio")
                                st.audio(audio_url, format="audio/mp3", start_time=0)

# --- Page 2: Stored Audio Files ---

def stored_audio_files():
    st.header("🎵 Stored Audio Files")
    st.markdown("---")
    
    # Create a placeholder for the audio player
    audio_player_placeholder = st.empty()
    
    with st.spinner("🔍 Fetching stored audio files from S3..."):
        audio_files, error = list_audio_files()
    
    if error:
        st.error(f"❌ {error}")
    elif not audio_files:
        st.info("ℹ️ No audio files found in the S3 bucket.")
    else:
        # Create a selection dropdown for audio files
        selected_audio = st.selectbox("🔎 Select an audio file to play:", options=audio_files)
        
        col1, col2 = st.columns([3, 1])
        with col1:
            if selected_audio:
                with st.spinner("⏳ Generating presigned URL..."):
                    audio_url, error = generate_presigned_url(selected_audio)
        
                if error:
                    st.error(f"❌ {error}")
                else:
                    st.subheader("🔊 Playback")
                    # Update the audio player placeholder with the new audio
                    audio_player_placeholder.audio(audio_url, format="audio/mp3", start_time=0)
        
        with col2:
            if selected_audio:
                delete_confirm = st.button("🗑️ Delete Selected Audio")
                if delete_confirm:
                    with st.spinner("🗑️ Deleting audio file from S3..."):
                        success, delete_error = delete_audio_from_s3(selected_audio)
                    if success:
                        st.success(f"✅ `{selected_audio}` has been deleted successfully.")
                        # st.experimental_rerun()  # Refresh the page to update the list
                    else:
                        st.error(f"❌ {delete_error}")

# --- Streamlit App ---

def main():
    st.set_page_config(page_title="Website/Text Summarizer with Audio Playback", layout="wide")
    
    st.sidebar.title("Navigation")
    page = st.sidebar.selectbox("Select a Page", ["Summarize and Generate Audio", "Stored Audio Files"])
    
    st.sidebar.markdown("---")
    st.sidebar.info("This app summarizes website content or raw text and generates audio playback using AWS services.")
    
    if page == "Summarize and Generate Audio":
        summarize_and_generate_audio()
    elif page == "Stored Audio Files":
        stored_audio_files()

if __name__ == "__main__":
    main()
