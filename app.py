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
import time
from pytube import YouTube
from pydub import AudioSegment
import tempfile  # for temporary files
import ssl
import certifi
from pytube import YouTube
from pydub import AudioSegment
import tempfile
import os
import ssl
from pytube import YouTube
import yt_dlp
import tempfile
import os

# Create an unverified SSL context
ssl_context = ssl._create_unverified_context()
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

# Initialize AWS clients
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

try:
    s3_client = boto3.client('s3', region_name=aws_region)
except Exception as e:
    st.error(f"Error initializing S3 client: {e}")
    st.stop()

try:
    transcribe_client = boto3.client('transcribe', region_name=aws_region)
except Exception as e:
    st.error(f"Error initializing Transcribe client: {e}")
    st.stop()

# Initialize LangChain Bedrock LLM
from langchain_community.embeddings import BedrockEmbeddings
from langchain.llms.bedrock import Bedrock

bedrock_embeddings = BedrockEmbeddings(model_id="amazon.titan-embed-text-v1", client=bedrock_client)

llm = Bedrock(
    model_id="anthropic.claude-v2:1",  # or "anthropic.claude-v2" if needed
    client=bedrock_client,
    model_kwargs={
        'max_tokens_to_sample': 5000,  # Adjust as needed
        'temperature': 0.5  # Adjust as needed
    }
)

# --- Helper Functions --------------------------------------------------------

def download_youtube_audio_to_mp3(youtube_url):
    """
    Downloads the audio from a YouTube video using yt-dlp, converts it to MP3 in memory,
    and returns the MP3 data as bytes.
    """
    temp_mp3_path = None

    try:
        # 1) Set up yt-dlp options for audio-only download
        ydl_opts = {
            'format': 'bestaudio/best',
            'outtmpl': os.path.join(tempfile.gettempdir(), '%(id)s.%(ext)s'),
            'postprocessors': [{
                'key': 'FFmpegExtractAudio',
                'preferredcodec': 'mp3',
                'preferredquality': '192',
            }],
        }

        # 2) Use yt-dlp to download the audio
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info_dict = ydl.extract_info(youtube_url, download=True)
            temp_mp3_path = ydl.prepare_filename(info_dict).replace(info_dict['ext'], 'mp3')

        # 3) Read the MP3 bytes from disk
        with open(temp_mp3_path, "rb") as f:
            mp3_bytes = f.read()

        return mp3_bytes, None

    except Exception as e:
        return None, str(e)

    finally:
        # Clean up the temporary MP3 file if it exists
        if temp_mp3_path and os.path.exists(temp_mp3_path):
            os.remove(temp_mp3_path)
            
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
    Summarizes the given text using AWS Bedrock (Claude-2 / v2) via LangChain's Bedrock LLM.
    """
    # Prepare the prompt with the required structure
    prompt = (
        f"Human: Summarize the following text in {max_words} words with the first line as the topic of the summary:\n\n"
        f"{text}\n\nAssistant: "
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
            response = polly_client.synthesize_speech(
                Text=chunk,
                OutputFormat='mp3',
                VoiceId=voice_id,
                Engine='neural'
            )
            
            if "AudioStream" in response:
                audio_stream.write(response["AudioStream"].read())
            else:
                return None, "No audio stream returned from Polly."
        except Exception as e:
            return None, f"Error generating speech for chunk {idx + 1}: {e}"
    
    return audio_stream.getvalue(), None

def upload_audio_to_s3(audio_bytes, file_name, s3_folder="stored_audio_files"):
    """
    Uploads the given audio bytes to the specified S3 bucket within a given folder (prefix).
    Returns the S3 object key.
    """
    # Construct the folder/prefix in S3
    s3_key = f"{s3_folder}/{file_name}"

    try:
        s3_client.put_object(
            Bucket=s3_bucket_name,
            Key=s3_key,
            Body=audio_bytes,
            ContentType='audio/mpeg'
        )
        return s3_key, None
    except Exception as e:
        return None, f"Error uploading to S3: {e}"

def upload_text_to_s3(text, file_name, s3_folder="view_transcriptions_and_summaries"):
    """
    Uploads the given text string as a .txt file to the specified S3 bucket 
    within a given folder (prefix).
    """
    s3_key = f"{s3_folder}/{file_name}"

    try:
        s3_client.put_object(
            Bucket=s3_bucket_name,
            Key=s3_key,
            Body=text.encode("utf-8"),
            ContentType='text/plain'
        )
        return s3_key, None
    except Exception as e:
        return None, f"Error uploading text to S3: {e}"

def list_audio_files(s3_folder="stored_audio_files"):
    """
    Lists all audio (.mp3) files in the specified S3 bucket folder (prefix).
    Returns a list of object keys.
    """
    try:
        paginator = s3_client.get_paginator('list_objects_v2')
        pages = paginator.paginate(Bucket=s3_bucket_name, Prefix=s3_folder)
        audio_files = []
        for page in pages:
            if 'Contents' in page:
                audio_files.extend([
                    obj['Key'] for obj in page['Contents'] 
                    if obj['Key'].lower().endswith('.mp3')
                ])
        return audio_files, None
    except Exception as e:
        return None, f"Error listing S3 bucket contents: {e}"

def list_text_files(s3_folder="view_transcriptions_and_summaries"):
    """
    Lists all text (.txt) files in the specified S3 bucket folder (prefix).
    Returns a list of object keys.
    """
    try:
        paginator = s3_client.get_paginator('list_objects_v2')
        pages = paginator.paginate(Bucket=s3_bucket_name, Prefix=s3_folder)
        text_files = []
        for page in pages:
            if 'Contents' in page:
                text_files.extend([
                    obj['Key'] for obj in page['Contents'] 
                    if obj['Key'].lower().endswith('.txt')
                ])
        return text_files, None
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
    Deletes the specified file (e.g., .mp3) from the S3 bucket.
    """
    try:
        s3_client.delete_object(Bucket=s3_bucket_name, Key=object_key)
        return True, None
    except Exception as e:
        return False, f"Error deleting file: {e}"

def delete_text_from_s3(object_key):
    """
    Deletes the specified .txt file from the S3 bucket.
    """
    try:
        s3_client.delete_object(Bucket=s3_bucket_name, Key=object_key)
        return True, None
    except Exception as e:
        return False, f"Error deleting file: {e}"

def transcribe_audio_from_s3(object_key, language_code="en-US"):
    """
    Starts an Amazon Transcribe job on an S3 .mp3 file, waits for completion,
    then returns the transcribed text.
    """
    job_name = f"transcription-job-{uuid4()}"
    media_uri = f"s3://{s3_bucket_name}/{object_key}"

    try:
        transcribe_client.start_transcription_job(
            TranscriptionJobName=job_name,
            Media={'MediaFileUri': media_uri},
            MediaFormat='mp3',
            LanguageCode=language_code
        )
    except transcribe_client.exceptions.ConflictException:
        return None, "A transcription job with the same name already exists. Please try again."
    except Exception as e:
        return None, f"Error starting transcription job: {e}"

    # Wait for transcription to complete
    while True:
        status = transcribe_client.get_transcription_job(TranscriptionJobName=job_name)
        if status['TranscriptionJob']['TranscriptionJobStatus'] in ['COMPLETED', 'FAILED']:
            break
        placeholder = st.empty()
        placeholder.success("Operation successful!")
        time.sleep(2)
        placeholder.empty()

    if status['TranscriptionJob']['TranscriptionJobStatus'] == 'FAILED':
        return None, "Transcription job failed."

    # Retrieve the transcript
    try:
        transcript_url = status['TranscriptionJob']['Transcript']['TranscriptFileUri']
        transcript_json = requests.get(transcript_url).json()
        transcript_text = transcript_json['results']['transcripts'][0]['transcript']
        return transcript_text, None
    except Exception as e:
        return None, f"Error retrieving/parsing transcript: {e}"

# --- Page 1: Summarize and Generate Audio / Generate Audio Without Summarizing ----

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
                max_value=1000,  # Reduced max to prevent extremely large prompts
                value=100,       # Adjusted default
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
                    with st.spinner("📝 Summarizing with Claude..."):
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

                        # Upload to S3 in the "stored_audio_files" folder
                        with st.spinner("💾 Uploading audio to Amazon S3..."):
                            uploaded_file_key, error = upload_audio_to_s3(
                                audio_bytes, 
                                file_name, 
                                s3_folder="stored_audio_files"  # <--- Folder for Summarize/Generate
                            )

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

# --- Page 2: Stored Audio Files ---------------------------------------------


def stored_audio_files():
    st.header("🎵 Stored Audio Files")
    st.markdown("---")
    
    # Create a placeholder for the audio player
    audio_player_placeholder = st.empty()
    
    with st.spinner("🔍 Fetching stored audio files from S3..."):
        audio_files, error = list_audio_files(s3_folder="stored_audio_files")
    
    if error:
        st.error(f"❌ {error}")
    elif not audio_files:
        st.info("ℹ️ No audio files found in the S3 bucket folder: `stored_audio_files`.")
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

# --- Page 3 (NEW): Transcribe & Summarize Uploaded Audio --------------------


def transcribe_and_summarize_audio():
    st.header("🎙️ Transcribe & Summarize Audio (MP3 or YouTube)")
    st.markdown("---")

    st.info(
        "You can either upload an MP3 file **or** provide a YouTube link. "
        "The audio will then be transcribed. Optionally, you can also save:\n"
        "1) The full transcription (.txt)\n"
        "2) A summary of that transcription (.txt)\n"
        "3) An audio version of the summary (.mp3)."
    )

    # NEW: Select Input Type
    input_choice = st.radio(
        "Choose Input Source:",
        options=["Upload MP3 File", "YouTube Link"],
        index=0,
        horizontal=True
    )

    # Container to hold either file_uploader or YouTube link text input
    uploaded_file = None
    youtube_url = None

    with st.form(key='transcribe_form'):
        if input_choice == "Upload MP3 File":
            uploaded_file = st.file_uploader("Upload your MP3 file:", type=["mp3"])
        else:
            # YouTube Link
            youtube_url = st.text_input("Enter YouTube video URL:")
        
        # Checkboxes for what to save
        save_transcript = st.checkbox("Save Transcription (.txt) to S3?", value=True)
        custom_transcript_name = st.text_input("Optional custom name for Transcript file (without .txt):")

        save_summary = st.checkbox("Save Summary (.txt) to S3?", value=True)
        custom_summary_name = st.text_input("Optional custom name for Summary file (without .txt):")

        generate_summary_audio = st.checkbox("Generate and Save Summary Audio (.mp3)?", value=True)
        custom_summary_audio_name = st.text_input("Optional custom name for Summary Audio file (without .mp3):")

        # If generating summary, let user pick max words
        if generate_summary_audio or save_summary:
            max_summary_words = st.number_input(
                "Maximum words for summary:",
                min_value=10,
                max_value=1000,
                value=100,
                step=10
            )

        submit_transcribe = st.form_submit_button("Process Audio")

    if submit_transcribe:
        # ----------------------------------------------------
        # Validate user input 
        # ----------------------------------------------------
        mp3_bytes = None
        base_file_name = None

        if input_choice == "Upload MP3 File":
            if not uploaded_file:
                st.error("❗ Please upload an MP3 file.")
                return

            # Read MP3 bytes from uploaded file
            mp3_bytes = uploaded_file.read()
            # For naming, remove .mp3 extension from original name
            base_file_name = os.path.splitext(uploaded_file.name)[0]

        else:  # "YouTube Link"
            if not youtube_url.strip():
                st.error("❗ Please enter a valid YouTube URL.")
                return
            
            # Download and convert YouTube audio to MP3 bytes
            with st.spinner("📥 Downloading and converting YouTube audio..."):
                mp3_bytes, download_error = download_youtube_audio_to_mp3(youtube_url)
            
            if download_error:
                st.error(f"❌ Error downloading YouTube audio: {download_error}")
                return
            
            # Generate a base file name from the YouTube URL
            # For example, just use the last part or domain
            base_file_name = "youtube_audio"

        # ----------------------------------------------------
        # Step 1: Upload the MP3 to S3 (for Transcribe)
        # ----------------------------------------------------
        unique_id = str(uuid4())
        s3_key_mp3 = f"view_transcriptions_and_summaries/{base_file_name}_{unique_id}.mp3"

        try:
            s3_client.put_object(
                Bucket=s3_bucket_name,
                Key=s3_key_mp3,
                Body=mp3_bytes,
                ContentType='audio/mpeg'
            )
            st.success(f"✅ MP3 uploaded to S3 as: {s3_key_mp3}")
        except Exception as e:
            st.error(f"Error uploading MP3 to S3: {e}")
            return

        # ----------------------------------------------------
        # Step 2: Transcribe the MP3
        # ----------------------------------------------------
        with st.spinner("📜 Transcribing audio..."):
            transcribed_text, error = transcribe_audio_from_s3(s3_key_mp3)
        
        if error:
            st.error(f"❌ {error}")
            return

        st.subheader("Transcribed Text")
        st.write(transcribed_text)

        # ----------------------------------------------------
        # Step 3: (Optional) Save transcript to S3
        # ----------------------------------------------------
        if save_transcript:
            if custom_transcript_name.strip():
                # Sanitize custom name
                transcript_file_name = "".join(c for c in custom_transcript_name if c.isalnum() or c in (' ', '_', '-')).rstrip()
                if not transcript_file_name.lower().endswith(".txt"):
                    transcript_file_name += ".txt"
            else:
                transcript_file_name = f"{base_file_name}_{unique_id}-transcript.txt"
            
            with st.spinner("💾 Saving transcript to S3..."):
                _, error = upload_text_to_s3(
                    transcribed_text, 
                    transcript_file_name, 
                    s3_folder="view_transcriptions_and_summaries"
                )
            if error:
                st.error(f"❌ {error}")
            else:
                st.info(f"📝 Transcript saved as `{transcript_file_name}` in 'view_transcriptions_and_summaries'.")

        # ----------------------------------------------------
        # Step 4: (Optional) Summarize the transcription
        # ----------------------------------------------------
        summary_text = None
        if save_summary or generate_summary_audio:
            with st.spinner("📝 Summarizing transcription with Claude..."):
                summary_text, error = summarize_text_bedrock(transcribed_text, max_words=max_summary_words)
            if error:
                st.error(f"❌ {error}")
                return

            st.subheader("Summary Text")
            st.write(summary_text)

            # Save summary text to S3
            if save_summary:
                if custom_summary_name.strip():
                    # Sanitize custom name
                    summary_file_name = "".join(c for c in custom_summary_name if c.isalnum() or c in (' ', '_', '-')).rstrip()
                    if not summary_file_name.lower().endswith(".txt"):
                        summary_file_name += ".txt"
                else:
                    summary_file_name = f"{base_file_name}_{unique_id}-summary.txt"

                with st.spinner("💾 Saving summary to S3..."):
                    _, error = upload_text_to_s3(
                        summary_text, 
                        summary_file_name, 
                        s3_folder="view_transcriptions_and_summaries"
                    )
                if error:
                    st.error(f"❌ {error}")
                else:
                    st.info(f"📝 Summary saved as `{summary_file_name}` in 'view_transcriptions_and_summaries'.")

        # ----------------------------------------------------
        # Step 5: (Optional) Convert summary to MP3
        # ----------------------------------------------------
        if generate_summary_audio and summary_text:
            if custom_summary_audio_name.strip():
                # Sanitize custom name
                summary_audio_file_name = "".join(c for c in custom_summary_audio_name if c.isalnum() or c in (' ', '_', '-')).rstrip()
                if not summary_audio_file_name.lower().endswith(".mp3"):
                    summary_audio_file_name += ".mp3"
            else:
                summary_audio_file_name = f"{base_file_name}_{unique_id}-summary.mp3"

            with st.spinner("🎤 Generating summary audio with Polly..."):
                summary_audio_bytes, error = text_to_speech_polly(summary_text)
            if error:
                st.error(f"❌ {error}")
            elif not summary_audio_bytes:
                st.error("❌ Failed to generate summary audio.")
            else:
                with st.spinner("💾 Uploading summary audio to S3..."):
                    summary_mp3_key, error = upload_audio_to_s3(
                        summary_audio_bytes, 
                        summary_audio_file_name, 
                        s3_folder="view_transcriptions_and_summaries"
                    )
                if error:
                    st.error(f"❌ {error}")
                else:
                    st.success("✅ Summary audio successfully generated and uploaded to S3.")
                    audio_url, error = generate_presigned_url(summary_mp3_key)
                    if error:
                        st.error(f"❌ {error}")
                    else:
                        st.subheader("🔊 Summary Audio Playback")
                        st.audio(audio_url, format="audio/mp3")

# --- Page 4 (NEW): View Transcriptions & Summaries --------------------------

def view_transcriptions_and_summaries():
    st.header("📑 View Transcriptions & Summaries")
    st.markdown("---")
    st.info("Below is a combined list of `.mp3` and `.txt` files in the `view_transcriptions_and_summaries` folder.")

    # We can combine them or separate them. Let's combine for convenience:
    with st.spinner("🔎 Fetching files from S3..."):
        text_files, txt_error = list_text_files(s3_folder="view_transcriptions_and_summaries")
        audio_files, mp3_error = list_audio_files(s3_folder="view_transcriptions_and_summaries")
    
    if txt_error:
        st.error(f"❌ {txt_error}")
        text_files = []
    if mp3_error:
        st.error(f"❌ {mp3_error}")
        audio_files = []

    all_files = []
    if text_files:
        all_files += text_files
    if audio_files:
        all_files += audio_files

    if not all_files:
        st.info("ℹ️ No text or audio files found in the `view_transcriptions_and_summaries` folder.")
        return

    selected_file = st.selectbox("Select a file to view/play:", all_files)
    if selected_file:
        # Provide download or playback
        if selected_file.lower().endswith(".txt"):
            st.subheader("Text File Content")
            with st.spinner("⏳ Generating presigned URL..."):
                presigned_url, error = generate_presigned_url(selected_file)
            if error:
                st.error(f"❌ {error}")
            else:
                # We can display or allow user to download
                try:
                    resp = requests.get(presigned_url)
                    resp.raise_for_status()
                    file_content = resp.text
                    st.text_area("File Content (.txt):", value=file_content, height=300)
                except Exception as e:
                    st.error(f"Error fetching text file: {e}")

        elif selected_file.lower().endswith(".mp3"):
            st.subheader("Audio File Playback")
            with st.spinner("⏳ Generating presigned URL..."):
                presigned_url, error = generate_presigned_url(selected_file)
            if error:
                st.error(f"❌ {error}")
            else:
                st.audio(presigned_url, format="audio/mp3", start_time=0)

        # Deletion button
        delete_button = st.button(f"🗑️ Delete `{selected_file}`")
        if delete_button:
            with st.spinner("🗑️ Deleting file from S3..."):
                if selected_file.lower().endswith(".mp3"):
                    success, delete_error = delete_audio_from_s3(selected_file)
                else:
                    success, delete_error = delete_text_from_s3(selected_file)
            if success:
                st.success(f"✅ `{selected_file}` has been deleted.")
                # st.experimental_rerun()
            else:
                st.error(f"❌ {delete_error}")

# --- Streamlit App ----------------------------------------------------------

def main():
    st.set_page_config(page_title="Website/Text Summarizer & Audio Suite", layout="wide")
    
    st.sidebar.title("Navigation")
    page = st.sidebar.selectbox(
        "Select a Page", 
        [
            "[F1] Summarize and Generate Audio", 
            "[F1] Stored Audio Files",
            "[F2] Transcribe & Summarize Audio",
            "[F2] View Transcriptions & Summaries"
        ]
    )
    
    st.sidebar.markdown("---")
    st.sidebar.info(
        "This app can:\n"
        "1) Summarize website content or raw text and generate audio using AWS services.\n"
        "2) Transcribe and Summarize uploaded MP3 audio.\n"
        "3) View stored audio and text files (with separate S3 folders)."
    )
    
    if page == "[F1] Summarize and Generate Audio":
        summarize_and_generate_audio()
    elif page == "[F1] Stored Audio Files":
        stored_audio_files()
    elif page == "[F2] Transcribe & Summarize Audio":
        transcribe_and_summarize_audio()
    elif page == "[F2] View Transcriptions & Summaries":
        view_transcriptions_and_summaries()

if __name__ == "__main__":
    main()
