#!/usr/bin/env python
"""Test script to verify Hugging Face authentication and model access"""

import os
from pathlib import Path
from dotenv import load_dotenv


def find_and_load_env():
    """Find and load .env file from various locations"""
    current_path = Path(__file__).parent

    # Check current directory first
    env_path = current_path / '.env'
    if env_path.exists():
        load_dotenv(env_path)
        print(f"✓ Loaded .env from: {env_path}")
        return True

    # Check parent directory
    env_path = current_path.parent / '.env'
    if env_path.exists():
        load_dotenv(env_path)
        print(f"✓ Loaded .env from: {env_path}")
        return True

    # Check root directory (two levels up)
    env_path = current_path.parent.parent / '.env'
    if env_path.exists():
        load_dotenv(env_path)
        print(f"✓ Loaded .env from: {env_path}")
        return True

    print("⚠ No .env file found")
    return False


def test_environment():
    """Test environment variables"""
    print("\n=== Testing Environment Variables ===")

    model_id = os.getenv("HF_MODEL")
    token = os.getenv("HF_HUB_TOKEN") or os.getenv("HF_TOKEN")

    print(f"HF_MODEL: {model_id}")
    print(f"HF_HUB_TOKEN: {'Found' if token else 'Not found'}")
    if token:
        print(f"Token preview: {token[:10]}...")

    return model_id, token


def test_huggingface_connection():
    """Test connection to Hugging Face"""
    print("\n=== Testing Hugging Face Connection ===")

    try:
        from huggingface_hub import HfApi

        model_id = os.getenv("HF_MODEL", "mistralai/Mistral-7B-Instruct-v0.3")
        token = os.getenv("HF_HUB_TOKEN") or os.getenv("HF_TOKEN")

        api = HfApi(token=token)

        # Test if we can access the model info
        model_info = api.model_info(model_id)
        print(f"✓ Successfully accessed model: {model_id}")
        print(f"  Model ID: {model_info.modelId}")
        print(f"  Tags: {model_info.tags[:5]}...")  # Show first 5 tags

        return True

    except Exception as e:
        print(f"✗ Failed to access model: {e}")
        return False


def test_transformers_loading():
    """Test loading model with transformers"""
    print("\n=== Testing Transformers Model Loading ===")

    try:
        from transformers import AutoTokenizer

        model_id = os.getenv("HF_MODEL", "mistralai/Mistral-7B-Instruct-v0.3")
        token = os.getenv("HF_HUB_TOKEN") or os.getenv("HF_TOKEN")

        print(f"Attempting to load tokenizer for: {model_id}")

        tokenizer = AutoTokenizer.from_pretrained(
            model_id,
            token=token,
            trust_remote_code=True
        )

        print(f"✓ Successfully loaded tokenizer")
        print(f"  Vocab size: {tokenizer.vocab_size}")
        print(f"  Model max length: {tokenizer.model_max_length}")

        return True

    except Exception as e:
        print(f"✗ Failed to load tokenizer: {e}")
        return False


def main():
    print("=== Hugging Face Authentication Test ===")

    # Load environment
    env_loaded = find_and_load_env()

    # Test environment variables
    model_id, token = test_environment()

    if not token:
        print("\n❌ No authentication token found!")
        print("Make sure your .env file contains HF_HUB_TOKEN")
        return

    # Test HF Hub connection
    hub_success = test_huggingface_connection()

    # Test transformers loading (just tokenizer for speed)
    transformers_success = test_transformers_loading()

    print("\n=== Test Summary ===")
    print(f"Environment loaded: {'✓' if env_loaded else '✗'}")
    print(f"Token found: {'✓' if token else '✗'}")
    print(f"HF Hub access: {'✓' if hub_success else '✗'}")
    print(f"Transformers loading: {'✓' if transformers_success else '✗'}")

    if hub_success and transformers_success:
        print("\n🎉 All tests passed! Your authentication is working correctly.")
        print("You can now run the full pipeline with the Mistral model.")
    else:
        print("\n❌ Some tests failed. Check the errors above.")


if __name__ == "__main__":
    main()