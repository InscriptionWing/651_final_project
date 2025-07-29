#!/usr/bin/env python
"""Enhanced test script to verify Hugging Face authentication and model access"""

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
        print(f"Token length: {len(token)} chars")

    return model_id, token


def test_token_validity():
    """Test if the token is valid by checking user info"""
    print("\n=== Testing Token Validity ===")

    try:
        from huggingface_hub import HfApi

        token = os.getenv("HF_HUB_TOKEN") or os.getenv("HF_TOKEN")

        api = HfApi(token=token)

        # Test basic token validity
        user_info = api.whoami()
        print(f"✓ Token is valid!")
        print(f"  Username: {user_info['name']}")
        print(f"  Account type: {user_info.get('type', 'user')}")

        return True, user_info

    except Exception as e:
        print(f"✗ Token validation failed: {e}")
        return False, None


def test_model_access():
    """Test access to the specific Mistral model"""
    print("\n=== Testing Model Access ===")

    try:
        from huggingface_hub import HfApi

        model_id = os.getenv("HF_MODEL", "mistralai/Mistral-7B-Instruct-v0.3")
        token = os.getenv("HF_HUB_TOKEN") or os.getenv("HF_TOKEN")

        api = HfApi(token=token)

        # Check if model exists and is accessible
        try:
            model_info = api.model_info(model_id)
            print(f"✓ Model is accessible: {model_id}")
            print(f"  Model name: {model_info.modelId}")
            print(f"  Downloads: {model_info.downloads:,}")
            print(f"  Gated: {getattr(model_info, 'gated', 'Unknown')}")
            return True

        except Exception as model_error:
            if "gated" in str(model_error).lower() or "401" in str(model_error):
                print(f"⚠ Model is gated and requires approval: {model_id}")
                print(f"  Go to: https://huggingface.co/{model_id}")
                print(f"  Click 'Request Access' and wait for approval")
                return False
            else:
                print(f"✗ Cannot access model: {model_error}")
                return False

    except Exception as e:
        print(f"✗ Failed to test model access: {e}")
        return False


def suggest_alternative_models():
    """Suggest alternative models that don't require gating"""
    print("\n=== Alternative Models (No Gating Required) ===")

    alternatives = [
        ("microsoft/DialoGPT-medium", "Good for conversational tasks"),
        ("gpt2", "Classic, reliable text generation"),
        ("google/flan-t5-base", "Good instruction following"),
        ("TinyLlama/TinyLlama-1.1B-Chat-v1.0", "Lightweight chat model"),
        ("microsoft/DialoGPT-large", "Larger conversational model"),
    ]

    print("You can use these models without approval:")
    for model, description in alternatives:
        print(f"  • {model}")
        print(f"    {description}")

    print(f"\nTo use an alternative, update your .env file:")
    print(f"HF_MODEL=microsoft/DialoGPT-medium")


def test_alternative_model():
    """Test access to a non-gated model"""
    print("\n=== Testing Alternative Model Access ===")

    try:
        from huggingface_hub import HfApi

        token = os.getenv("HF_HUB_TOKEN") or os.getenv("HF_TOKEN")
        api = HfApi(token=token)

        # Test a known public model
        test_model = "microsoft/DialoGPT-medium"
        model_info = api.model_info(test_model)

        print(f"✓ Alternative model accessible: {test_model}")
        print(f"  This model works without gating restrictions")

        return True

    except Exception as e:
        print(f"✗ Cannot access alternative model: {e}")
        return False


def main():
    print("=== Enhanced Hugging Face Authentication Test ===")

    # Load environment
    env_loaded = find_and_load_env()

    # Test environment variables
    model_id, token = test_environment()

    if not token:
        print("\n❌ No authentication token found!")
        print("1. Go to: https://huggingface.co/settings/tokens")
        print("2. Create a new 'Read' token")
        print("3. Add it to your .env file as HF_HUB_TOKEN=hf_your_token_here")
        return

    # Test token validity
    token_valid, user_info = test_token_validity()

    # Test model access
    model_accessible = False
    if token_valid:
        model_accessible = test_model_access()

    # Test alternative model
    alternative_works = test_alternative_model()

    print("\n=== Test Summary ===")
    print(f"Environment loaded: {'✓' if env_loaded else '✗'}")
    print(f"Token found: {'✓' if token else '✗'}")
    print(f"Token valid: {'✓' if token_valid else '✗'}")
    print(f"Model accessible: {'✓' if model_accessible else '✗'}")
    print(f"Alternative models work: {'✓' if alternative_works else '✗'}")

    if token_valid and model_accessible:
        print("\n🎉 Perfect! Your authentication works and you have access to the Mistral model.")
        print("You can now run: python merge_pipeline.py --count 10 --outfile data/test.jsonl")
    elif token_valid and alternative_works:
        print("\n⚠ Your token works, but you need approval for the Mistral model.")
        print("OPTION 1: Request access at https://huggingface.co/mistralai/Mistral-7B-Instruct-v0.3")
        print("OPTION 2: Use an alternative model (see suggestions above)")
        suggest_alternative_models()
    else:
        print("\n❌ Authentication issues detected.")
        print("1. Generate a new token at: https://huggingface.co/settings/tokens")
        print("2. Make sure it has 'Read' permissions")
        print("3. Update your .env file with the new token")


if __name__ == "__main__":
    main()