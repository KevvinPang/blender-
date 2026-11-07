import traceback

from huggingface_hub import login
from runtime import MODEL_URL, download_model

if __name__ == "__main__":
    print("SAM 3D Body - model login and download")
    print("First request and receive access at:", MODEL_URL)
    print("Create a read token at https://huggingface.co/settings/tokens")
    print("Paste the token ONLY into the hidden terminal prompt, not into chat.")
    try:
        login(add_to_git_credential=False)
        print(download_model())
    except Exception:
        traceback.print_exc()
    input("Press Enter to close...")
