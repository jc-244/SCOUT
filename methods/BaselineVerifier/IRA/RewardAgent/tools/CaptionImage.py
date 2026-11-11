from smolagents import Tool
import os
from dotenv import load_dotenv
from openai import AzureOpenAI, OpenAI
import base64

class CaptionImageTool(Tool):
    name = "caption_image"
    description = "To ask a question about an image."
    inputs = { 
        "path_to_image": {
            "description": "caption the given image or answer a qustion.",
            "type": "string",
        },
        "question": {
            "description": "the question you want to ask.",
            "type": "string",
        },
    }
    output_type = "string"
    def __init__(self):
        super().__init__()
    
    def forward(self, path_to_image,question):
        return self.__call__(path_to_image,question)
    
    def __call__(self,path_to_image,question):
        
        load_dotenv()
        api_key = os.getenv("IRA_API_KEY") or os.getenv("deerapi_key")
        base_url = os.getenv("IRA_BASE_URL") or os.getenv("OPENAI_BASE_URL") or "https://api.cometapi.com/v1/"
        client = OpenAI(
            api_key=api_key,
            base_url=base_url
        )
        ext = os.path.splitext(path_to_image)[1].lower()
        if ext == ".png":
            mime_type = "image/png"
        elif ext in [".jpg", ".jpeg"]:
            mime_type = "image/jpeg"
        elif ext == ".gif":
            mime_type = "image/gif"
        else:
            mime_type = "application/octet-stream"
        with open(path_to_image, "rb") as image_file:
            encoded_img = base64.b64encode(image_file.read()).decode('utf-8')
            
        base64_url = f"data:{mime_type};base64,{encoded_img}"
        messages=[
            {"role":"user",
            "content":[
                {"type":"text","text":question},
                {"type": "image_url", "image_url": {"url": base64_url}}]}
            ]
        
        response = client.chat.completions.create(
                    model='gpt-4o-mini',
                    messages=messages
                )
        return  response.choices[0].message.content

    def to_code_prompt(self):
        return (
            "def caption_image(path_to_image: str, question: str) -> str:\n"
            "    '''caption the given image or answer a qustion..'''\n"
        )
        

if __name__ == "__main__":
    import argparse
    import sys

    parser = argparse.ArgumentParser(description="Test CaptionImageTool")
    parser.add_argument("path_to_image", help="Path to the image file to caption")
    parser.add_argument(
        "-q", "--question",
        default="Describe the image.",
        help="Question to ask about the image"
    )
    parser.add_argument(
        "--env",
        default=None,
        help="Path to a .env file (optional; falls back to default search)"
    )
    args = parser.parse_args()

    try:
        if args.env:
            load_dotenv(args.env)
        else:
            load_dotenv()
    except Exception as e:
        print(f"Warning: failed to load .env: {e}", file=sys.stderr)

    deer_key = os.getenv("IRA_API_KEY") or os.getenv("deerapi_key")
    if not deer_key and os.getenv("DEER_API_KEY"):
        os.environ["deerapi_key"] = os.getenv("DEER_API_KEY")
        deer_key = os.environ["deerapi_key"]

    if not deer_key:
        print("Error: missing deerapi_key (or DEER_API_KEY) in environment.", file=sys.stderr)
        sys.exit(1)

    if not os.path.exists(args.path_to_image):
        print(f"Error: image not found: {args.path_to_image}", file=sys.stderr)
        sys.exit(2)

    try:
        tool = CaptionImageTool()
        result = tool.forward(args.path_to_image, args.question)
        print("\n=== Caption/Answer ===\n")
        print(result)
    except Exception as e:
        print(f"Error while running CaptionImageTool: {e}", file=sys.stderr)
        sys.exit(3)
