from sys import argv

from openai import OpenAI
from openai.types.responses.parsed_response import ParsedResponse
from pydantic import BaseModel


class ResponseFormat(BaseModel):
    urls: list[str]


client = OpenAI(base_url="http://localhost:11434/v1", api_key="dummy")

with open("sample.txt") as f:
    text = f.read()


resp: ParsedResponse[ResponseFormat] = client.responses.parse(
    model="gemma4:e4b",
    text_format=ResponseFormat,
    input=[
        {
            "role": "system",
            "content": "Please extract URLs from the markdown text provided by the user.",
        },
        {"role": "user", "content": text},
    ],
)

parsed = resp.output_parsed

with open(f"{argv[0]}.json", "w") as f:
    f.write(parsed.model_dump_json(indent=2, ensure_ascii=False))
