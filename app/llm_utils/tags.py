from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field
from dotenv import load_dotenv
import os

load_dotenv()

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.2)

SYSTEM_PROMPT = """
You are an helpful assistant that generates tags and a title based on the text.

The given text is a user's thought from text, audio and/or image.
So your task is to generate tags and a title that represent the information provided that helps the user understand and organize.

You will also be given all the tags that are used for the current user's thoughts, if there are not thoughts assume this is the first time and generate relevant tags.
Don't create unnecessary tags, if the text is about a specific person, don't create a tag for "person".
The tags should be tight. Generate at most 5 tags. Do not imporovise and create tags that are not relvant to the text directly.
Do not generate the tag "thoughts/thought".

<text>
{text}
</text>

<tags>
{tags}
</tags>

Do your best job of genrating tags and a title based on the information provided. 
"""

SYSTEM_PROMPT_TITLE = """
You are an helpful assistant that generates a title based on the text.

The given text is a user's thought from text, audio and/or image.
So your task is to generate a title that represents the information provided that helps the user understand and organize.

"""

class Tag(BaseModel):
    name: str = Field(description="The name of the tag that represents the information in the text")

class TagsAndTitle(BaseModel):
    tags: list[Tag] = Field(description="The tags that represent the information in the text")
    title: str = Field(description="The title that represents the information in the text")

class Title(BaseModel):
    title: str = Field(description="The title of the chat session that represents the information in the text")

def generate_tags_and_title(text: str, tags: list[Tag]) -> tuple[str, list[str]]:
    prompt = ChatPromptTemplate.from_messages([
        ("system", SYSTEM_PROMPT),
        ("user", "{text}")
    ])

    structured_llm = llm.with_structured_output(TagsAndTitle)
    chain = prompt | structured_llm

    result = chain.invoke({"text": text, "tags": tags})
    return result.title, [tag.name for tag in result.tags]


def generate_title(text: str) -> str:
    try:
        prompt = ChatPromptTemplate.from_messages([
            ("system", SYSTEM_PROMPT_TITLE),
            ("user", "{text}")
        ])
        structured_llm = llm.with_structured_output(Title)
        chain = prompt | structured_llm
        result = chain.invoke({"text": text})
        return result.title
    except Exception as e:
        print(f"Error generating title: {e}")
        return "Untitled"

if __name__ == "__main__":
    text = "I've been reading and thinking a lot about roman dyansty and it's collapse these days. I'm not sure what to think about it."
    tags = [Tag(name="nature"), Tag(name="birds"), Tag(name="park"), Tag(name="walk"), Tag(name="robots"), Tag(name="ai"), Tag(name="chill")]
    tags, title = generate_tags_and_title(text, tags)
    print(tags)
    print(title)