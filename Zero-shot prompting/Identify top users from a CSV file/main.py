from openai import OpenAI

# DO NOT modify the SYSTEM_PROMPT!
SYSTEM_PROMPT = """You are a knowledgeable and experienced Python Developer. 
You specialize in writing idiomatic code that is valid, follows best practices and impresses Senior Developers.
In the OUTPUT you MUST ONLY provide the code itself WITHOUT any additional information or meta text."""


def main():
    # Students would enter their OpenAI API key here
    openai_api_key = "ENTER YOUR OPENAI API KEY HERE"
    client = OpenAI(api_key=openai_api_key)

    # Students would write the prompt to generate the code here
    user_prompt = """WRITE YOUR PROMPT HERE
...
..."""

    # DO NOT DELETE THIS LINE, IT IS USED TO CHECK YOUR PROMPT!
    print_gpt_response(client, openai_api_key, SYSTEM_PROMPT, user_prompt)


# DO NOT DELETE OR MODIFY THE CODE WITHIN THE print_gpt_response FUNCTION!
def print_gpt_response(client, openai_api_key, system_prompt, user_prompt):
    chat_completion = client.chat.completions.create(
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ],
        model="gpt-3.5-turbo-1106",
        temperature=0,
        seed=42
    )
    response_content = chat_completion.choices[0].message.content
    print(f"API_KEY:{openai_api_key}")
    print(response_content)


if __name__ == "__main__":
    main()
