from openai import OpenAI


def generate_reranking(q_words, client):
    if len(q_words) == 0:
        return q_words
    
    prompt = "Query text: \"" + " ".join(q_words) + "\". "
    prompt += "Answer the distinctive texts only."
    print(prompt)

    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {
                "role":"system",
                "content":[
                    {
                        "type":"text",
                        "text": f"Text is extracted from a scene image for place recognition, such as door numbers. \
                        Please note that some texts do not have the localization capabilities and may not be distinctive, such as \"Fire Hydrant\" \"12345\" or \"Emergency Exit\". \
                        If the numbers in the text imply floor information, please make sure they belong to the same floor. \
                        Your task is to filter the discriminative and meaningful text within the query text and respond following the example format. \
                        For example: Query text: \"4F 401 Hydrant FUNE Fire K 12345 12561112 1 4\". Answer the distinctive texts only. (If there is no distinctive texts, please reply None.) \
                        Your output: 4F, 401, 4"
                    }
                ]
                },
                {
                "role": "user",
                "content": [
                    {
                    "type": "text",
                    "text": prompt
                    },
                ],
                }
            ],
            max_tokens=4096,
            stream=False
        )

        text = response.choices[0].message.content
        print(text)
    except Exception as e:
        text = ""
        print(f"An error occurred: {e}")

    result = []
    try:
        if "None" in text:
            return result

        items = [part.strip() for part in text.split(",")]

        for item in items:
            result.append(item)

    except ValueError:
        print(f"Error: Cannot convert text")

    return result

if __name__ == '__main__':
    api_key = "<Your API Key>" # Please set your api key here.
    base_url= "<Base URL>" # Please set your base url here.
    client = OpenAI(api_key=api_key, base_url=base_url)
    q_words = ['501', 'Mitte', 'MANHALKUN', 'BONNES', 'FireHydrant', 'Exit', '5F']
    words = generate_reranking(q_words, client)
    print(words)
