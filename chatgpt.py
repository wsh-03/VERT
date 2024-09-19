from openai import OpenAI
import json

OPENAI_API_KEY = ""
model_type = "gpt-3.5-turbo"

response_history = []
prompt_history = []


def prompt2gpt(prompt):
    client = OpenAI(api_key=OPENAI_API_KEY)
    response = client.chat.completions.create(
        model=model_type,
        messages=[{"role": "user", "content": prompt}],
        temperature=0
        # max_tokens=500,
    )
    # reply_content = response.choices[0].message.content
    # print(reply_content, end="")
    # prompt_history.append({"role": "user", "content": prompt})
    # response_history.append({"role": "system", "content": reply_content})
    
    return response.choices[0].message.content


def read_file(path2file):
    try:
        with open(path2file, 'r', encoding='utf-8') as file:
            return file.read()
    except FileNotFoundError:
        return f"File not Found, Check the {path2file}"
    except Exception as e:
        return f"Error Occurred {e}"


path2file = "/home/wsh-v22/test/c2rust_test/test_file/c_transcoder/ADD_1_TO_A_GIVEN_NUMBER/ADD_1_TO_A_GIVEN_NUMBER.c"
text = f"""
\nTranspile above C code into Rust code. \
\nRust code must obey following rules:
1 - Use the same function name, same argument types and return types. \
2 - Make sure it includes all imports, uses safe rust, and compiles on its own. \
3 - Give only code, no comments and no main function. \
4 - Convert i32 types to f32 if necessary. \
5 - Use mut variables if necessary. \
\n Provide your answer in JSON format with following keys: \
rust_code<only contain raw rust code>.
"""
prompt = read_file(path2file) + text
test = prompt2gpt(prompt)

print("Raw output from GPT: \n", test)

try:
    json_format = json.loads(test,strict=False)
    test = json_format["rust_code"]
    print("test: \n", test)
except json.JSONDecodeError as e:
    print("JSON decode error:", e)
