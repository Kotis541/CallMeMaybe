from llm_sdk import llm_sdk as llm

def main():
    try:
        print("trying to initilized ..")
        model = llm.Small_LLM_Model()
        print(f"model: {model}")

        input_id = model.encode("Hello")
        print(f"inputed id: {input_id}")

        decode_id = model.decode(input_id)
        print(f"decoded id: {decode_id}")

    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    main()
