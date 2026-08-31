import click
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, TextStreamer, AutoProcessor, AutoModelForImageTextToText
from transformers.generation.logits_process import LogitsProcessorList
import refactx
from refactx.generate import (
    CONSTRAINED_STATES,
    get_constrained_logits_processor,
)
from refactx import patch_model
import json
from dotenv import load_dotenv
import os

@click.command()
@click.option("--model", "model_path", default="Qwen/Qwen2.5-3B-Instruct", help="Model name or path")
@click.option("--index", "index_path", default=None, help="Path to the index file (otherwise uses environment INTERACTIVE_INDEX_PATH)")
@click.option("--device", default="auto", help="Device to use (e.g. 'auto', 'cuda', 'cpu')")
@click.option("--http-rootcert", required=False, default=None, help="Speficy https certificates file (or false to disable verification)")
@click.option("--avoid-duplicates", required=False, default=True, help="Speficy whether to avoid generating duplicates or not.")
@click.option("--thinking", is_flag=True, required=False, default=False, help="Enable model thinking mode.")
@click.option("--ignore-case", is_flag=True, default=True, help="Whether to ignore case when matching patterns.")
@click.option("--pattern", default='<fact>', help="Pattern that triggers constrained generation of KB facts.")
@click.option("--torch-dtype", "torch_dtype", default='bfloat16', help="Torch dtype")
@click.option("--prompt", "prompt_path", default="prompts/prompt_qwen36_angular2.json", show_default=True,
              help="Prompt file (.json/.yml/.yaml message list or .txt system prompt).")
def main(model_path, index_path, device, http_rootcert, avoid_duplicates, thinking, ignore_case, pattern, torch_dtype, prompt_path):
    """
    An interactive script to ask questions to the ReFactX model.
    """
    print("Loading tokenizer and model...")
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        eos_token_id = tokenizer.eos_token_id
    except Exception as e:
        print('exc vlm', e)
        tokenizer = AutoProcessor.from_pretrained(model_path)
        eos_token_id = tokenizer.tokenizer.eos_token_id
    if device == "auto":
        try:
            model = AutoModelForCausalLM.from_pretrained(model_path, device_map="auto", dtype=torch_dtype)
        except Exception as e:
            print('exc loading model. maybe a vlm', e)
            model = AutoModelForImageTextToText.from_pretrained(model_path, device_map="auto", dtype=torch_dtype)
    else:
        try:
            model = AutoModelForCausalLM.from_pretrained(model_path).to(device)
        except Exception as e:
            print('exc loading model. maybe a vlm', e)
            model = AutoModelForImageTextToText.from_pretrained(model_path).to(device)

    patch_model(model)
    model.eval()

    print("Loading index...")
    load_dotenv()
    if index_path is None:
        index_path = os.getenv("INTERACTIVE_INDEX_PATH")
    assert index_path is not None, 'ERROR: index must be provided via --index or INTERACTIVE_INDEX_PATH environment variables.'

    index = refactx.load_index(
        index_path,
        rootcert=http_rootcert)
    index.set_tokenizer(tokenizer)

    streamer_tokenizer = getattr(tokenizer, "tokenizer", tokenizer)
    streamer = TextStreamer(streamer_tokenizer, skip_prompt=True)

    current_prompt_path = prompt_path
    current_prompt_template = refactx.load_prompt(current_prompt_path)
    print(f"Loaded prompt from {current_prompt_path}")

    num_beams = 1
    logits_processor_list = get_constrained_logits_processor(
        tokenizer, index, num_beams=num_beams, num_batches=1,
        fact_pattern=pattern, count_pattern="<count>",
        avoid_duplicates=avoid_duplicates,
    )
    constrained_processor = logits_processor_list[0]

    print("Ready to chat!")

    gen_config = {
        "max_new_tokens": 800,
        "do_sample": False,
        "temperature": None,
        "top_k": None,
        "num_beams": 1,
        "top_p": None,
        "min_p": None,
    }

    while True:
        try:
            question = input("> ")
            if question.strip() == "":
                print("Please enter a valid question.")
                continue
            if question.startswith("!"):
                parts = question.strip().split(" ", 2)
                cmd = parts[0]
                if cmd == "!exit":
                    break
                elif cmd == "!reloadprompt":
                    current_prompt_template = refactx.load_prompt(current_prompt_path)
                    print(f"Reloaded prompt from {current_prompt_path}")
                    continue
                elif cmd == "!get":
                    if len(parts) == 1:
                        print(f"Prompt template: {json.dumps(current_prompt_template)}")
                        print(f"Generation config: {gen_config}")
                        print(f"avoid_duplicates: {avoid_duplicates}")
                        print(f"thinking: {thinking}")
                        print(f"ignore_case: {ignore_case}")
                        print(f"pattern: {pattern}")
                    elif len(parts) >= 2:
                        key = parts[1]
                        if key == "prompt_template":
                            print(f"{key}: {json.dumps(current_prompt_template)}")
                        elif key == "avoid_duplicates":
                            print(f"{key}: {avoid_duplicates}")
                        elif key == "thinking":
                            print(f"{key}: {thinking}")
                        elif key == "ignore_case":
                            print(f"{key}: {ignore_case}")
                        elif key == "pattern":
                            print(f"{key}: {pattern}")
                        elif key in gen_config:
                            print(f"{key}: {gen_config[key]}")
                        else:
                            print(f"Unknown key: {key}")
                    continue
                elif cmd == "!set":
                    if len(parts) < 3:
                        print("Usage: !set <key> <value>")
                        continue
                    key = parts[1]
                    val = parts[2]
                    if key == "prompt_template":
                        current_prompt_template = refactx.load_prompt(val)
                        current_prompt_path = val
                        print(f"Updated {key} to {val}")
                    elif key in gen_config or key in ["avoid_duplicates", "ignore_case", "thinking", "pattern"]:
                        if val.lower() == "none":
                            val = None
                        elif val.lower() == "true":
                            val = True
                        elif val.lower() == "false":
                            val = False
                        else:
                            try:
                                val = int(val)
                            except ValueError:
                                try:
                                    val = float(val)
                                except ValueError:
                                    pass
                        if key == "avoid_duplicates":
                            avoid_duplicates = bool(val)
                        elif key == "thinking":
                            thinking = bool(val)
                        elif key == "ignore_case":
                            ignore_case = bool(val)
                        elif key == "pattern":
                            pattern = str(val)
                        else:
                            gen_config[key] = val
                        print(f"Updated {key} to {val}")
                    else:
                        print(f"Unknown key: {key}")
                    continue
                else:
                    print(f"Unknown command: {cmd}")
                    continue

            if question.lower() in ["exit", "quit"]:
                break

            prompted_text = refactx.apply_prompt_template(
                tokenizer, prompt_template=current_prompt_template, question=question, enable_thinking=thinking
            )
            tokenizer_for_inputs = getattr(tokenizer, "tokenizer", tokenizer)
            inputs = tokenizer_for_inputs([prompted_text], return_tensors="pt").to(model.device)

            num_beams = gen_config["num_beams"]
            # Keep the processor stable; reset only per-request state.
            constrained_processor.reset_states(
                num_beams=num_beams, num_batches=1,
            )

            with torch.no_grad():
                model.generate(
                    **inputs,
                    logits_processor=logits_processor_list,
                    streamer=streamer,
                    num_return_sequences=num_beams,
                    use_cache=True,
                    eos_token_id=eos_token_id,
                    **gen_config,
                )

            print('Triples generated:')
            state = CONSTRAINED_STATES.states[0][0]
            for i, triple in enumerate(state.generated_triples):
                print(i, streamer_tokenizer.decode(triple), end='\n')


        except (KeyboardInterrupt, EOFError):
            print("\nExiting...")
            break
        except Exception as e:
            print(f"An error occurred: {e}")


if __name__ == "__main__":
    main()
