import click
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, TextStreamer
from transformers.generation.logits_process import LogitsProcessorList
import refactx
from refactx.prompt_base import PROMPT_TEMPLATE
from refactx.generate import (
    ConstrainedLogitsProcessor,
    ConstrainedStateList,
    PatternConstrainedState,
    DictIndex,
)
from refactx import patch_model

@click.command()
@click.option("--model", "model_path", default="Qwen/Qwen2.5-3B-Instruct", help="Model name or path")
@click.option("--index", "index_path", default="./indexes/simple_index.txt.gz", help="Path to the index file")
@click.option("--device", default="auto", help="Device to use (e.g. 'auto', 'cuda', 'cpu')")
@click.option("--http-rootcert", required=False, default=None, help="Speficy https certificates file (or false to disable verification)")
def main(model_path, index_path, device, http_rootcert):
    """
    An interactive script to ask questions to the ReFactX model.
    """
    print("Loading tokenizer and model...")
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    if device == "auto":
        model = AutoModelForCausalLM.from_pretrained(model_path, device_map="auto")
    else:
        model = AutoModelForCausalLM.from_pretrained(model_path).to(device)
    patch_model(model)
    model.eval()

    print("Loading index...")
    index = refactx.load_index(
        index_path,
        rootcert=http_rootcert)
    index.set_tokenizer(tokenizer)

    streamer = TextStreamer(tokenizer, skip_prompt=True)

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
    current_prompt_template = PROMPT_TEMPLATE

    while True:
        try:
            question = input("> ")
            if question.startswith("!"):
                parts = question.strip().split(" ", 2)
                cmd = parts[0]
                if cmd == "!exit":
                    break
                elif cmd == "!get":
                    if len(parts) == 1:
                        print(f"Prompt template: {current_prompt_template[0]['content']}")
                        print(f"Generation config: {gen_config}")
                    elif len(parts) >= 2:
                        key = parts[1]
                        if key == "prompt_template":
                            print(f"{key}: {current_prompt_template[0]['content']}")
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
                        current_prompt_template[0]['content'] = val.replace("\\n", "\n")
                        print(f"Updated {key}")
                    elif key in gen_config:
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
                tokenizer, prompt_template=current_prompt_template, question=question
            )
            inputs = tokenizer([prompted_text], return_tensors="pt").to(model.device)

            num_beams = gen_config["num_beams"]
            states = [
                [
                    PatternConstrainedState(
                        pattern="Fact:",
                        tokenizer=tokenizer,
                        cache_index=DictIndex(),
                        subtree_cache=DictIndex(),
                    )
                ]
            ]
            refactx.CONSTRAINED_STATES = ConstrainedStateList(
                states,
                num_beams=num_beams,
                num_batches=1,
            )

            constrained_processor = ConstrainedLogitsProcessor(
                index=index, states=refactx.CONSTRAINED_STATES, tokenizer=tokenizer
            )
            logits_processor_list = LogitsProcessorList([constrained_processor])

            with torch.no_grad():
                model.generate(
                    **inputs,
                    logits_processor=logits_processor_list,
                    streamer=streamer,
                    num_return_sequences=num_beams,
                    use_cache=True,
                    **gen_config,
                )

            print('Triples generated:')
            for i, triple in enumerate(refactx.CONSTRAINED_STATES[0][0].generated_triples):
                print(i, tokenizer.decode(triple), end='\n')

        except (KeyboardInterrupt, EOFError):
            print("\nExiting...")
            break
        except Exception as e:
            print(f"An error occurred: {e}")


if __name__ == "__main__":
    main()
