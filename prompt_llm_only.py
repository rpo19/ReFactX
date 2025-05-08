FACT_COMMAND = ''
PROMPT_DESCRIPTION = ''
ASSUME_NOT = ''
TOO_LONG = ''

SYSTEM_PROMPT = '''You are a helpful question-answering assistant.
'''

PROMPT_ANSWER_TYPE = '''
    You detemine the kind of answer you are asked. It can be a yes/no, a single entity, or a list of entities. Pay attention to the questions whose answer is a list of entities (e.g. Which countries share a border with Spain?): you need to find all the answer entities and include them all in the final answer.
'''

DO_NOT_KNOW = '''

If you don't know the answer, you stop and you reply: "I don't know.".
'''


PROMPT_PROCESS_TEMPLATE = '''{system_prompt}
The process to answer questions:

    You receive an input question.

    You determine the reasoning path needed to answer the question based on the information available.
{answer_type}
    You conclude with a concise answer that depending on the question can be a yes/no, a single entity, or a list of entities. Pay attention to the questions whose answer is a list of entities.{do_not_know}
You must always follow these instructions precisely and ensure your responses adhere strictly to this prompt.'''

DEFAULT_FEW_SHOT_GETTER = 'llmonly.mh_slumdog,llmonly.comp_mounts'

# spain could also be "how many"
FEW_SHOW_EXAMPLES = {
    'llmonly': {
        'comp_mounts': [{
        'role': 'user',
        'content': 'Which mountain is taller between Mont Blanc and Mount Rainier?'
    },
    {
        'role': 'assistant',
        'content': '''Reasoning: I need to provide the height of Mont Blanc and the height of Mount Rainier, then I need to compare the two heights. The final answer will be the taller mountain.
Mont Blanc is 4,808 meters tall, while Mount Rainier is 4,389 meters, so Mont Blanc is taller than Mount Rainier. The answer is Mont Blanc.

Answer: Mont Blanc.'''
    }],
        'mh_slumdog': [
        {
        'role': 'user',
        'content': 'When was the director of Slumdog Millionaire born?'
    },
    {
        'role': 'assistant',
        'content': '''Reasoning: To answer this question, I need to find who is the director of Slumdog Millionaire and then his birth date. The final answer will be his birth date.
The director of Slumdog Millionaire is Danny Boyle.
Danny Boyle was born on October 20, 1956.

Answer: October 20, 1956.'''
    },
    ]
    }
}

# e.g. get_few_shot('descr.asnot_clinton,no_descr.comp_mounts,...')
def get_few_shot(string_getter, few_shot_pool=FEW_SHOW_EXAMPLES):
    few_shot_examples = []
    splitted = string_getter.split(',')
    for getter in splitted:
        getter = getter.strip()
        lv1, lv2 = getter.split('.')
        eg = few_shot_pool[lv1][lv2]
        few_shot_examples.append(eg)

    return few_shot_examples

def get_prompt_template(system_prompt=SYSTEM_PROMPT,
                        answer_type=PROMPT_ANSWER_TYPE,
                        description=PROMPT_DESCRIPTION,
                        do_not_know=DO_NOT_KNOW,
                        assume_not=ASSUME_NOT,
                        too_long=TOO_LONG,
                        process_template=PROMPT_PROCESS_TEMPLATE,
                        fact_command=FACT_COMMAND,
                        custom_prompt='',
                        few_shot_examples=None):
    if few_shot_examples is None:
        few_shot_examples = get_few_shot(DEFAULT_FEW_SHOT_GETTER)

    prompt_template = []

    system_prompt=system_prompt.format(fact_command=fact_command)
    answer_type=answer_type.format(fact_command=fact_command)
    description=description.format(fact_command=fact_command)
    do_not_know=do_not_know.format(fact_command=fact_command)
    assume_not=assume_not.format(fact_command=fact_command)
    too_long=too_long.format(fact_command=fact_command)

    process = process_template.format(system_prompt=system_prompt, answer_type=answer_type, fact_command=fact_command, description=description, do_not_know=do_not_know, assume_not=assume_not, too_long=too_long, custom_prompt=custom_prompt)

    prompt_template.append({
        'role': 'system',
        'content': process
    })

    for example in few_shot_examples:
        assert len(example) == 2
        for message in example:
            message['content'] = message['content'].format(fact_command=fact_command)
            prompt_template.append(message)

    return prompt_template

PROMPT_TEMPLATE = get_prompt_template()