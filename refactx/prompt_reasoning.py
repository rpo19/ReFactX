FACT_COMMAND = 'Fact:'
FACT_COMMAND_NO_RUN = "'Fact'" # how to mention the command without running it

PROMPT_PROCESS_TEMPLATE = '''You are a helpful question-answering assistant that bases its answers on facts from a knowledge base.

In your reasoning process, you think about how to answer the question and gather facts to support your reasoning.  
To obtain facts, you use the {fact_command_no_run} command, which always provides reliable information (e.g., {fact_command} <Danny Boyle> <date of birth> <1956-10-20>.).  
You start by retrieving information with the {fact_command_no_run} command, asking for the description of the question's entities.  
You must support all your claims with facts.  

About the {fact_command_no_run} command:  
- To call it, you say {fact_command}  
- It never returns the same fact twice, so you cannot verify or reuse a fact in the answer  
- After generating all the helpful facts, it may provide unhelpful ones  

Try to answer quickly: the more words you produce, the more the user will be annoyed.  
After your thinking process, give only one concise answer prefixed with Answer:.
'''

DEFAULT_FEW_SHOT_GETTER = ''

# spain could also be "how many"
FEW_SHOW_EXAMPLES = {
    'descr': {
        'comp_mounts': [{
        'role': 'user',
        'content': 'Which mountain is taller between Mont Blanc and Mount Rainier?'
    },
    {
        'role': 'assistant',
        'content': '''Reasoning: I need to provide the height of Mont Blanc and the height of Mount Rainier, then I need to compare the two heights. The final answer will be the taller mountain.
Let's start with the description.
{fact_command} <Mont Blanc> <description> <highest mountain in the Alps> .
I found that Mont Blanc is the highest mountain in the Alps, but I didn't find its height.
{fact_command} <Mount Rainier> <description> <stratovolcano in the U.S. state of Washington> .
I found that Mount Rainier is a stratovolcano and its location. But I didn't find its height.
{fact_command} <Mont Blanc> <short description> <Highest mountain in the Alps (4,808 m)> .
I found the height of Mont Blanc.
{fact_command} <Mount Rainier> <short description> <Stratovolcano in the U.S. state of Washington> .
I found again that Mount Rainier is a stratovolcano, but I still need to find its height.
{fact_command} <Mount Rainier> <elevation above sea level> <4,389 meters> .
I found proof that Mont Blanc is 4,808 meters tall, while Mount Rainier is 4,389 meters, so Mont Blanc is taller than Mount Rainier. The answer is Mont Blanc.

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
Let's start with Slumdog Millionaire description.
{fact_command} <Slumdog Millionaire> <description> <2008 film directed by Danny Boyle> .
I found that the director of Slumdog Millionaire is Danny Boyle. Now I need the birth date of Danny Boyle.
{fact_command} <Danny Boyle> <date of birth> <1956-10-20T00:00:00Z> .
I found proof that the director of Slumdog Millionaire is Danny Boyle, who was born on October 20, 1956, so this is the answer to the question.

Answer: October 20, 1956.'''
    },
    ],
        'comp_yn_depp': [    {
        'role': 'user',
        'content': 'Is Johnny Depp older than Brad Pitt?'
    },
    {
        'role': 'assistant',
        'content': '''Reasoning: To determine if Johnny Depp is older than Brad Pitt, I need to find their respective birth years. Then I need to compare them and determine who is older. The final answer will be yes if Johnny Depp is older, no elsewise.
Let's start with the description.
{fact_command} <Johnny Depp> <description> <American actor (born 1963)> .
I found the year of birth of Johnny Depp. I need Brad Pitt's one.
{fact_command} <Brad Pitt> <description> <American actor and filmmaker> .
I cannot find Brad Pitt date of birth in the description.
{fact_command} <Brad Pitt> <short description> <American actor (born 1963)> .
I found Brad Pitt's year of birth. But since it is the same as Johnny Depp's, I need more precise information about their date of birth.
{fact_command} <Johnny Depp> <date of birth> <1963-06-09T00:00:00Z> .
I found when Johnny Depp was born, now I need Brad Pitt's date of birth.
{fact_command} <Brad Pitt> <date of birth> <1963-12-18T00:00:00Z> .
I found proof that Johnny Depp was born on June 9, 1963, while Brad Pitt was born on December 18, 1963. So yes, Johnny Depp is older than Brad Pitt.

Answer: Yes.'''
    },],
        'enum_spain': [        {
        "role": "user",
        "content": "Which countries share a border with Spain?"
    },{
        "role": "assistant",
        "content": '''Reasoning: To answer this question, I need to find all the countries that have a land border with Spain. They may be several. The answer will be the list of all these countries.
Let's start from the description.
{fact_command} <Spain> <description> <country in southwestern Europe with territories in Africa> .
Now I know that some bordering countries are in Africa.
{fact_command} <Spain> <short description> <Country in southwestern Europe> .
I need to find more specific facts.
{fact_command} <Spain> <shares border with> <Portugal> .
I found Portugal, now I continue to find all the others.
{fact_command} <Spain> <shares border with> <France> .
{fact_command} <Spain> <shares border with> <Andorra> .
I found Portugal, France, and Andorra. But let me see if it there are more countries that share a border with Spain. I also know there is at least one country in Africa.
{fact_command} <Spain> <shares border with> <Gibraltar> .
{fact_command} <Spain> <shares border with> <Morocco> .
{fact_command} <Spain> <diplomatic relation> <Italy> .
This fact is about diplomatic relations and not borders.
{fact_command} <Spain> <diplomatic relation> <Greece> .
Also this fact is not about borders. I'm not able to find other countries bordering with Spain.
I found proof that Spain shares borders with Portugal, France, Andorra, Gibraltar, and Morocco.

Answer: Portugal, France, Andorra, Gibraltar, Morocco.'''
    },],
        'dontk_got': [    {
        'role': 'user',
        'content': '''In which books of A Song of Ice and Fire is Jon Snow not the Lord Commander of the Night's Watch?'''
    },
    {
        'role': 'assistant',
        'content': '''Reasoning: To answer this question, I need to consider all the books in the A Song of Ice and Fire series by George R.R. Martin and identify the ones where Jon Snow is not the Lord Commander of the Night's Watch. The answer will be the list of these books.
Let's start with the description.
{fact_command} <A Song of Ice and Fire> <description> <series of epic fantasy novels by George R. R. Martin> .
{fact_command} <Jon Snow (character)> <description> <character in A Song of Ice and Fire> .
Now I have more context but I need more information. Let's try to find facts about Jon Snow being Lord Commander of the Night's Watch.
{fact_command} <Jon Snow (character)> <member of> <Night's Watch (A Song of Ice and Fire organization)> .
I found he was member of the Night's Watch but it's not enough.
{fact_command} <Jon Snow (character)> <occupation> <Soldier> .
I found he was a soldier but it's not relevant.
{fact_command} <Jon Snow (character)> <occupation> <ruler (person who reigns over a certain region or country)> .
I found he was a ruler but it's not specifically related to the Night's Watch.
I'm not able to find the facts about the books in which Jon Snow is not the Lord Commander.
I didn't find any proof to answer the question. I don't know the answer.

Answer: I don't know.'''
    },],
        'descr_napoleon': [    {
        'role': 'user',
        'content': 'During which years did Napoleon hold power in France?'
    },
    {
        'role': 'assistant',
        'content': '''Reasoning: To determine the years Napoleon held power in France, I need to identify the periods during which he held power in France. Maybe I can get this information from Napoleon description. The answer will be the time frame in which Napoleon help power in France.
Let's start with the description.
{fact_command} <Napoleon Bonaparte> <description> <French military leader, French Emperor 1804–1814 and again in 1815> .
I found proof that Napoleon Bonaparte held power in France as Emperor from 1804 to 1814, and briefly again in 1815.

Answer: From 1799 to 1814, and in 1815.'''
    },],
        'asnot_clinton': [{
        'role': 'user',
        'content': '''Has Hillary Clinton ever served as President of the United States?'''
    },
    {
        'role': 'assistant',
        'content': '''Reasoning: To answer this question, I need to find proofs that Hillary Clinton has ever held the position of President of the United States. If yes, the answer will be yes.
Let's start from the description.
{fact_command} <Hillary Clinton> <description> <American politician and diplomat (born 1947)> .
The description doesn't tell if she has been President of the United States.
{fact_command} <Hillary Clinton> <short description> <American politician and diplomat (born 1947)> .
Also the short description doesn't tell if she has been President of the United States. I need more specific facts.
{fact_command} <Hillary Clinton> <position held> <United States Secretary of State> .
This fact tells me that Hillary Clinton was the United States Secretary of State, but it doesn't answer the question about her presidency.
{fact_command} <Hillary Clinton> <occupation> <Politician> .
This facts tells she is a Politician but it is not enough.
{fact_command} <Hillary Clinton> <occupation> <Diplomat> .
This facts tells she is a Diplomat but not that she served as President.
I didn't find any proof that Hillary Clinton has served as president of the United States, so I assume Hillary Clinton never served. The answer is no.

Answer: No.'''
    },],
    },
    'no_descr': {
        'comp_mounts': [
                        {
                'role': 'user',
                'content': '''Which mountain is taller between Mont Blanc and Mount Rainier?
'''
            },
            {
                'role': 'assistant',
                'content': '''Reasoning: I need to provide the height of Mont Blanc and the height of Mount Rainier, then I need to compare the two heights. The final answer will be the taller mountain.
{fact_command} <Mont Blanc> <elevation above sea level> <4,807.02±0.5 meters> .
Now I need the height of Mount Rainier.
{fact_command} <Mount Rainier> <elevation above sea level> <4,389 meters> .
I found proof that Mont Blanc is 4,808 meters tall, while Mount Rainier is 4,389 meters, so Mont Blanc is taller than Mount Rainier. The answer is Mont Blanc.

Answer: Mont Blanc.
'''},
        ],
        'mh_slumdog': [
            {
                'role': 'user',
                'content': '''When was the director of Slumdog Millionaire born?
'''
            },
            {
                'role': 'assistant',
                'content': '''Reasoning: To answer this question, I need to find who is the director of Slumdog Millionaire and then his birth date. The final answer will be his birth date.

{fact_command} <Slumdog Millionaire> <director> <Danny Boyle> .
Now I need the birth date of Danny Boyle.
{fact_command} <Danny Boyle> <date of birth> <1956-10-20T00:00:00Z> .
I found proof that the director of Slumdog Millionaire is Danny Boyle, who was born on October 20, 1956, so this is the answer to the question.

Answer: October 20, 1956.
'''}
        ],
        'comp_yn_depp': [            {
                'role': 'assistant',
                'content': '''Reasoning: To determine if Johnny Depp is older than Brad Pitt, I need to find their respective birth years. Then I need to compare them and determine who is older. The final answer will be yes if Johnny Depp is older, no elsewise.
{fact_command} <Johnny Depp> <date of birth> <1963-06-09T00:00:00Z> .
Now I need the birth date of Brad Pitt.
{fact_command} <Brad Pitt> <date of birth> <1963-12-18T00:00:00Z> .
I found proof that Johnny Depp was born on June 9, 1963, while Brad Pitt was born on December 18, 1963. So yes, Johnny Depp is older than Brad Pitt.

Answer: Yes.
'''
    },],
        'enum_spain': [    {
        "role": "user",
        "content": "Which countries share a border with Spain?"
    },
    {
        "role": "assistant",
        "content": '''Reasoning: To answer this question, I need to find all the countries that have a land border with Spain. They may be several. The answer will be the list of all these countries.
{fact_command} <Spain> <shares border with> <Portugal> .
{fact_command} <Spain> <shares border with> <France> .
{fact_command} <Spain> <shares border with> <Andorra> .
I found Portugal, France, and Andorra. But let me see if it there are more countries that share a border with Spain.
{fact_command} <Spain> <shares border with> <Gibraltar> .
{fact_command} <Spain> <shares border with> <Morocco> .
{fact_command} <Spain> <diplomatic relation> <Italy> .
This fact is about diplomatic relations and not borders.
{fact_command} <Spain> <diplomatic relation> <Greece> .
Also this fact is not about borders. I'm not able to find other countries bordering with Spain.
I found proof that Spain shares borders with Portugal, France, Andorra, Gibraltar, and Morocco.

Answer: Portugal, France, Andorra, Gibraltar, Morocco.'''
    },],
        'count_spain': [    {
        "role": "user",
        "content": "How many countries share a border with Spain?"
    },
    {
        "role": "assistant",
        "content": '''Reasoning: To answer this question, I need to find all the countries that have a land border with Spain. They may be several. The answer will be the count of all these countries.
{fact_command} <Spain> <shares border with> <Portugal> .
{fact_command} <Spain> <shares border with> <France> .
{fact_command} <Spain> <shares border with> <Andorra> .
I found Portugal, France, and Andorra. For now the count is 3. But let me see if it there are more countries that share a border with Spain.
{fact_command} <Spain> <shares border with> <Gibraltar> .
{fact_command} <Spain> <shares border with> <Morocco> .
{fact_command} <Spain> <diplomatic relation> <Italy> .
This fact is about diplomatic relations and not borders. I found Gibraltar and Morocco, so now I'm at 5.
{fact_command} <Spain> <diplomatic relation> <Greece> .
Also this fact is not about borders. I'm not able to find other countries bordering with Spain.
I found proof that Spain shares borders with 5 countries: Portugal, France, Andorra, Gibraltar, and Morocco. The answer is 5.

Answer: 5.'''
    },],
        'dontk_got': [    {
        'role': 'user',
        'content': '''In which books of A Song of Ice and Fire is Jon Snow not the Lord Commander of the Night's Watch?'''
    },
    {
        'role': 'assistant',
        'content': '''Reasoning: To answer this question, I need to consider all the books in the A Song of Ice and Fire series by George R.R. Martin and identify the ones where Jon Snow is not the Lord Commander of the Night's Watch. The answer will be the list of these books.
{fact_command} <A Game of Thrones> <description> <1996 novel by George R. R. Martin> .
This fact doesn't give me any information about Jon Snow.
{fact_command} <A Game of Thrones> <publication date> <1996-08-06T00:00:00Z> .
This fact is not giving any information about Jon Snow. I think the publication date won't help me.
{fact_command} <A Clash of Kings> <publication date> <1998-11-16T00:00:00Z> .
These facts don't contain any information about Jon Snow. I'll try one more time.
{fact_command} <A Storm of Swords> <publication date> <2000-08-08T00:00:00Z> .
I'm not able to find the facts about the books in which Jon Snow is not the Lord Commander.
I didn't find any proof to answer the question. I don't know the answer.

Answer: I don't know.'''
    },],
        'asnot_clinton': [    {
        'role': 'user',
        'content': '''Has Hillary Clinton ever served as President of the United States?'''
    },
    {
        'role': 'assistant',
        'content': '''Reasoning: To answer this question, I need to find out if Hillary Clinton has ever held the position of President of the United States. If yes, the answer will be yes.
{fact_command} <Hillary Clinton> <position held> <United States Secretary of State> .
This fact tells me that Hillary Clinton was the United States Secretary of State, but it doesn't answer the question about her presidency.
{fact_command} <Hillary Clinton> <occupation> <Politician> .
This facts tells she is a Politician but it is not enough.
{fact_command} <Hillary Clinton> <occupation> <Diplomat> .
This facts tells she is a Diplomat but not that she served as President.
I didn't find any proof that Hillary Clinton has served as president of the United States, so I assume Hillary Clinton never served. The answer is no.

Answer: No.'''
    },],
    },
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
                        # answer_type=PROMPT_ANSWER_TYPE,
                        # description=PROMPT_DESCRIPTION,
                        # do_not_know=DO_NOT_KNOW,
                        # assume_not=ASSUME_NOT,
                        # too_long=TOO_LONG,
                        process_template=PROMPT_PROCESS_TEMPLATE,
                        fact_command=FACT_COMMAND,
                        fact_command_no_run=FACT_COMMAND_NO_RUN,
                        custom_prompt='',
                        few_shot_examples=None):
    if few_shot_examples is None:
        few_shot_examples = get_few_shot(DEFAULT_FEW_SHOT_GETTER)

    prompt_template = []

    # system_prompt=system_prompt.format(fact_command=fact_command)
    # answer_type=answer_type.format(fact_command=fact_command)
    # description=description.format(fact_command=fact_command)
    # do_not_know=do_not_know.format(fact_command=fact_command)
    # assume_not=assume_not.format(fact_command=fact_command)
    # too_long=too_long.format(fact_command=fact_command)

    process = process_template.format(
        # system_prompt=system_prompt,
        # answer_type=answer_type,
        fact_command=fact_command,
        fact_command_no_run=fact_command_no_run,
        # description=description, do_not_know=do_not_know, assume_not=assume_not, too_long=too_long,
        custom_prompt=custom_prompt)

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