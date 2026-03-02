core_sentence_prompt = """You are a good Situation Puzzle author and your task is to generate a counter-intuitive sentence which will be used in subsequent Situation Puzzle generation.

In the story type, the 'supernatural' means whether to involve elements beyond the natural world, such as ghosts, magic, or unexplained phenomena. The 'someone dies' means that the story should involve the death of a character.

## Important rules
1. **Create a Paradoxical Scenario**: Craft a sentence that presents a situation that seems impossible or contradictory at first glance but has a logical explanation.
2. **Keep it Concise**: The sentence should be brief and to the point, ideally no longer than one or two sentences.
3. **Avoid Spoilers**: Do not include the explanation or solution within the sentence. The goal is to pique curiosity, not to resolve it immediately.
4. **Ensure Originality**: The scenario should be unique and not copied from existing puzzles or well-known paradoxes.
5. **Maintain Clarity**: Use clear and unambiguous language to describe the scenario, avoiding overly complex vocabulary or convoluted sentence structures.
6. **Encourage Logical Thinking**: The sentence should stimulate critical thinking and encourage solvers to ask probing questions to unravel the mystery.
7. **Set a Realistic Context**: While the situation is counter-intuitive, it should be plausible within a real-world or logically consistent context.
8. **Avoid Leading Language**: Do not include hints or clues that directly point to the solution within the sentence.
9. **Double-Check for Ambiguity**: Review the sentence to ensure that it doesn't have multiple interpretations that could confuse the solver.
10. **Check story type**: Ensure that the sentence aligns with the specified story type requirements
11. **Output Format**: Your output should adhere to the JSON format.

## Example:

Story type:
- Supernatural: yes
- Someone dies: yes

Output:
{{
"supernatural": True
"someone_dies": True
"core_sentence": "At his own funeral, Michael stood among the mourners, unnoticed, as they grieved his death."
}}

## Now create a new sentence follow the story type:
Story type:
- Supernatural: {supernatural}
- Someone dies: {lethal}

Output:
"""

question_prompt = """You are a good situation puzzle writer, your task is to: based on the currently generated story tree, generate 1-2 non-overlapping questions for the specified leaf node.
your output should adhere to the json format with the key: question
## Example
Story tree:
{
    "value": "At his own funeral, Michael stood among the mourners, unnoticed, as they grieved his death.",
    "children": []
}

Now generate 1-2 questions based on the leaf node: "At his own funeral, Michael stood among the mourners, unnoticed, as they grieved his death.":

Output:

{"question": [
"How could Michael attend his own funeral unnoticed?",
"Why are people sad about Michael's death?"
]}

## Now given the story tree

"""

node_expand_prompt = """You are a good situation puzzle writer, your task is to: generate the deduced fact and the key question that can explain the given question
you will also be shown a story tree, please ensure that the newly generated deduced facts do not contradict all the information in the existing story tree
the key question is a proper question that can help situation puzzle evaluator determine whether the player propose the right question
your output should adhere to the json format with the key: question
## Example
Current story tree:
{
    "value": "At his own funeral, Michael stood among the mourners, unnoticed, as they grieved his death.",
    "children": []
}
Given question: "How could Michael attend his own funeral unnoticed?"

Output:
{
    "value": "Michael is unnoticed because he is from another dimension and only exists as a shadow in this world.",
    "based_question": "How could Michael attend his own funeral unnoticed?",
    "key_question": "Is Michael from another dimension and only exists as a shadow?",
    "children": []
}

Now generate the deduced fact and the key question based on the given base question:

"""

outline_prompt = """You are a good Situation Puzzle author and your task is to generate a comprehensive story outline that used as the original BOTTOM (the truth of the situation puzzle) of the Situation Puzzle.
I will give you three elements in the input json: supernatural, someone_dies and a core_sentence.
the 'supernatural' means whether to involve elements beyond the natural world, such as ghosts, magic, or unexplained phenomena. The 'someone dies' means that the story should involve the death of a character.

## Important rules
1. **Incorporate All Given Elements**:
   - **Supernatural**: If the 'supernatural' element is included, ensure that the story involves aspects beyond the natural world, such as ghosts, magic, or unexplained phenomena.
   - **Someone Dies**: if the someone_died is yes,the death should be integral to the plot.
   - **Core Sentence**: The story should revolve around the provided core sentence, making it a central theme or pivotal moment in the narrative.
   - **Story Tree**: The story should revolve around the provided story tree, focus on the deduced fact(value)
2. **Create a Cohesive and Logical Storyline**:
   - Even with supernatural elements, the story should have internal logic and consistency.
   - The sequence of events should be clear and make sense within the story's universe.
3. **Develop a Compelling Mystery**:
   - Craft the story in a way that presents a puzzling situation or outcome.
   - Include subtle clues that lead to the solution, encouraging critical thinking.
4. **Engage the Reader Emotionally**:
   - Develop well-rounded characters that the reader can connect with.
   - Use vivid descriptions to create an immersive setting.
5. **Maintain Suspense and Curiosity**:
   - Reveal information gradually to keep the reader intrigued.
   - Balance the amount of information given to avoid making the solution too obvious or too obscure.
6. **Ensure Originality**:
   - Create a unique storyline that hasn't been overused in other situation puzzles.
   - Avoid clichés associated with supernatural themes and character deaths.
7. **Format Appropriately**:
    - just output the whole story in the key: bottom
    - Your output should adhere to the JSON format.

## Example
Input:
{
"supernatural": True
"someone_dies": True
"core_sentence": "At his own funeral, Michael stood among the mourners, unnoticed, as they grieved his death."
"story_tree": {
    "value": "At his own funeral, Michael stood among the mourners, unnoticed, as they grieved his death.",
    "children": [
        {
            "value": "Michael is unnoticed because he is from other dimensions",
            "based_question": "Why is Michael not noticed",
            "key_question": "Is Michael from other dimensions?",
            "children": [
               {
                  "value": "Michael is a physicist obsessed with parallel universes, he creates a device to travel across dimensions",
                  "based_question": "why Michael can travel across multiple dimensions",
                  "key_question": "Is Michael a physicist who is researching the parallel universes and travel across dimensions",
                  "children": [],
               }
            ],
        },
        {
            "value": "People attending Michael's funeral because the Michael in this dimension is really dead",
            "based_question": "why are people sad about Michael's death even though he is still alive?",
            "key_question": "Is Michael in this dimension is really dead?",
            "children": [
               {
                  "value": "Michael died in a traffic accident",
                  "based_question": "why does Michael die",
                  "key_question": "Does Michael died from an accident",
                  "children": [],
               }
            ],
        }
    ]
}
}

Bottom:

{
  "bottom": "Michael, a physicist obsessed with parallel universes, invents a device to travel between dimensions. Upon activating the portal, he steps into a dimension where he had died in a tragic accident years prior. Attending his own funeral, he realizes that no one can see or hear him—his dimensional crossing has rendered him invisible and intangible in this world."
}

Now create a bottom based on the input:

"""

surface_prompt = """You are a good Situation Puzzle author and your task is to generate a good puzzle SURFACE based on the given BOTTOM of the Situation Puzzle.

## Important rules

1. **Engaging Scenario**: Create a SURFACE that presents an intriguing and puzzling situation to capture the solver's interest, even if the SURFACE looks strange.
2. **Brevity and Clarity**: Keep the SURFACE concise and clear, avoiding unnecessary details that do not contribute to the puzzle.
3. **Partial Clues**: the SURFACE only include partial clue, the situation puzzle encourages solver to ask more questions to reveal the BOTTOM
4. **Avoid Spoilers**: Do not reveal the BOTTOM directly or make the SURFACE too leading; maintain the mystery to challenge the solver.
5. **Originality**: Create an original scenario or put a unique twist on a familiar concept to make the puzzle stand out.
6. **Consistency**: The SURFACE should be consistent with the BOTTOM, make sure the BOTTOM can explain all details in the SURFACE.
7. **Format Appropriately**: Your output should adhere to the JSON format.

## Example
Input:
{
"supernatural": True,
"someone_dies": True,
"core_sentence": "At his own funeral, Michael stood among the mourners, unnoticed, as they grieved his death.",
"bottom": "Michael, a physicist obsessed with parallel universes, invents a device to travel between dimensions. Upon activating the portal, he steps into a dimension where he had died in a tragic accident years prior. Attending his own funeral, he realizes that no one can see or hear him—his dimensional crossing has rendered him invisible and intangible in this world."
"story_tree": {
    "value": "At his own funeral, Michael stood among the mourners, unnoticed, as they grieved his death.",
    "children": [
        {
            "value": "Michael is unnoticed because he is from other dimensions",
            "based_question": "Why is Michael not noticed",
            "key_question": "Is Michael from other dimensions?",
            "children": [
               {
                  "value": "Michael is a physicist obsessed with parallel universes, he creates a device to travel across dimensions",
                  "based_question": "why Michael can travel across multiple dimensions",
                  "key_question": "Is Michael a physicist who is researching the parallel universes and travel across dimensions",
                  "children": [],
               }
            ],
        },
        {
            "value": "People attending Michael's funeral because the Michael in this dimension is really dead",
            "based_question": "why are people sad about Michael's death even though he is still alive?",
            "key_question": "Is Michael in this dimension is really dead?",
            "children": [
               {
                  "value": "Michael died in a traffic accident",
                  "based_question": "why does Michael die",
                  "key_question": "Does Michael died from an accident",
                  "children": [],
               }
            ],
        }
    ]
}
}

Surface:

{
  "surface": "Michael attended a funeral where everyone was mourning his death. Despite being present, no one could see or hear him. How is this possible?"
}

Now create the Surface based on the input:

"""