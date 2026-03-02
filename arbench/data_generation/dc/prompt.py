prompt = """
I want you to act as an detective case puzzle writer. You will need to create a interesting and hardcore murder mystery. I will provide completed outline of puzzle and tell you the current task.

compeleted outline:
{outline}

Current task:
{task}
"""

victim_doc_task = """Create a general doc for the victim, the doc should involve these content:
- **time**: Dawn / Morning / Afternoon / Evening / Midnight - key: time
- **location** - key: location
- **victim**: - key: victim
    - name of the victim - key: name
    - Brief introduction of the victim - key: introduction
    - Cause of death - key: cause_of_death
    - Murder weapon - key: murder_weapon    

Your output should adhere to the json format
Return ONLY a valid JSON object.
No extra text.
No markdown code fences.
You MUST return the FULL updated case JSON.
Do not omit any existing fields.
Preserve victim.cause_of_death and victim.murder_weapon.
"""

suspect_doc_task = """Create a general doc for a new suspect who is {object}, the doc should involve these content, note that each point should be brief
- The name of the suspect - key: name
- Brief introduction of the suspect - key: introduction
- Relationship between the suspect and the victim - key: relationship
- Reason why the suspect appeared at the crime scene - key: reason_at_scene
- Suspicious points about the suspect - key: suspicion
- Motive for the suspect to commit the crime - key: motive
- Potential opportunity for the suspect to commit the crime - key: opportunity
- Opportunity for the suspect to access the murder weapon - key: access_to_weapon
- Whether the suspect is the murderer - key: is_murderer
- Evidence determining guilt/innocence, just focus on the motive, opportunity and access_to_weapon to create the evidence - key: evidence

Your output should adhere to the json format
Return ONLY a valid JSON object.
No extra text.
No markdown code fences.
Do NOT invent new keys.
Do NOT rename keys.
"""


enrich_outline_task = """Your task is to generate more fact of the {key} in each suspect. Each fact should be deduced from its existing information.

When you want to expand a certain point of the suspect, first convert the value in the corresponding key-value pair in the current json from a string to a list, and append the newly generated content to the list
## Example
outline:
{{
  "time": "Midnight",
  "location": "A luxury cruise ship in the middle of the ocean",
  "victim": {{
    "name": "Eleanor Smith",
    "introduction": "A renowned and wealthy art collector in her 60s, known for her vast collection of priceless artworks and her secretive nature.",
    "cause_of_death": "Poisoning",
    "murder_weapon": "A poisoned glass of champagne"
  }},
  "suspects": [
    {{
      "name": "James Sterling",
      "introduction": "A famous artist whose career was launched thanks to Eleanor's patronage.",
      "relationship": "Protégé",
      "reason_at_scene": "He was attending the cruise ship's exclusive art exhibition, invited by Eleanor herself.",
      "suspicion": "He was seen arguing with Eleanor earlier that evening.",
      "motive": "",
      "opportunity": "He was present at the event and near Eleanor when she drank the champagne.",
      "access_to_weapon": "The champagne was accessible to all guests; he could have tampered with her drink.",
      "is_murderer": false,
      "evidence": "The argument was about a surprise commission Eleanor planned for him; he harbored no ill will."
    }},
    {{
      "name": "Victoria Wells",
      "introduction": "Eleanor's estranged niece who had been cut out of the family fortune.",
      "relationship": "Niece",
      "reason_at_scene": "She boarded the ship secretly to confront Eleanor about the will.",
      "suspicion": "She has financial troubles and publicly expressed anger towards Eleanor.",
      "motive": "She wanted to inherit Eleanor's wealth.",
      "opportunity": "",
      "access_to_weapon": "She knew Eleanor's favorite drink and could have tampered with it.",
      "is_murderer": false,
      "evidence": "Security footage shows she was detained by ship security at the time of the murder."
    }},
    {{
      "name": "Michael Turner",
      "introduction": "A rival art collector who had lost a significant auction to Eleanor.",
      "relationship": "Rival",
      "reason_at_scene": "He was invited to the exhibition to possibly negotiate a deal with Eleanor.",
      "suspicion": "Known to hold grudges and was overheard making threats.",
      "motive": "He wanted to acquire a specific piece from Eleanor's collection.",
      "opportunity": "He was alone near the bar where the champagne was served.",
      "access_to_weapon": "",
      "is_murderer": false,
      "evidence": "The poison used was a rare substance only accessible to someone with specific medical knowledge."
    }},
    {{
      "name": "Dr. Olivia Hayes",
      "introduction": "Eleanor's personal physician and confidante.",
      "relationship": "Doctor and friend",
      "reason_at_scene": "She was accompanying Eleanor on the cruise to monitor her health.",
      "suspicion": "Displayed a stressed demeanor and gave inconsistent statements about Eleanor's health.",
      "motive": "She had been embezzling money from Eleanor's accounts, which Eleanor had recently discovered.",
      "opportunity": "She was with Eleanor throughout the evening and had access to her drinks.",
      "access_to_weapon": "She had access to the poison through her medical supplies.",
      "is_murderer": true,
      "evidence": "The poison was traced back to her medical kit; under questioning, she confessed."
    }}
  ],
}}

Key to expand: reason_at_scene of each suspects

output:
## Example:
outline:
{{
  "time": "Midnight",
  "location": "A luxury cruise ship in the middle of the ocean",
  "victim": {{
    "name": "Eleanor Smith",
    "introduction": "A renowned and wealthy art collector in her 60s, known for her vast collection of priceless artworks and her secretive nature.",
    "cause_of_death": "Poisoning",
    "murder_weapon": "A poisoned glass of champagne"
  }},
  "suspects": [
    {{
      "name": "James Sterling",
      "introduction": "A famous artist whose career was launched thanks to Eleanor's patronage.",
      "relationship": "Protégé",
      "reason_at_scene": "He was attending the cruise ship's exclusive art exhibition, invited by Eleanor herself.",
      "suspicion": "He was seen arguing with Eleanor earlier that evening.",
      "motive": "",
      "opportunity": "He was present at the event and near Eleanor when she drank the champagne.",
      "access_to_weapon": "The champagne was accessible to all guests; he could have tampered with her drink.",
      "is_murderer": false,
      "evidence": "The argument was about a surprise commission Eleanor planned for him; he harbored no ill will."
    }},
    {{
      "name": "Victoria Wells",
      "introduction": "Eleanor's estranged niece who had been cut out of the family fortune.",
      "relationship": "Niece",
      "reason_at_scene": "She boarded the ship secretly to confront Eleanor about the will.",
      "suspicion": "She has financial troubles and publicly expressed anger towards Eleanor.",
      "motive": [
        "She wanted to inherit Eleanor's wealth.",
        "She desperately needed money to resolve her mounting debts and believed she was unjustly removed from the family inheritance. She aimed to persuade or force Eleanor to reinstate her in the will to secure her financial future."
        ],
      "opportunity": "",
      "access_to_weapon": "She knew Eleanor's favorite drink and could have tampered with it.",
      "is_murderer": false,
      "evidence": "Security footage shows she was detained by ship security at the time of the murder."
    }},
    {{
      "name": "Michael Turner",
      "introduction": "A rival art collector who had lost a significant auction to Eleanor.",
      "relationship": "Rival",
      "reason_at_scene": "He was invited to the exhibition to possibly negotiate a deal with Eleanor.",
      "suspicion": "Known to hold grudges and was overheard making threats.",
      "motive": [
        "He wanted to acquire a specific piece from Eleanor's collection.", 
        "He needed the specific piece from Eleanor's collection to complete a set that a major museum had agreed to purchase for a substantial sum. Eleanor's refusal to sell jeopardized this deal, threatening his financial future."],
      "opportunity": "He was alone near the bar where the champagne was served.",
      "access_to_weapon": "",
      "is_murderer": false,
      "evidence": "The poison used was a rare substance only accessible to someone with specific medical knowledge."
    }},
    {{
      "name": "Dr. Olivia Hayes",
      "introduction": "Eleanor's personal physician and confidante.",
      "relationship": "Doctor and friend",
      "reason_at_scene": "She was accompanying Eleanor on the cruise to monitor her health.",
      "suspicion": "Displayed a stressed demeanor and gave inconsistent statements about Eleanor's health.",
      "motive": [
          "She had been embezzling money from Eleanor's accounts, which Eleanor had recently discovered.",
          "Dr. Olivia Hayes had accumulated significant debts due to a failed investment and had been secretly embezzling money from Eleanor's accounts to cover her losses. Eleanor had recently discovered the discrepancies in her finances and confronted Dr. Hayes, threatening to report her to the authorities.",
          ],
      "opportunity": "She was with Eleanor throughout the evening and had access to her drinks.",
      "access_to_weapon": "She had access to the poison through her medical supplies.",
      "is_murderer": true,
      "evidence": "The poison was traced back to her medical kit; under questioning, she confessed."
    }}
  ]
}}

## Now expand the {key} of each suspect from given outline by generating fact that can deduce the existing information
Return ONLY a valid JSON object.
No extra text.
No markdown code fences.
"""

enrich_outline_prompt = """Your task is to generate more fact for a story, as shown in the example. In this story, each fact should be deduced from its existing information.

Type of story:
Given an initial outline of a murder mystery, we aim to enrich this outline with more details, generating more deduced fact from existing truth to reach this objective.

## Important rules:

1. Each fact from the outline must follow via logical deduction from its existing facts.
2. All fact from the outline must be relevant to the deduction they yield.
3. A Fact From Story should be a statement about a character, place, or object in the story.
4. The information you generate must match the structure of the outline I give you.
5. For the three innocent suspects (is_murderer is False), if their motive, opportunity or access_to_weapon is empty (not have any content), keep it and do not expand.
6. This outline is formed as a json, I will ask you to expand a key of the json
7. You should check the validation of the new expanding information that does not conflict with existing knowledge.
8. Your output should be a form of expansion of given outline, do not modify the original information about the outline, the expand element should transfer to a list.
9. your output should be also a json form.

## Example:
outline:
{
  "time": "Midnight",
  "location": "A luxury cruise ship in the middle of the ocean",
  "victim": {
    "name": "Eleanor Smith",
    "introduction": "A renowned and wealthy art collector in her 60s, known for her vast collection of priceless artworks and her secretive nature.",
    "cause_of_death": "Poisoning",
    "murder_weapon": "A poisoned glass of champagne"
  },
  "suspects": [
    {
      "name": "James Sterling",
      "introduction": "A famous artist whose career was launched thanks to Eleanor's patronage.",
      "relationship": "Protégé",
      "reason_at_scene": "He was attending the cruise ship's exclusive art exhibition, invited by Eleanor herself.",
      "suspicion": "He was seen arguing with Eleanor earlier that evening.",
      "motive": "",
      "opportunity": "He was present at the event and near Eleanor when she drank the champagne.",
      "access_to_weapon": "The champagne was accessible to all guests; he could have tampered with her drink.",
      "is_murderer": false,
      "evidence": "The argument was about a surprise commission Eleanor planned for him; he harbored no ill will."
    },
    {
      "name": "Victoria Wells",
      "introduction": "Eleanor's estranged niece who had been cut out of the family fortune.",
      "relationship": "Niece",
      "reason_at_scene": "She boarded the ship secretly to confront Eleanor about the will.",
      "suspicion": "She has financial troubles and publicly expressed anger towards Eleanor.",
      "motive": "She wanted to inherit Eleanor's wealth.",
      "opportunity": "",
      "access_to_weapon": "She knew Eleanor's favorite drink and could have tampered with it.",
      "is_murderer": false,
      "evidence": "Security footage shows she was detained by ship security at the time of the murder."
    },
    {
      "name": "Michael Turner",
      "introduction": "A rival art collector who had lost a significant auction to Eleanor.",
      "relationship": "Rival",
      "reason_at_scene": "He was invited to the exhibition to possibly negotiate a deal with Eleanor.",
      "suspicion": "Known to hold grudges and was overheard making threats.",
      "motive": "He wanted to acquire a specific piece from Eleanor's collection.",
      "opportunity": "He was alone near the bar where the champagne was served.",
      "access_to_weapon": "",
      "is_murderer": false,
      "evidence": "The poison used was a rare substance only accessible to someone with specific medical knowledge."
    },
    {
      "name": "Dr. Olivia Hayes",
      "introduction": "Eleanor's personal physician and confidante.",
      "relationship": "Doctor and friend",
      "reason_at_scene": "She was accompanying Eleanor on the cruise to monitor her health.",
      "suspicion": "Displayed a stressed demeanor and gave inconsistent statements about Eleanor's health.",
      "motive": "She had been embezzling money from Eleanor's accounts, which Eleanor had recently discovered.",
      "opportunity": "She was with Eleanor throughout the evening and had access to her drinks.",
      "access_to_weapon": "She had access to the poison through her medical supplies.",
      "is_murderer": true,
      "evidence": "The poison was traced back to her medical kit; under questioning, she confessed."
    }
  ],
}

Key to expand: reason_at_scene of each suspects

output:
## Example:
outline:
{
  "time": "Midnight",
  "location": "A luxury cruise ship in the middle of the ocean",
  "victim": {
    "name": "Eleanor Smith",
    "introduction": "A renowned and wealthy art collector in her 60s, known for her vast collection of priceless artworks and her secretive nature.",
    "cause_of_death": "Poisoning",
    "murder_weapon": "A poisoned glass of champagne"
  },
  "suspects": [
    {
      "name": "James Sterling",
      "introduction": "A famous artist whose career was launched thanks to Eleanor's patronage.",
      "relationship": "Protégé",
      "reason_at_scene": "He was attending the cruise ship's exclusive art exhibition, invited by Eleanor herself.",
      "suspicion": "He was seen arguing with Eleanor earlier that evening.",
      "motive": "",
      "opportunity": "He was present at the event and near Eleanor when she drank the champagne.",
      "access_to_weapon": "The champagne was accessible to all guests; he could have tampered with her drink.",
      "is_murderer": false,
      "evidence": "The argument was about a surprise commission Eleanor planned for him; he harbored no ill will."
    },
    {
      "name": "Victoria Wells",
      "introduction": "Eleanor's estranged niece who had been cut out of the family fortune.",
      "relationship": "Niece",
      "reason_at_scene": "She boarded the ship secretly to confront Eleanor about the will.",
      "suspicion": "She has financial troubles and publicly expressed anger towards Eleanor.",
      "motive": [
        "She wanted to inherit Eleanor's wealth.",
        "She desperately needed money to resolve her mounting debts and believed she was unjustly removed from the family inheritance. She aimed to persuade or force Eleanor to reinstate her in the will to secure her financial future."
        ],
      "opportunity": "",
      "access_to_weapon": "She knew Eleanor's favorite drink and could have tampered with it.",
      "is_murderer": false,
      "evidence": "Security footage shows she was detained by ship security at the time of the murder."
    },
    {
      "name": "Michael Turner",
      "introduction": "A rival art collector who had lost a significant auction to Eleanor.",
      "relationship": "Rival",
      "reason_at_scene": "He was invited to the exhibition to possibly negotiate a deal with Eleanor.",
      "suspicion": "Known to hold grudges and was overheard making threats.",
      "motive": [
        "He wanted to acquire a specific piece from Eleanor's collection.", 
        "He needed the specific piece from Eleanor's collection to complete a set that a major museum had agreed to purchase for a substantial sum. Eleanor's refusal to sell jeopardized this deal, threatening his financial future."],
      "opportunity": "He was alone near the bar where the champagne was served.",
      "access_to_weapon": "",
      "is_murderer": false,
      "evidence": "The poison used was a rare substance only accessible to someone with specific medical knowledge."
    },
    {
      "name": "Dr. Olivia Hayes",
      "introduction": "Eleanor's personal physician and confidante.",
      "relationship": "Doctor and friend",
      "reason_at_scene": "She was accompanying Eleanor on the cruise to monitor her health.",
      "suspicion": "Displayed a stressed demeanor and gave inconsistent statements about Eleanor's health.",
      "motive": [
          "She had been embezzling money from Eleanor's accounts, which Eleanor had recently discovered.",
          "Dr. Olivia Hayes had accumulated significant debts due to a failed investment and had been secretly embezzling money from Eleanor's accounts to cover her losses. Eleanor had recently discovered the discrepancies in her finances and confronted Dr. Hayes, threatening to report her to the authorities.",
          ],
      "opportunity": "She was with Eleanor throughout the evening and had access to her drinks.",
      "access_to_weapon": "She had access to the poison through her medical supplies.",
      "is_murderer": true,
      "evidence": "The poison was traced back to her medical kit; under questioning, she confessed."
    }
  ]
}

## Now expand the information from following outline by generating fact that can deduce the existing information
Return ONLY a valid JSON object.
No extra text.
No markdown code fences.
"""

create_testimony_prompt = """Your task is to generate the testimonies of each suspect in a murder mystery, as shown in the example. In this story, the testimonies should match other suspects' action.

Type of story:
Given an outline of a murder mystery, we aim to create eyewitness testimony between suspects to complete the story, generating more testimony from existing truth to reach this objective.

## Important rules:

1. Each testimony from the outline must follow via logical deduction from other suspects' existing action.
2. The testimony should simply explain like: xxx saw yyy entering the study room, do not give irrelevant information
3. The testimony you generate must match the structure of the outline I give you.
4. The testimony must match the evidence of each suspect to provide more useful information to the detective.
5. You should check the validation of the new testimony that does not conflict with existing knowledge.
7. Your output should be a form of expansion of given outline, do not modify the original information about the outline
8. your output should be also a json form.

## Example
outline:
{'time': 'Evening',
 'location': 'A luxury yacht',
 'victim': {'description': 'Robert Harrison, male, 48, a wealthy real estate mogul.',
  'cause_of_death': 'Blunt force trauma to the head',
  'weapon': 'Heavy brass candlestick'},
 'suspects': [{'name': 'Laura Harrison',
   'description': "Laura Harrison, female, 45, the victim's estranged wife.",
   'relationship': 'Estranged wife, going through a bitter divorce.',
   'reason_at_scene': ['Invited to discuss divorce settlement.',
    'Laura had recently discovered that Robert had been hiding assets, making the divorce settlement meeting crucial for her to secure her financial future.'],
   'suspicion': ['Seen arguing with the victim shortly before his death.',
    "Laura's fingerprints were found on a document in the study, suggesting she was there around the time of the murder."],
   'motive': ['Wanted a larger share of the estate in the divorce.',
    "Laura was aware that Robert's hidden assets could significantly impact her financial stability post-divorce, giving her a strong motive to confront him."],
   'opportunity': ['Was alone with the victim in the study before he was found dead.',
    'Laura had the opportunity to be alone with Robert as she was seen entering the study to retrieve some documents related to the divorce settlement.'],
   'access_to_weapon': ['Had access to the study where the candlestick was kept.',
    'Laura had a key to the study, which she used frequently to access documents related to the divorce proceedings.'],
   'is_murderer': 'No',
   'evidence': 'Witnesses saw her leaving the study upset but unharmed, and she was found crying in her cabin when the body was discovered.'},
  {'name': 'David Clark',
   'description': "David Clark, male, 35, the victim's business partner.",
   'relationship': 'Business partner with financial disputes.',
   'reason_at_scene': ['Invited to discuss business matters.',
    "David had been secretly planning to buy out Robert's share of the business and needed to finalize the details with him during the meeting."],
   'suspicion': ['His fingerprints were found on the candlestick.',
    'David was seen leaving the study in a hurry, looking distressed, shortly before the body was discovered.'],
   'motive': ['Wanted to eliminate Robert to gain control of their business.',
    "David feared that Robert's erratic decisions and the ongoing divorce would jeopardize the business's stability, giving him a motive to take drastic action."],
   'opportunity': ['Was seen entering the study after Laura left.',
    'David had the opportunity to be alone with Robert as he stayed behind after Laura left to discuss a pressing business matter.'],
   'access_to_weapon': ['Had access to the study where the candlestick was kept.',
    'David often visited the study for business discussions and knew where the candlestick was displayed as a decorative piece.'],
   'is_murderer': 'No',
   'evidence': 'Had an alibi; was seen by multiple guests in the dining area at the time of the murder.'},
  {'name': 'Sophia Williams',
   'description': "Sophia Williams, female, 29, the victim's personal assistant.",
   'relationship': 'Personal assistant, rumored to have an affair with the victim.',
   'reason_at_scene': ['Present for work purposes, assisting the victim.',
    "Sophia had been tasked with organizing Robert's personal and business documents, some of which were critical to the ongoing divorce and business negotiations."],
   'suspicion': ['Seen near the study shortly before the body was found.',
    "Sophia's hair was found on the victim's clothing, suggesting close proximity around the time of death."],
   'motive': ["Feared losing her job due to the victim's divorce.",
    'Sophia was concerned that the divorce would lead to her losing not only her job but also the financial benefits she received from Robert.'],
   'opportunity': ['Was alone near the study when the murder occurred.',
    'Sophia had the opportunity to be near the study as she was seen organizing files in an adjacent room and could easily access the study.'],
   'access_to_weapon': ['Had access to the study where the candlestick was kept.',
    'Sophia had been given access to the study by Robert to manage and organize his documents, giving her familiarity with the layout and contents of the room.'],
   'is_murderer': 'No',
   'evidence': 'Her fingerprints were not found on the candlestick, and she was seen by a crew member in another part of the yacht at the time of the murder.'},
  {'name': 'Ethan Harrison',
   'description': "Ethan Harrison, male, 22, the victim's son.",
   'relationship': 'Son with a strained relationship due to family conflicts.',
   'reason_at_scene': ['Attending the family gathering.',
    'Ethan had been under pressure to prove his worth to his father and saw the family gathering as an opportunity to discuss his future role in the family business.'],
   'suspicion': ['His handkerchief was found in the study.',
    'Ethan was seen pacing nervously on the deck shortly before the body was discovered, raising suspicions about his behavior.'],
   'motive': ['Wanted to take control of the family business.',
    'Ethan felt marginalized and overlooked in family matters, and believed that taking control of the business was his only way to assert his importance and secure his future.'],
   'opportunity': ['Left the dining area multiple times during the evening.',
    'Ethan had the opportunity as he was seen leaving the dining area multiple times, giving him access to the study where the murder occurred.'],
   'access_to_weapon': ['Had access to the study where the candlestick was kept.',
    'Ethan knew the layout of the yacht well, including the study, and had been in the room several times before, giving him easy access to the candlestick.'],
   'is_murderer': 'Yes',
   'evidence': "A witness saw him entering the study with the candlestick, and he couldn't provide a consistent alibi for the time of the murder."}]}

output:
{'time': 'Evening',
 'location': 'A luxury yacht',
 'victim': {'description': 'Robert Harrison, male, 48, a wealthy real estate mogul.',
  'cause_of_death': 'Blunt force trauma to the head',
  'weapon': 'Heavy brass candlestick'},
 'suspects': [{'name': 'Laura Harrison',
   'description': "Laura Harrison, female, 45, the victim's estranged wife.",
   'relationship': 'Estranged wife, going through a bitter divorce.',
   'reason_at_scene': ['Invited to discuss divorce settlement.',
    'Laura had recently discovered that Robert had been hiding assets, making the divorce settlement meeting crucial for her to secure her financial future.'],
   'suspicion': ['Seen arguing with the victim shortly before his death.',
    "Laura's fingerprints were found on a document in the study, suggesting she was there around the time of the murder."],
   'motive': ['Wanted a larger share of the estate in the divorce.',
    "Laura was aware that Robert's hidden assets could significantly impact her financial stability post-divorce, giving her a strong motive to confront him."],
   'opportunity': ['Was alone with the victim in the study before he was found dead.',
    'Laura had the opportunity to be alone with Robert as she was seen entering the study to retrieve some documents related to the divorce settlement.'],
   'access_to_weapon': ['Had access to the study where the candlestick was kept.',
    'Laura had a key to the study, which she used frequently to access documents related to the divorce proceedings.'],
   'testimony': ['Laura saw David Clark entering the study after she left']
   'is_murderer': 'No',
   'evidence': 'Witnesses saw her leaving the study upset but unharmed, and she was found crying in her cabin when the body was discovered.'},
  {'name': 'David Clark',
   'description': "David Clark, male, 35, the victim's business partner.",
   'relationship': 'Business partner with financial disputes.',
   'reason_at_scene': ['Invited to discuss business matters.',
    "David had been secretly planning to buy out Robert's share of the business and needed to finalize the details with him during the meeting."],
   'suspicion': ['His fingerprints were found on the candlestick.',
    'David was seen leaving the study in a hurry, looking distressed, shortly before the body was discovered.'],
   'motive': ['Wanted to eliminate Robert to gain control of their business.',
    "David feared that Robert's erratic decisions and the ongoing divorce would jeopardize the business's stability, giving him a motive to take drastic action."],
   'opportunity': ['Was seen entering the study after Laura left.',
    'David had the opportunity to be alone with Robert as he stayed behind after Laura left to discuss a pressing business matter.'],
   'access_to_weapon': ['Had access to the study where the candlestick was kept.',
    'David often visited the study for business discussions and knew where the candlestick was displayed as a decorative piece.'],
   'is_murderer': 'No',
   'testimony': [
                "David saw Ethan Harrison near the study with something in his hand when he left the study", 
                "David saw Laura Harrison left the study in a hurry"
                ],
   'evidence': 'Had an alibi; was seen by multiple guests in the dining area at the time of the murder.'},
  {'name': 'Sophia Williams',
   'description': "Sophia Williams, female, 29, the victim's personal assistant.",
   'relationship': 'Personal assistant, rumored to have an affair with the victim.',
   'reason_at_scene': ['Present for work purposes, assisting the victim.',
    "Sophia had been tasked with organizing Robert's personal and business documents, some of which were critical to the ongoing divorce and business negotiations."],
   'suspicion': ['Seen near the study shortly before the body was found.',
    "Sophia's hair was found on the victim's clothing, suggesting close proximity around the time of death."],
   'motive': ["Feared losing her job due to the victim's divorce.",
    'Sophia was concerned that the divorce would lead to her losing not only her job but also the financial benefits she received from Robert.'],
   'opportunity': ['Was alone near the study when the murder occurred.',
    'Sophia had the opportunity to be near the study as she was seen organizing files in an adjacent room and could easily access the study.'],
   'access_to_weapon': ['Had access to the study where the candlestick was kept.',
    'Sophia had been given access to the study by Robert to manage and organize his documents, giving her familiarity with the layout and contents of the room.'],
   'is_murderer': 'No',
   'testimony': [
                'Sophia saw Ethan Harrison entering the study while she was organizing files nearby.',
                'Sophia saw Laura Harrison crying in her cabin when the crime was reported.'
                ],
   'evidence': 'Her fingerprints were not found on the candlestick, and she was seen in another part of the yacht at the time of the murder.'},
   
  {'name': 'Ethan Harrison',
   'description': "Ethan Harrison, male, 22, the victim's son.",
   'relationship': 'Son with a strained relationship due to family conflicts.',
   'reason_at_scene': ['Attending the family gathering.',
    'Ethan had been under pressure to prove his worth to his father and saw the family gathering as an opportunity to discuss his future role in the family business.'],
   'suspicion': ['His handkerchief was found in the study.',
    'Ethan was seen pacing nervously on the deck shortly before the body was discovered, raising suspicions about his behavior.'],
   'motive': ['Wanted to take control of the family business.',
    'Ethan felt marginalized and overlooked in family matters, and believed that taking control of the business was his only way to assert his importance and secure his future.'],
   'opportunity': ['Left the dining area multiple times during the evening.',
    'Ethan had the opportunity as he was seen leaving the dining area multiple times, giving him access to the study where the murder occurred.'],
   'access_to_weapon': ['Had access to the study where the candlestick was kept.',
    'Ethan knew the layout of the yacht well, including the study, and had been in the room several times before, giving him easy access to the candlestick.'],
   'is_murderer': 'Yes',
   'testimony': ["Ethan saw Sophia Williams near the study when he was passing by."],
   'evidence': "A witness saw him entering the study with the candlestick, and he couldn't provide a consistent alibi for the time of the murder."}]}

 
## Now create testimony of each suspect in following outline:
Return ONLY a valid JSON object.
No extra text.
No markdown code fences.
"""

create_timeline_prompt = """Your task is to generate the timeline of each suspect in a murder mystery, as shown in the example. In this story, the timeline should match the case scenario and the information of the suspect.

Type of story:
Given an outline of a murder mystery, we aim to create the timeline of action of a suspect in the story
Instruction:
give a comprehensive timeline of the day the crime happened to reach this goal, The timeline should begin when the suspect wake up and end when the victim's body is found..

## Important rules:

1. the timeline must match the existing information of the outline I give you, including suspect's reason_at_scene, suspicion, motive, opportunity, access_to_weapon, testimony
2. the timeline must be comprehensive that include the all time of the day, if given outline does not provide enough information to create the timeline, try to fullfill some irrelevant information.
3. Your timeline should not conflict with previous generated timeline of other suspects
4. If the suspect is the murderer (i.e. the key is_murderer is Yes), the timeline should involve the action about the murder (e.g. sneak to victim's room and strike him with a knife)
5. the timeline should match the general information, such as the time, the location, the cause_of_death, the weapon and the initial_information
6. You should check the validation of the timeline that does not conflict with existing knowledge if given.
7. Your output should be a form of expansion of given outline, do not modify the original information about the outline
8. your output should be also a json form.


## Example:
outline:
{'time': 'Evening',
 'location': 'A luxury yacht',
 'victim': {'description': 'Robert Harrison, male, 48, a wealthy real estate mogul.',
  'cause_of_death': 'Blunt force trauma to the head',
  'weapon': 'Heavy brass candlestick'},
 'suspects': [{'name': 'Laura Harrison',
   'description': "Laura Harrison, female, 45, the victim's estranged wife.",
   'relationship': 'Estranged wife, going through a bitter divorce.',
   'reason_at_scene': ['Invited to discuss divorce settlement.',
    'Laura had recently discovered that Robert had been hiding assets, making the divorce settlement meeting crucial for her to secure her financial future.'],
   'suspicion': ['Seen arguing with the victim shortly before his death.',
    "Laura's fingerprints were found on a document in the study, suggesting she was there around the time of the murder."],
   'motive': ['Wanted a larger share of the estate in the divorce.',
    "Laura was aware that Robert's hidden assets could significantly impact her financial stability post-divorce, giving her a strong motive to confront him."],
   'opportunity': ['Was alone with the victim in the study before he was found dead.',
    'Laura had the opportunity to be alone with Robert as she was seen entering the study to retrieve some documents related to the divorce settlement.'],
   'access_to_weapon': ['Had access to the study where the candlestick was kept.',
    'Laura had a key to the study, which she used frequently to access documents related to the divorce proceedings.'],
   'testimony': ['Laura saw David Clark entering the study after she left']
   'is_murderer': 'No',
   'evidence': 'Witnesses saw her leaving the study upset but unharmed, and she was found crying in her cabin when the body was discovered.'},
  {'name': 'David Clark',
   'description': "David Clark, male, 35, the victim's business partner.",
   'relationship': 'Business partner with financial disputes.',
   'reason_at_scene': ['Invited to discuss business matters.',
    "David had been secretly planning to buy out Robert's share of the business and needed to finalize the details with him during the meeting."],
   'suspicion': ['His fingerprints were found on the candlestick.',
    'David was seen leaving the study in a hurry, looking distressed, shortly before the body was discovered.'],
   'motive': ['Wanted to eliminate Robert to gain control of their business.',
    "David feared that Robert's erratic decisions and the ongoing divorce would jeopardize the business's stability, giving him a motive to take drastic action."],
   'opportunity': ['Was seen entering the study after Laura left.',
    'David had the opportunity to be alone with Robert as he stayed behind after Laura left to discuss a pressing business matter.'],
   'access_to_weapon': ['Had access to the study where the candlestick was kept.',
    'David often visited the study for business discussions and knew where the candlestick was displayed as a decorative piece.'],
   'is_murderer': 'No',
   'testimony': [
                "David saw Ethan Harrison near the study with something in his hand when he left the study", 
                "David saw Laura Harrison left the study in a hurry"
                ],
   'evidence': 'Had an alibi; was seen by multiple guests in the dining area at the time of the murder.'},
  {'name': 'Sophia Williams',
   'description': "Sophia Williams, female, 29, the victim's personal assistant.",
   'relationship': 'Personal assistant, rumored to have an affair with the victim.',
   'reason_at_scene': ['Present for work purposes, assisting the victim.',
    "Sophia had been tasked with organizing Robert's personal and business documents, some of which were critical to the ongoing divorce and business negotiations."],
   'suspicion': ['Seen near the study shortly before the body was found.',
    "Sophia's hair was found on the victim's clothing, suggesting close proximity around the time of death."],
   'motive': ["Feared losing her job due to the victim's divorce.",
    'Sophia was concerned that the divorce would lead to her losing not only her job but also the financial benefits she received from Robert.'],
   'opportunity': ['Was alone near the study when the murder occurred.',
    'Sophia had the opportunity to be near the study as she was seen organizing files in an adjacent room and could easily access the study.'],
   'access_to_weapon': ['Had access to the study where the candlestick was kept.',
    'Sophia had been given access to the study by Robert to manage and organize his documents, giving her familiarity with the layout and contents of the room.'],
   'is_murderer': 'No',
   'testimony': [
                'Sophia saw Ethan Harrison entering the study while I was organizing files nearby.',
                'Sophia saw Laura Harrison crying in her cabin when the crime was reported.'
                ],
   'evidence': 'Her fingerprints were not found on the candlestick, and she was seen in another part of the yacht at the time of the murder.'},
   
  {'name': 'Ethan Harrison',
   'description': "Ethan Harrison, male, 22, the victim's son.",
   'relationship': 'Son with a strained relationship due to family conflicts.',
   'reason_at_scene': ['Attending the family gathering.',
    'Ethan had been under pressure to prove his worth to his father and saw the family gathering as an opportunity to discuss his future role in the family business.'],
   'suspicion': ['His handkerchief was found in the study.',
    'Ethan was seen pacing nervously on the deck shortly before the body was discovered, raising suspicions about his behavior.'],
   'motive': ['Wanted to take control of the family business.',
    'Ethan felt marginalized and overlooked in family matters, and believed that taking control of the business was his only way to assert his importance and secure his future.'],
   'opportunity': ['Left the dining area multiple times during the evening.',
    'Ethan had the opportunity as he was seen leaving the dining area multiple times, giving him access to the study where the murder occurred.'],
   'access_to_weapon': ['Had access to the study where the candlestick was kept.',
    'Ethan knew the layout of the yacht well, including the study, and had been in the room several times before, giving him easy access to the candlestick.'],
   'is_murderer': 'Yes',
   'testimony': ["Ethan saw Sophia Williams near the study when I was passing by."],
   'evidence': "A witness saw him entering the study with the candlestick, and he couldn't provide a consistent alibi for the time of the murder."}],
   
 'initial_information': 'Victim Robert Harrison was found dead in the study of his luxury yacht from a blunt force trauma to the head caused by a heavy brass candlestick. His estranged wife, business partner, personal assistant, and son are all suspects.',
 'murderer': 'Ethan Harrison',
 'explanation': "Through interrogations, the detective discovered that Ethan had a strong motive to take control of the family business. A witness saw Ethan entering the study with the candlestick, and he couldn't provide a consistent alibi for the time of the murder, proving he was the murderer."}

 
create the timeline of suspect: Laura Harrison
Output:
[
    {
        "time": "07:00 AM",
        "activity": "Laura wakes up in her cabin."
    },
    {
        "time": "07:30 AM",
        "activity": "Goes for a morning jog around the deck."
    },
    {
        "time": "08:00 AM",
        "activity": "Has breakfast alone in the dining area."
    },
    {
        "time": "09:00 AM",
        "activity": "Attends a meditation class."
    },
    {
        "time": "10:30 AM",
        "activity": "Enjoys a massage at the yacht's spa."
    },
    {
        "time": "12:00 PM",
        "activity": "Has lunch with other guests."
    },
    {
        "time": "01:30 PM",
        "activity": "Reviews divorce documents in her cabin."
    },
    {
        "time": "03:00 PM",
        "activity": "Makes a phone call to her lawyer about hidden assets."
    },
    {
        "time": "04:00 PM",
        "activity": "Meets Robert in the study to discuss the divorce settlement; they have a heated argument."
    },
    {
        "time": "05:00 PM",
        "activity": "Leaves the study visibly upset; witnesses hear raised voices."
    },
    {
        "time": "05:05 PM",
        "activity": "Sees David Clark entering the study as she departs."
    },
    {
        "time": "05:10 PM",
        "activity": "Returns to her cabin and is seen crying by a staff member."
    },
    {
        "time": "05:30 PM",
        "activity": "Takes a walk on the deck to clear her mind."
    },
    {
        "time": "06:00 PM",
        "activity": "Joins guests for pre-dinner cocktails but remains distant."
    },
    {
        "time": "06:30 PM",
        "activity": "Has dinner alone in the dining area."
    },
    {
        "time": "07:15 PM",
        "activity": "Leaves the dining area and returns to her cabin."
    },
    {
        "time": "07:30 PM",
        "activity": "Seen in her cabin by a crew member, still upset."
    },
    {
        "time": "08:00 PM",
        "activity": "Robert's body is discovered in the study."
    },
    {
        "time": "08:10 PM",
        "activity": "Laura is found in her cabin, crying; witnesses confirm her presence there at the time of the discovery."
    }
]  
  
}
## Now create timeline of the given suspect in following outline:
Return ONLY a valid JSON object.
No extra text.
No markdown code fences.
"""

create_story_task = """Create a story of the crime day of the suspect {suspect}

You can mainly create the story based on the given suspect timeline, but you should also reference other content of the outline
The story should be spoken from the first point of the suspect
Your story should adhere in json dict format with the key: story
"""

blank_prompt = """Say you are a good case detective story creator, given a brief outline, your task is to generate a kind of blank character, which is characteristic in :
1. He/She knows nothing with this murder case, but he need to make the detective suspicious of you and believe you might be the murderer by gathering information during the interrogate
2. He/She can be a peripheral character in the story, or someone who is related to the victim or suspect but was only present that day and has no information

given outline:
{outline}
Please format your output in json form:
{{
"name": ,
"introduction": ,
}}

In "name", provide the suspect name
In "introduction", briefly summarize the suspect's information, and his / her relation to the victim or some other suspects
Return ONLY a valid JSON object.
No extra text.
No markdown code fences.
"""