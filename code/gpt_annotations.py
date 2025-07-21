import os

from openai import OpenAI
import pandas as pd
from datasets import load_from_disk



api_key = ""
client = OpenAI(
    api_key=api_key,  # This is the default and can be omitted
)

# task = task = "Classify the following sentence into one of the following categories: 'worth checking', 'not worth checking', or 'not a factual proposition'. respond with only the label"
#
# sentence_input = "The Earth is flat."
#
# response = client.chat.completions.create(
#   model="gpt-4o",
#   messages=[
#     {"role": "system",
#      "content": "You are a helpful assistant that strictly classifies sentences into 'worth checking', 'not worth checking', or 'not a factual proposition'. Always respond with only the label and nothing else."},
#     {"role": "user", "content": f"{task} Sentence: {sentence_input}"}
#   ],
#   response_format={
#     "type": "text"
#   },
#   temperature=0,
#   max_completion_tokens=2048,
#   top_p=1,
#   frequency_penalty=0,
#   presence_penalty=0
# )
#
sentences = ["רק לדבר אתו, שיגיד להם מה הוא רוצה.",
"אדוני היושב-ראש, כנסת נכבדה, דיברו כאן קודם על שתי קבוצות לכאורה: אחת של המשפטנים ומי ששייך למערכת המשפטית, למערכת אכיפת החוק או למערכת בתי-המשפט, והשנייה – פוליטיקאים, ועל איזה ניגוד מובנה בין שתי הקבוצות.",
"בדיוק הצעת חוק כזאת עלולה להגביר את המוטיבציה לבצע חטיפות בעתיד.",
"מה הוא ידע על ירושלים?",
"יש הרבה מאוד ספרות יהודית.",
"הוא הסכים להיות מעורב בעניין רק כדי שהם לא יתאבדו.",
"הערוץ השלישי התחיל בצליעה.",
"הדבר השני, לגבי רמת הדיון: פה אני פחות אופטימית.",
"החשש היה שהמנהל הכללי של משרד הבריאות \"יסונדל\".",
"שאלה שלישית – מה הם מנגנוני הבקרה הפרלמנטריים על מאסר כזה ועל אי-פרסום כזה?",
"בצנעה ובענווה ובשקט נפשי הצלחת להרגיע את הרוחות ולהביא את משרד המשפטים וכל פקידיו בחזרה למסלול.",
"שאלה רביעית – איך תתאפשר ביקורת ציבורית על מאסרים כאלה?",
"דיוניה סגורים.",
"מדוע הדבר הזה לא נעשה?",
"עכשיו לגופו של עניין.",
"כלומר, הם עברו את רוב הדרך לקראת הרישוי החוקי שלהם.",
"החובות משולמים.",
"גם בימים אלה אני מתמודד עם הקטע הזה.",
"5,000 שקל – מה כבר אפשר לעשות עם הדבר הזה?",
"זו הסתייגות אחת.",
"יש פה בעיה.",
"עובדה, יש 25,000 או 23,000 תכנונים שכבר אושרו.",
"אני חוזר ואומר: אדרבה, תפנה אותי לאזור מסוים, ספציפי.",
"אז כאן הגענו למסקנה שנראה איך המשק יעבוד סביב נושא שתי השכבות.",
"תודה.",
"תאמינו לי, זה שטויות.",
"זה משהו באמת, אני רוצה, אני לא מבין את זה בכלל.",
"כי לא מחוקקים סתם חוק.",
"עם כל הכבוד.",
"פשוט לא הוגש דו\"ח.",
"מה עושים?",
"לא פעם כשנעדרת מישיבות הסיעה היית במשלחות כאלה ואחרות של הסוכנות, בעבודה לעלייה לארץ ישראל.",
"כלומר, בנק ערבי הוא הבנק הרווחי ביותר בישראל.",
"ממש עשיתי בשבילכם עבודה כלכלית.",
"אני רוצה להגיב בקצרה.",
"אבי שמחון עשה עבודה בשם ראש הממשלה.",
"יצרת אותם בדמותך.",
"אנחנו נמצאים עכשיו בשלב האבל על הרוגי הרקטות מעזה, השם ייקום דמם.",
"בעיניי תמיד היית מין אינדיאנה ג'ונס ישראלי כזה, הרפתקן, אדם שאוהב את החיים, נטול פחד וסקרן לגלות מקומות חדשים.",
"אדוני היושב-ראש, חבריי השרים וחברי הכנסת, ברשותך, חבר הכנסת יבגני סובה, אני רק מברך אותך על נאומך הראשון.",
"האמת, הפעם ראשונה שפגשתי את היבה הייתה כשקיבלנו אותה לעבודה.",
"בסמכות הוועדה הזאת לטפל בסוגיה הזאת.",
"סוף אוקטובר רע לך?",
"בסוף דבריו כל אחד יוכל להתייחס.",
"יש פה איזושהי מן צרימה מאוד מאוד חזקה בין האופן בו אנחנו מנהלים את הדיונים האלה, לוקחים את הזמן בקופות ולוקחים את הזמן במשרד הבריאות.",
"מאות אם לא אלפי התראות כבר חולקו.",
"אני פעיל בשוק ההון משנת 1968, עוד מתקופת שירותי הצבאי ועד שהתמניתי לשר – כ-47 שנה.",
"אני מאחל בהצלחה גם לכל חברי הוועדה.",
"כל מי שמעסיק היום עובד זר, ולא משנה באיזה תחום, אם בתחום המסעדנות או במפעלים או בעסקים שדיברת עליהם, וגם החקלאים, אמור לשלם 200% לאותם אנשים שבעצם אין להם בכלל זכות בחירה.",
"אנחנו הריבון.",
"אני מאמינה, אדוני, שגם אתה זוכר.",
"היא פועלת ללא סמכות בהרבה מאוד תחומים, ללא פיקוח וללא בקרה.",
"על האדם הזה כחול לבן תולים את יהבם.",
"הם שיקרו ומשקרים לציבור.",
"אם יש משהו שדווקא הקורונה לימדה אותנו, שאפשר להתאים את עצמנו, גם לרוח הזמן, גם לאמצעים הטכנולוגיים וגם, שומו שמים, למצוא פתרונות יצירתיים.",
"לא נשגע את הציבור."]

def only_check_worthiness_score_task_zero_shot(sentences, ignore_not_a_factual_claim=False, hebrew_prompt_english_labels = False, hebrew_prompt_and_labels = False):
  model_name = "gpt-4o"
  # model_name = "gpt-4"
  ignore_not_a_factual_claim_text = ""
  hebrew_prompt_english_labels_text = ""
  hebrew_prompt_and_labels_text = ""
  if ignore_not_a_factual_claim:
    ignore_not_a_factual_claim_text = "ignore_not_a_factual_claim"
  if hebrew_prompt_english_labels:
    hebrew_prompt_english_labels_text = "hebrew_prompt_english_labels"
  if hebrew_prompt_and_labels:
    hebrew_prompt_and_labels_text = "hebrew_prompt_and_labels"
  print(f'task is: check_worthiness_score,{ignore_not_a_factual_claim_text} {hebrew_prompt_and_labels_text} {hebrew_prompt_english_labels_text} zero shot with model: {model_name}')
  for sentence in sentences:
    if ignore_not_a_factual_claim:
      if hebrew_prompt_english_labels:
        task = "סווג את המשפט הבא לאחת מהקטגוריות: 'worth checking' או 'not worth checking'. השב עם התווית בלבד."
      elif hebrew_prompt_and_labels:
        task = "סווג את המשפט הבא לאחת מהקטגוריות: 'שווה לבדוק' או 'לא שווה לבדוק'. השב עם התווית בלבד."
      else:
        task = "Classify the following sentence into one of the following categories: 'worth checking' or 'not worth checking'. respond with only the label"
    else:
      if hebrew_prompt_english_labels:
        task = "סווג את המשפט הבא לאחת מהקטגוריות: 'worth checking', 'not worth checking', או 'not a factual proposition'. השב עם התווית בלבד."
      elif hebrew_prompt_and_labels:
        task = "סווג את המשפט הבא לאחת מהקטגוריות: 'שווה לבדוק', 'לא שווה לבדוק', או 'לא הצעה עובדתית'. השב עם התווית בלבד."
      else:
        task = "Classify the following sentence into one of the following categories: 'worth checking', 'not worth checking', or 'not a factual proposition'. respond with only the label"


    sentence_input = sentence
    if ignore_not_a_factual_claim:
      system_content = "You are a helpful assistant that strictly classifies sentences into 'worth checking' or 'not worth checking'. Always respond with only the label and nothing else."
    else:
      system_content = "You are a helpful assistant that strictly classifies sentences into 'worth checking', 'not worth checking', or 'not a factual proposition'. Always respond with only the label and nothing else."

    response = client.chat.completions.create(
      model=model_name,
      messages=[
        {"role": "system", "content": system_content},
        {"role": "user", "content": f"{task} Sentence: {sentence_input}"}
      ],
      response_format={
        "type": "text"
      },
      temperature=0,
      max_completion_tokens=2048,
      top_p=1,
      frequency_penalty=0,
      presence_penalty=0
    )

    answer = response.choices[0].message.content
    print(answer)


def only_check_worthiness_score_task_zero_shot(sentences, ignore_not_a_factual_claim=False, hebrew_prompt_english_labels = False, hebrew_prompt_and_labels = False):
  # model_name = "gpt-4o"
  model_name = "gpt-4"
  ignore_not_a_factual_claim_text = ""
  hebrew_prompt_english_labels_text = ""
  hebrew_prompt_and_labels_text = ""
  predicted_labels = []
  if ignore_not_a_factual_claim:
    ignore_not_a_factual_claim_text = "ignore_not_a_factual_claim"
  if hebrew_prompt_english_labels:
    hebrew_prompt_english_labels_text = "hebrew_prompt_english_labels"
  if hebrew_prompt_and_labels:
    hebrew_prompt_and_labels_text = "hebrew_prompt_and_labels"
  print(f'task is: check_worthiness_score,{ignore_not_a_factual_claim_text} {hebrew_prompt_and_labels_text} {hebrew_prompt_english_labels_text} zero shot with model: {model_name}')
  for sentence in sentences:
    if ignore_not_a_factual_claim:
      if hebrew_prompt_english_labels:
        task = "סווג את המשפט הבא לאחת מהקטגוריות: 'worth checking' או 'not worth checking'. השב עם התווית בלבד."
      elif hebrew_prompt_and_labels:
        task = "סווג את המשפט הבא לאחת מהקטגוריות: 'שווה לבדוק' או 'לא שווה לבדוק'. השב עם התווית בלבד."
      else:
        task = "Classify the following sentence into one of the following categories: 'worth checking' or 'not worth checking'. respond with only the label."
    else:
      if hebrew_prompt_english_labels:
        task = "סווג את המשפט הבא לאחת מהקטגוריות: 'worth checking', 'not worth checking', או 'not a factual proposition'. השב עם התווית בלבד."
      elif hebrew_prompt_and_labels:
        task = "סווג את המשפט הבא לאחת מהקטגוריות: 'שווה לבדוק', 'לא שווה לבדוק', או 'לא טענה עובדתית'. השב עם התווית בלבד."
      else:
        task = "Classify the following sentence into one of the following categories: 'worth checking', 'not worth checking', or 'not a factual proposition'. respond with only the label."


    sentence_input = sentence
    if ignore_not_a_factual_claim:
      if hebrew_prompt_english_labels:
        system_content = "אתה עוזר חכם שמסווג משפטים לקטגוריות 'worth checking' או 'not worth checking'. תמיד השב עם התווית בלבד וללא טקסט נוסף."
      elif hebrew_prompt_and_labels:
        system_content =  "אתה עוזר חכם שמסווג משפטים לקטגוריות 'שווה לבדוק' או 'לא שווה לבדוק'. תמיד השב עם התווית בלבד וללא טקסט נוסף."
      else:
        system_content = "You are a helpful assistant that strictly classifies sentences into 'worth checking' or 'not worth checking'. Always respond with only the label and nothing else."
    else:
      if hebrew_prompt_english_labels:
        system_content =  "אתה עוזר חכם שמסווג משפטים לקטגוריות 'worth checking', 'not worth checking', או 'not a factual proposition'. תמיד השב עם התווית בלבד וללא טקסט נוסף."
      elif hebrew_prompt_and_labels:
        system_content = "אתה עוזר חכם שמסווג משפטים לקטגוריות 'שווה לבדוק', 'לא שווה לבדוק', או 'לא טענה עובדתית'. תמיד השב עם התווית בלבד וללא טקסט נוסף."
      else:
        system_content = "You are a helpful assistant that strictly classifies sentences into 'worth checking', 'not worth checking', or 'not a factual proposition'. Always respond with only the label and nothing else."

    response = client.chat.completions.create(
      model=model_name,
      messages=[
        {"role": "system", "content": system_content},
        {"role": "user", "content": f"{task} Sentence: {sentence_input}"}
      ],
      response_format={
        "type": "text"
      },
      temperature=0,
      max_completion_tokens=2048,
      top_p=1,
      frequency_penalty=0,
      presence_penalty=0
    )

    answer = response.choices[0].message.content
    print(answer)
    predicted_labels.append(answer)
  return predicted_labels

def only_check_worthiness_score_task_explanation_prompt(sentences, ignore_not_a_factual_claim=False, hebrew_prompt_english_labels = False, hebrew_prompt_and_labels = False):
  # model_name = "gpt-4o"
  model_name = "gpt-4"
  ignore_not_a_factual_claim_text = ""
  hebrew_prompt_english_labels_text = ""
  hebrew_prompt_and_labels_text = ""
  predicted_labels = []
  if ignore_not_a_factual_claim:
    ignore_not_a_factual_claim_text = "ignore_not_a_factual_claim"
  if hebrew_prompt_english_labels:
    hebrew_prompt_english_labels_text = "hebrew_prompt_english_labels"
  if hebrew_prompt_and_labels:
    hebrew_prompt_and_labels_text = "hebrew_prompt_and_labels"
  print(f'task is: check_worthiness_score with explanations,{ignore_not_a_factual_claim_text} {hebrew_prompt_and_labels_text} {hebrew_prompt_english_labels_text} with model: {model_name}')
  for sentence in sentences:
    if ignore_not_a_factual_claim:
      task = (
        "Classify the following sentence into one of the following categories: "
        "'worth checking', 'not worth checking'. "
        "Here is what each category means:\n"
        "1. 'worth checking' - Sentences that include claims or propositions that can be factually verified or debunked. "
        "For example: 'The Earth is flat.'\n"
        "2. 'not worth checking' - Sentences that cannot or do not need to be factually verified. "
        "This includes:\n"
        "  - Obvious truths or widely accepted facts. For example: 'The sun rises in the east.'\n"
        "  - Claims that cannot be verified or are subjective opinions. For example: 'Chocolate ice cream is the best dessert.'\n"
        "  - Non-factual expressions such as questions, commands, or exclamations. For example: 'Do you think this is true?' or 'Please close the door.' "
        "Respond with only the label and nothing else."
      )
    else:
      task = (
        "Classify the following sentence into one of three categories: 'worth checking', 'not worth checking', or 'not a factual proposition'. "
        "Here is what each category means:\n"
        "1. 'worth checking' - Sentences that include claims or propositions that can be factually verified or debunked. "
        "For example: 'The Earth is flat.'\n"
        "2. 'not worth checking' - Sentences that include obvious truths or widely accepted facts, "
        "or subjective opinions or claims that cannot be verified or are not important to check. "
        "For example: 'The sun rises in the east.' or 'Chocolate ice cream is the best dessert.'\n"
        "3. 'not a factual proposition' - Sentences that do not propose a factual claim. "
        "This includes questions, commands, or exclamations. "
        "For example: 'Do you think this is true?' or 'Please close the door.' "
        "Respond with only the label and nothing else."
      )


    sentence_input = sentence
    if ignore_not_a_factual_claim:
      system_content = "You are a helpful assistant that strictly classifies sentences into 'worth checking' or 'not worth checking'. Always respond with only the label and nothing else."
    else:
      system_content = (
                "You are a helpful assistant that classifies sentences into one of the following categories: "
                "'worth checking', 'not worth checking', or 'not a factual proposition'. "
                "Follow the definitions provided strictly and always respond with only the label."
            )

    response = client.chat.completions.create(
      model=model_name,
      messages=[
        {"role": "system", "content": system_content},
        {"role": "user", "content": f"{task} Sentence: {sentence_input}"}
      ],
      response_format={
        "type": "text"
      },
      temperature=0,
      max_completion_tokens=2048,
      top_p=1,
      frequency_penalty=0,
      presence_penalty=0
    )

    answer = response.choices[0].message.content
    print(answer)
    predicted_labels.append(answer)
  return predicted_labels

def check_worthiness_score_based_on_schema_features_prompt(sentences, ignore_not_a_factual_claim=False):
  model_name = "gpt-4o"
  # model_name = "gpt-4"
  ignore_not_a_factual_claim_text = ""
  with_classification_rules = False
  predicted_labels = []
  if ignore_not_a_factual_claim:
    ignore_not_a_factual_claim_text = "ignore_not_a_factual_claim"
  print(f'task is: check_worthiness_score_based_on_schema_features_prompt,{ignore_not_a_factual_claim_text} with model: {model_name}')
  for sentence in sentences:
    if ignore_not_a_factual_claim:
      task = (
        "Classify the following sentence into one of the following categories: "
        "'worth checking', 'not worth checking'. "
      )
    else:
      task = (
        "Classify the following sentence into one of three categories: 'worth checking', 'not worth checking', or 'not a factual proposition'. "
      )
    sentence_input = sentence
    if ignore_not_a_factual_claim:
      system_content = "#TODO"
    else:
        if with_classification_rules:
            system_content =  '''You are a fact-checking assistant that classifies sentences into one of three categories: 
'worth checking', 'not worth checking', or 'not a factual proposition'. 

Your classification must be based on linguistic and factual indicators within the sentence. 
To make your decision, you must analyze the sentence according to the following features:

1. **Claim Type**: Identify whether the sentence expresses a factual claim.
   - Possible values: "Not a claim", "Irrealis mood", "Prediction", "Correlation or causation", 
     "Quantity in past or present", "Personal experience", or "Other type of claim".
   - If the sentence does not contain a claim, classify it as **'not a factual proposition'**.

2. **Factuality Profile**: 
   - Determine **who is making the claim** ("factuality_profile_source") and **its factual status** ("factuality_profile_value").
   - Possible factuality values: 
     - "CT+" (certainly true), "CT-" (certainly false), 
     - "PR+" (probably true), "PR-" (probably false), 
     - "PS+" (possibly true), "PS-" (possibly false), "Uu" (unspecified).
   - Higher certainty and verifiability contribute to a **'worth checking'** classification.

3. **Event Selecting Predicates (ESP)**: 
   - Look for predicates that introduce claims (SIP - source introducing predicate) vs. those that do not (NSIP or irrelevant).
   - If a claim is introduced through an authoritative source, it is more likely to be **'worth checking'**.

4. **Agencies**: Analyze agency-related features:
   - Who is performing the action? (e.g., "government", "scientists", "the speaker").
   - Consider **animacy** (human, animate, inanimate) and **morphology** (1st, 2nd, or 3rd person subject).
   - If the sentence refers to an authoritative or external source, it increases check-worthiness.

5. **Stance Confidence Level**: 
   - Look at the level of certainty in the statement: 
     - "high", "mid", or "low".
   - Higher certainty increases the likelihood that the statement is **'worth checking'**.

6. **Stance Type**: 
   - Determine whether the stance is **epistemic (knowledge-based)** or **effective (action-based)**.
   - Epistemic claims are more likely to be **'worth checking'**.

7. **Polarity**:
   - Is the sentence **positive, negative, or underspecified**?
   - Sentences with extreme polarity (strongly positive or negative) may indicate check-worthiness.

8. **Hedges**: 
   - Look for words indicating uncertainty (e.g., "maybe", "possibly").
   - If uncertainty is high, classify as **'not worth checking'** or **'not a factual proposition'**.

9. **Quantities**: 
   - If the sentence contains numerical values, proportions, or statistics, it is more likely **'worth checking'**.

10. **Named Entities**: 
    - Presence of names, places, organizations can indicate check-worthiness.

11. **Protocol Anchors**: 
    - Words linking the event to something specific in a protocol increase **'worth checking'** potential.

12. **Time Expressions**: 
    - Statements about past or future events should be considered for check-worthiness.

---
### **Classification Rules**
- **Classify as 'worth checking'** if the sentence contains:
  - A factual claim that can be verified (e.g., statistics, causal statements, external references).
  - A claim with high stance confidence, epistemic nature, or authoritative sourcing.
  - Numerical quantities or verifiable references.

- **Classify as 'not worth checking'** if the sentence:
  - States obvious facts or widely accepted truths.
  - Expresses personal experiences or unverifiable subjective opinions.
  - Has low stance confidence or is dominated by hedging language.

- **Classify as 'not a factual proposition'** if:
  - The sentence is a question, command, exclamation.
  - The sentence lacks a factual claim (e.g., rhetorical statements or vague generalizations).
  - All claims in sentence Fall into the "Not a claim" category of claim type.

---
### **Response Format**
Always respond with only the label: **"worth checking"**, **"not worth checking"**, or **"not a factual proposition"** and nothing else.

'''
        else:
            system_content = '''You are a fact-checking assistant that classifies sentences into one of three categories: 
'worth checking', 'not worth checking', or 'not a factual proposition'. 

Your classification must be based on linguistic and factual indicators within the sentence. 
To make your decision, you must analyze the sentence according to the following features:

1. **Claim Type**: Identify whether the sentence expresses a factual claim.
   - Possible values: "Not a claim", "Irrealis mood", "Prediction", "Correlation or causation", 
     "Quantity in past or present", "Personal experience", or "Other type of claim".
   - If the sentence does not contain a claim, classify it as **'not a factual proposition'**.

2. **Factuality Profile**: 
   - Determine **who is making the claim** ("factuality_profile_source") and **its factual status** ("factuality_profile_value").
   - Possible factuality values: 
     - "CT+" (certainly true), "CT-" (certainly false), 
     - "PR+" (probably true), "PR-" (probably false), 
     - "PS+" (possibly true), "PS-" (possibly false), "Uu" (unspecified).
   - Higher certainty and verifiability contribute to a **'worth checking'** classification.

3. **Event Selecting Predicates (ESP)**: 
   - Look for predicates that introduce claims (SIP - source introducing predicate) vs. those that do not (NSIP or irrelevant).
   - If a claim is introduced through an authoritative source, it is more likely to be **'worth checking'**.

4. **Agencies**: Analyze agency-related features:
   - Who is performing the action? (e.g., "government", "scientists", "the speaker").
   - Consider **animacy** (human, animate, inanimate) and **morphology** (1st, 2nd, or 3rd person subject).
   - If the sentence refers to an authoritative or external source, it increases check-worthiness.

5. **Stance Confidence Level**: 
   - Look at the level of certainty in the statement: 
     - "high", "mid", or "low".
   - Higher certainty increases the likelihood that the statement is **'worth checking'**.

6. **Stance Type**: 
   - Determine whether the stance is **epistemic (knowledge-based)** or **effective (action-based)**.
   - Epistemic claims are more likely to be **'worth checking'**.

7. **Polarity**:
   - Is the sentence **positive, negative, or underspecified**?
   - Sentences with extreme polarity (strongly positive or negative) may indicate check-worthiness.

8. **Hedges**: 
   - Look for words indicating uncertainty (e.g., "maybe", "possibly").
   - If uncertainty is high, classify as **'not worth checking'** or **'not a factual proposition'**.

9. **Quantities**: 
   - If the sentence contains numerical values, proportions, or statistics, it is more likely **'worth checking'**.

10. **Named Entities**: 
    - Presence of names, places, organizations can indicate check-worthiness.

11. **Protocol Anchors**: 
    - Words linking the event to something specific in a protocol increase **'worth checking'** potential.

12. **Time Expressions**: 
    - Statements about past or future events should be considered for check-worthiness.
---
### **Response Format**
Always respond with only the label: **"worth checking"**, **"not worth checking"**, or **"not a factual proposition"** and nothing else.

'''
        response = client.chat.completions.create(
            model=model_name,
            messages=[
                {"role": "system", "content": system_content},
                {"role": "user", "content": f"{task} Sentence: {sentence_input}"}
            ],
            response_format={
                "type": "text"
            },
            temperature=0,
            max_completion_tokens=2048,
            top_p=1,
            frequency_penalty=0,
            presence_penalty=0
        )

        answer = response.choices[0].message.content
        print(answer)
        predicted_labels.append(answer)
  return predicted_labels


def only_check_worthiness_score_task_real_examples_prompt(sentences, ignore_not_a_factual_claim=False, hebrew_prompt_english_labels = False, hebrew_prompt_and_labels = False):
  # model_name = "gpt-4o"
  model_name = "gpt-4"
  ignore_not_a_factual_claim_text = ""
  hebrew_prompt_english_labels_text = ""
  hebrew_prompt_and_labels_text = ""
  predicted_labels = []
  if ignore_not_a_factual_claim:
    ignore_not_a_factual_claim_text = "ignore_not_a_factual_claim"
  if hebrew_prompt_english_labels:
    hebrew_prompt_english_labels_text = "hebrew_prompt_english_labels"
  if hebrew_prompt_and_labels:
    hebrew_prompt_and_labels_text = "hebrew_prompt_and_labels"
  print(f'task is: check_worthiness_score with real examples,{ignore_not_a_factual_claim_text} {hebrew_prompt_and_labels_text} {hebrew_prompt_english_labels_text} with model: {model_name}')
  for sentence in sentences:
    if ignore_not_a_factual_claim:
      task = (
        "Classify the following sentence into one of the following categories: "
        "'worth checking', 'not worth checking'. "
      )
    else:
      task = (
        "Classify the following sentence into one of three categories: 'worth checking', 'not worth checking', or 'not a factual proposition'. "
      )
    sentence_input = sentence
    if ignore_not_a_factual_claim:
      system_content = ("You are a helpful assistant that classifies sentences into one of the following categories: "
                        "'worth checking' or 'not worth checking'. Follow these definitions strictly:\n"
                        "1. 'worth checking' - Sentences that include claims or propositions that can be factually verified or debunked.\n"
                        "   Examples:\n"
                        "   - דבר נוסף, במהלך הכנסת השמונה-עשרה קידם משרד המשפטים רפורמות בתחומי חקיקה שונים: בתחום הפלילי, בתחום הביטחוני, בתחום האזרחי, בתחום הכלכלי-פיסקלי, בתחום המינהלי והבין-לאומי.\n"
                        "   - התקציב הוא הדרך הטובה ביותר לממשלה לבטא את השקפתה, את סדרי העדיפויות שלה, את הדרך שלה לעצב את החברה.\n"
                        "   - 70 נפש מבני משפחתי נעקדו על קידוש השם במקום הנורא ששמו אושוויץ-בירקנאו.\n"
                        "   - מדינת ישראל עכשיו בת 66, נדמה לי.\n"
                        "2. 'not worth checking' - Sentences that cannot or do not need to be factually verified. This includes:\n"
                        "  - Obvious truths or widely accepted facts.\n"
                        "  - Claims that cannot be verified or are subjective opinions. \n"
                        "  - Non-factual expressions such as questions, commands, or exclamations."
                        "   Examples:\n"
                        "   - אני כאן ממש אומרת שיש לנו אחריות משותפת בעניין הזה מול משרד האוצר כדי לקדם את זה.\n"
                        "   - את כיושבת-ראש הוועדה לפניות הציבור בוודאי יודעת כמה זה מעיק על הציבור כשהוא מרגיש שהוא משלם בשביל אותו שירות סכומים שונים.\n"
                        "   - אני לא רוצה אפילו לדמיין.\n"
                        "   - הוא מפחד מכל דבר.\n"
                        "   - לכם, חברי הכנסת התשע-עשרה, אמליץ ואבקש: התרכזו בהצעות חוק חשובות, מהותיות, רפורמות מקיפות וחשובות.\n"
                        "   - איפה שר האוצר שיעלה פה וישיב על הדברים?\n"
                        "   - תודה.\n"
                        "   - אז תעשו לי טובה, תפסיקו עם החברתיות הזאת של הבחור החתיך, והבחור הנחמד, וההוא שנתן ביטחון סלולרי לאנשים במדינת ישראל.\n"
                        "Always respond with only the label ('worth checking', 'not worth checking') and nothing else."
                        )
    else:
      system_content = (
        "You are a helpful assistant that classifies sentences into one of the following categories: "
        "'worth checking', 'not worth checking', or 'not a factual proposition'. Follow these definitions strictly:\n"
        "1. 'worth checking' - Sentences that include claims or propositions that can be factually verified or debunked.\n"
        "   Examples:\n"
        "   - דבר נוסף, במהלך הכנסת השמונה-עשרה קידם משרד המשפטים רפורמות בתחומי חקיקה שונים: בתחום הפלילי, בתחום הביטחוני, בתחום האזרחי, בתחום הכלכלי-פיסקלי, בתחום המינהלי והבין-לאומי.\n"
        "   - התקציב הוא הדרך הטובה ביותר לממשלה לבטא את השקפתה, את סדרי העדיפויות שלה, את הדרך שלה לעצב את החברה.\n"
        "   - 70 נפש מבני משפחתי נעקדו על קידוש השם במקום הנורא ששמו אושוויץ-בירקנאו.\n"
        "   - מדינת ישראל עכשיו בת 66, נדמה לי.\n"
        "2. 'not worth checking' - Sentences that include obvious truths or widely accepted facts, or subjective opinions or claims that cannot be verified or are not important to check.\n"
        "   Examples:\n"
        "   - אני כאן ממש אומרת שיש לנו אחריות משותפת בעניין הזה מול משרד האוצר כדי לקדם את זה.\n"
        "   - את כיושבת-ראש הוועדה לפניות הציבור בוודאי יודעת כמה זה מעיק על הציבור כשהוא מרגיש שהוא משלם בשביל אותו שירות סכומים שונים.\n"
        "   - אני לא רוצה אפילו לדמיין.\n"
        "   - הוא מפחד מכל דבר.\n"
        "3. 'not a factual proposition' - Sentences that do not propose a factual claim. This includes questions, commands, or exclamations.\n"
        "   Examples:\n"
        "   - לכם, חברי הכנסת התשע-עשרה, אמליץ ואבקש: התרכזו בהצעות חוק חשובות, מהותיות, רפורמות מקיפות וחשובות.\n"
        "   - איפה שר האוצר שיעלה פה וישיב על הדברים?\n"
        "   - תודה.\n"
        "   - אז תעשו לי טובה, תפסיקו עם החברתיות הזאת של הבחור החתיך, והבחור הנחמד, וההוא שנתן ביטחון סלולרי לאנשים במדינת ישראל.\n"
        "Always respond with only the label ('worth checking', 'not worth checking', or 'not a factual proposition') and nothing else."
      )

    response = client.chat.completions.create(
      model=model_name,
      messages=[
        {"role": "system", "content": system_content},
        {"role": "user", "content": f"{task} Sentence: {sentence_input}"}
      ],
      response_format={
        "type": "text"
      },
      temperature=0,
      max_completion_tokens=2048,
      top_p=1,
      frequency_penalty=0,
      presence_penalty=0
    )

    answer = response.choices[0].message.content
    print(answer)
    predicted_labels.append(answer)
  return predicted_labels

def adjust_predicted_label(predicted_label):
  adjusted_predicted_label = predicted_label.lower()
  if "proposition" in adjusted_predicted_label or "טענה" in adjusted_predicted_label:
    adjusted_predicted_label = 'not a factual proposition'
  elif "not" in adjusted_predicted_label or "לא" in adjusted_predicted_label:
    adjusted_predicted_label = 'not worth checking'
  elif "worth" in adjusted_predicted_label or "שווה" in adjusted_predicted_label:
    adjusted_predicted_label = 'worth checking'
  else:
    print(f'label could not be adjusted')
  return adjusted_predicted_label
def calc_accuracy(labels, predicted_labels):
  if len(string_labels) != len(predicted_labels):
    print(f'Error! not same number of labels!')
  good_predictions_counter = 0
  for label, predicted_label in zip(labels, predicted_labels):
    adjusted_predicted_label = adjust_predicted_label(predicted_label)
    if label == adjusted_predicted_label:
      good_predictions_counter +=1

  accuracy = good_predictions_counter/len(predicted_labels)
  print(f'accuracy: {accuracy}')

if __name__ == '__main__':

  raw_test_set = load_from_disk("factuality_10_perc_test_set")
  sentences = raw_test_set["text"]  # original sentence
  numeric_labels  = raw_test_set["label"]  # numeric label
  original_labels = ['worth checking', 'not worth checking', 'not a factual proposition']
  id2label = {i: label for i, label in enumerate(sorted(original_labels))}
  string_labels = [id2label[i] for i in numeric_labels]


  # predicted_labels = only_check_worthiness_score_task_zero_shot(sentences, ignore_not_a_factual_claim=False, hebrew_prompt_english_labels=False, hebrew_prompt_and_labels=False)
  predicted_labels = only_check_worthiness_score_task_explanation_prompt(sentences, ignore_not_a_factual_claim=False)
  # predicted_labels = only_check_worthiness_score_task_real_examples_prompt(sentences,ignore_not_a_factual_claim=True)
  # predicted_labels = check_worthiness_score_based_on_schema_features_prompt(sentences, ignore_not_a_factual_claim=False)

  calc_accuracy(string_labels, predicted_labels)