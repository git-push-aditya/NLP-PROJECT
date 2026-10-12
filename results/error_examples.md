# Error examples: IndicSBERT, en-en, test split
tau* = 0.67. Most confident errors: false positives = highest-scoring non-duplicates; false negatives = lowest-scoring duplicates.

## False positives (predicted duplicate, actually not)

**FP1** pair_id=22374 score=0.994 bucket=B4
- Q1: Can someone translate the video?
- Q2: Can someone translate this video?
- Diagnosis: Likely label noise: 'the video' vs 'this video' is a near-identical paraphrase; the model is arguably right.

**FP2** pair_id=10400 score=0.993 bucket=B5
- Q1: What is the salary of Applications developer II in Oracle, India JD Edwards vertical with 3 years experience?
- Q2: What is the salary of Applications developer II in Oracle, India JD Edwards vertical with 1.3 years experience?
- Diagnosis: Numeric substitution (3 years vs 1.3 years) changes the answer, but the embedding ignores the number: lexical shortcut.

**FP3** pair_id=15663 score=0.993 bucket=B4
- Q1: How do I conduct a usability test?
- Q2: How do I perform a usability test?
- Diagnosis: Synonym swap (conduct/perform); semantically identical, so probable QQP label noise.

**FP4** pair_id=20823 score=0.992 bucket=B4
- Q1: What is the biggest achievement of your life?
- Q2: What is the biggest achievement in your life?
- Diagnosis: Preposition change (of/in your life); the same question, so probable label noise.

**FP5** pair_id=36856 score=0.989 bucket=B5
- Q1: What does this quote actually mean?
- Q2: What does this quote mean?
- Diagnosis: Adverb 'actually' dropped; no change in information need, so probable label noise.

**FP6** pair_id=23741 score=0.987 bucket=B4
- Q1: How can I develop a sense of humour?
- Q2: How can I get a good sense of humour?
- Diagnosis: 'develop' vs 'get' a sense of humour; near-paraphrase, label arguably wrong.

**FP7** pair_id=34030 score=0.985 bucket=B4
- Q1: What are some accessories that would be perfect with this latex maids outfit?
- Q2: What accessories would be perfect with this latex maids outfit?
- Diagnosis: Only 'some' removed; the same question, probable label noise.

**FP8** pair_id=3731 score=0.984 bucket=B3
- Q1: What is computer memory?
- Q2: What does computer memory hold?
- Diagnosis: Definition ('what is') vs contents ('what does it hold'): related topic, different information need.

**FP9** pair_id=28636 score=0.975 bucket=B3
- Q1: Should psychiatry be reformed?
- Q2: Can psychiatry be reformed much?
- Diagnosis: Different scope ('should' vs 'can ... much'); a modality shift the encoder smooths over.

**FP10** pair_id=27845 score=0.970 bucket=B3
- Q1: What would Jon have done differently as King in the North instead of Robb?
- Q2: What will Jon do differently than Robb as King in the North?
- Diagnosis: Past counterfactual vs future prediction about Jon/Robb; tense/modality ignored, topic overlap dominates.

## False negatives (predicted not duplicate, actually duplicate)

**FN1** pair_id=5911 score=0.380 bucket=B3
- Q1: How so I ask questions on Quora?
- Q2: How can I ask questions over here?
- Diagnosis: Typo ('so') plus 'on Quora' vs 'over here' (deictic reference): the model cannot resolve the context.

**FN2** pair_id=8912 score=0.396 bucket=B2
- Q1: Do Quora users still see questions that are marked as needing improvement?
- Q2: Most of my questions are marked as needing improvement. Where can I seek professional help?
- Diagnosis: Different surface questions; the duplicate label depends on Quora-specific context, a weak label.

**FN3** pair_id=31539 score=0.402 bucket=B2
- Q1: Who will win the election TRUMP or CLINTON?
- Q2: Who do you think will win the U.S. election in November?
- Diagnosis: Names (Trump/Clinton) dropped in Q2; Q2 is more general, so the duplicate label is loose.

**FN4** pair_id=16349 score=0.404 bucket=B2
- Q1: How can I join Indian army after BBA?
- Q2: How do I join defence after graduation in mechanical engineer?
- Diagnosis: Specific degree (BBA) vs another (mechanical): similar goal but different details, a loose label.

**FN5** pair_id=19130 score=0.407 bucket=B3
- Q1: Who do you think will win the U.S. election in November?
- Q2: Who do you think is going to win the 2016 presidential election?
- Diagnosis: Same intent with different wording ('in November' vs '2016 presidential'); the model underrates event paraphrase.

**FN6** pair_id=1262 score=0.429 bucket=B3
- Q1: I want to invest Rs 20,000 in mutual fund. Which mutual fund is best to invest?
- Q2: I've decided to invest monthly 5k in mutual fund. Which is the best mutual fund available right now?
- Diagnosis: Different figures (Rs 20,000 lump sum vs 5k monthly) with the same intent: elaboration noise.

**FN7** pair_id=15536 score=0.435 bucket=B3
- Q1: Who will be USA next president, Trump or Clinton?
- Q2: Who will be the next president of USA 2016?
- Diagnosis: Short and long phrasing of 'next US president'; low lexical overlap, real semantic abstraction missed.

**FN8** pair_id=11810 score=0.438 bucket=B2
- Q1: Oneplus 3 review from Indian user?
- Q2: As a user what is your review of OnePlus 3 including the battery part? Thank you
- Diagnosis: Extra detail (battery) and politeness in Q2 dilutes the embedding similarity.

**FN9** pair_id=25751 score=0.442 bucket=B2
- Q1: Why is GST amendment a gamechanger for Indian Economy?
- Q2: How is GST beneficial for the country? How would it help to improve the country's economy?
- Diagnosis: Same topic (GST benefit), different framing ('gamechanger' vs 'beneficial'); abstract paraphrase missed.

**FN10** pair_id=5949 score=0.443 bucket=B2
- Q1: Why does my question keep getting marked as needing improvement?
- Q2: Why is every question I post on Quora marked as needing clarification?
- Diagnosis: Same complaint with different vocabulary ('improvement'/'clarification'); platform-specific jargon.
