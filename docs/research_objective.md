# Research objective

Assess whether supervised fine-tuning improves correctness, evidence grounding, citation support and balanced abstention for UK motor insurance Q&A using 1B–3B instruct models.

Compare Base versus SFT without retrieval first. Then compare Base + RAG versus SFT + RAG using identical frozen retrieved passages. Closed-book arms diagnose guessing and abstention when insurer-specific evidence is unavailable. Citation metrics are N/A in those arms.

Predecessor work supplied a mixed insurance RAG system and a protected 140-question test, including 76 motor questions. Historical 21-example SFT runs remain separate from future production-data training. The final project deadline is 18 December 2026.
