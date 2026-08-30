"""Builds the v1 dataset: simple_recall, temporal_update, multi_hop, long_term_retention (matched + crossed). Run: python -m benchmark.dataset_generator"""

import json
import random
from datetime import datetime, timedelta

try:
    from .. import config
except (ImportError, ValueError):
    import sys
    from pathlib import Path
    sys.path.append(str(Path(__file__).resolve().parents[1]))
    import config

# scenario dict shape: category, memories: [{key, subcategory, day, text}],
# question: {subcategory, day, text, expected_answer, relevant_keys, answer_keys, superseded_keys}

SIMPLE_RECALL = [
    {  # gap 90
        "category": "simple_recall",
        "memories": [{"key": "s1_fact", "subcategory": "family", "day": 2,
                      "text": "My sister's name is Maya."}],
        "question": {"subcategory": "family", "day": 92,
                     "text": "What is my sister's name?",
                     "expected_answer": "Maya.",
                     "relevant_keys": ["s1_fact"], "answer_keys": ["s1_fact"]},
    },
    {  # gap 30
        "category": "simple_recall",
        "memories": [{"key": "s2_fact", "subcategory": "object_location", "day": 4,
                      "text": "I keep the spare house key in the blue ceramic bowl by the front door."}],
        "question": {"subcategory": "object_location", "day": 34,
                     "text": "Where is the spare house key?",
                     "expected_answer": "In the blue ceramic bowl by the front door.",
                     "relevant_keys": ["s2_fact"], "answer_keys": ["s2_fact"]},
    },
    {  # gap 180
        "category": "simple_recall",
        "memories": [{"key": "s3_fact", "subcategory": "food_allergy", "day": 1,
                      "text": "I'm allergic to shellfish."}],
        "question": {"subcategory": "food_allergy", "day": 181,
                     "text": "What am I allergic to?",
                     "expected_answer": "Shellfish.",
                     "relevant_keys": ["s3_fact"], "answer_keys": ["s3_fact"]},
    },
    {  # gap 7
        "category": "simple_recall",
        "memories": [{"key": "s4_fact", "subcategory": "family", "day": 6,
                      "text": "My son's name is Elias."}],
        "question": {"subcategory": "family", "day": 13,
                     "text": "What is my son's name?",
                     "expected_answer": "Elias.",
                     "relevant_keys": ["s4_fact"], "answer_keys": ["s4_fact"]},
    },
    {  # gap 90
        "category": "simple_recall",
        "memories": [{"key": "s5_fact", "subcategory": "recurring_household", "day": 8,
                      "text": "The thermostat filter needs to be replaced every 3 months."}],
        "question": {"subcategory": "recurring_household", "day": 98,
                     "text": "How often should the thermostat filter be replaced?",
                     "expected_answer": "Every 3 months.",
                     "relevant_keys": ["s5_fact"], "answer_keys": ["s5_fact"]},
    },
    {  # gap 30
        "category": "simple_recall",
        "memories": [{"key": "s6_fact", "subcategory": "object_location", "day": 10,
                      "text": "I keep the good scissors in the second kitchen drawer."}],
        "question": {"subcategory": "object_location", "day": 40,
                     "text": "Where are the good scissors?",
                     "expected_answer": "In the second kitchen drawer.",
                     "relevant_keys": ["s6_fact"], "answer_keys": ["s6_fact"]},
    },
    {  # gap 1
        "category": "simple_recall",
        "memories": [{"key": "s7_fact", "subcategory": "food_preference", "day": 12,
                      "text": "I take my tea with two sugars."}],
        "question": {"subcategory": "food_preference", "day": 13,
                     "text": "How do I take my tea?",
                     "expected_answer": "With two sugars.",
                     "relevant_keys": ["s7_fact"], "answer_keys": ["s7_fact"]},
    },
    {  # gap 7
        "category": "simple_recall",
        "memories": [{"key": "s8_fact", "subcategory": "appliance_preference", "day": 14,
                      "text": "My preferred dishwasher setting is eco mode."}],
        "question": {"subcategory": "appliance_preference", "day": 21,
                     "text": "Which dishwasher setting do I prefer?",
                     "expected_answer": "Eco mode.",
                     "relevant_keys": ["s8_fact"], "answer_keys": ["s8_fact"]},
    },
    {  # gap 180
        "category": "simple_recall",
        "memories": [{"key": "s9_fact", "subcategory": "recurring_household", "day": 3,
                      "text": "Our WiFi network is called Cedarwood."}],
        "question": {"subcategory": "recurring_household", "day": 183,
                     "text": "What is our WiFi network called?",
                     "expected_answer": "Cedarwood.",
                     "relevant_keys": ["s9_fact"], "answer_keys": ["s9_fact"]},
    },
    {  # gap 30
        "category": "simple_recall",
        "memories": [{"key": "s10_fact", "subcategory": "family", "day": 9,
                      "text": "My mother's birthday is March 12th."}],
        "question": {"subcategory": "family", "day": 39,
                     "text": "When is my mother's birthday?",
                     "expected_answer": "March 12th.",
                     "relevant_keys": ["s10_fact"], "answer_keys": ["s10_fact"]},
    },
]

# For temporal cases: relevant_keys = [old, new] (both are on-topic, matching the
# spec example). answer_keys = [new] (the one required for the correct answer).
# superseded_keys = [old].
TEMPORAL_UPDATE = [
    {
        "category": "temporal_update",
        "memories": [
            {"key": "t1_old", "subcategory": "food_preference", "day": 5,
             "text": "I drink my coffee black."},
            {"key": "t1_new", "subcategory": "food_preference", "day": 40,
             "text": "I've switched to oat milk in my coffee."},
        ],
        "question": {"subcategory": "food_preference", "day": 70,
                     "text": "How should you make my coffee?",
                     "expected_answer": "With oat milk.",
                     "relevant_keys": ["t1_old", "t1_new"],
                     "answer_keys": ["t1_new"], "superseded_keys": ["t1_old"]},
    },
    {
        "category": "temporal_update",
        "memories": [
            {"key": "t2_old", "subcategory": "wakeup_time", "day": 7,
             "text": "I usually wake up at 7 AM."},
            {"key": "t2_new", "subcategory": "wakeup_time", "day": 45,
             "text": "I've started waking up at 6:30 AM now."},
        ],
        "question": {"subcategory": "wakeup_time", "day": 75,
                     "text": "What time do I wake up?",
                     "expected_answer": "6:30 AM.",
                     "relevant_keys": ["t2_old", "t2_new"],
                     "answer_keys": ["t2_new"], "superseded_keys": ["t2_old"]},
    },
    {
        "category": "temporal_update",
        "memories": [
            {"key": "t3_old", "subcategory": "work_schedule", "day": 11,
             "text": "I work from home on Fridays."},
            {"key": "t3_new", "subcategory": "work_schedule", "day": 50,
             "text": "I now go into the office on Fridays."},
        ],
        "question": {"subcategory": "work_schedule", "day": 80,
                     "text": "Where do I work on Fridays?",
                     "expected_answer": "At the office.",
                     "relevant_keys": ["t3_old", "t3_new"],
                     "answer_keys": ["t3_new"], "superseded_keys": ["t3_old"]},
    },
    {
        "category": "temporal_update",
        "memories": [
            {"key": "t4_old", "subcategory": "room_temperature", "day": 15,
             "text": "I like the bedroom at 20 degrees."},
            {"key": "t4_new", "subcategory": "room_temperature", "day": 55,
             "text": "I now prefer the bedroom at 18 degrees at night."},
        ],
        "question": {"subcategory": "room_temperature", "day": 85,
                     "text": "What temperature should the bedroom be at night?",
                     "expected_answer": "18 degrees.",
                     "relevant_keys": ["t4_old", "t4_new"],
                     "answer_keys": ["t4_new"], "superseded_keys": ["t4_old"]},
    },
    {
        "category": "temporal_update",
        "memories": [
            {"key": "t5_old", "subcategory": "object_location", "day": 18,
             "text": "I keep my vitamins in the kitchen cupboard."},
            {"key": "t5_new", "subcategory": "object_location", "day": 60,
             "text": "I moved my vitamins to the bathroom cabinet."},
        ],
        "question": {"subcategory": "object_location", "day": 88,
                     "text": "Where are my vitamins?",
                     "expected_answer": "In the bathroom cabinet.",
                     "relevant_keys": ["t5_old", "t5_new"],
                     "answer_keys": ["t5_new"], "superseded_keys": ["t5_old"]},
    },
    {
        "category": "temporal_update",
        "memories": [
            {"key": "t6_old", "subcategory": "routine", "day": 20,
             "text": "I walk the dog in the morning."},
            {"key": "t6_new", "subcategory": "routine", "day": 62,
             "text": "I now walk the dog in the evening."},
        ],
        "question": {"subcategory": "routine", "day": 90,
                     "text": "When do I walk the dog?",
                     "expected_answer": "In the evening.",
                     "relevant_keys": ["t6_old", "t6_new"],
                     "answer_keys": ["t6_new"], "superseded_keys": ["t6_old"]},
    },
    {
        "category": "temporal_update",
        "memories": [
            {"key": "t7_old", "subcategory": "food_preference", "day": 22,
             "text": "I usually have toast for breakfast."},
            {"key": "t7_new", "subcategory": "food_preference", "day": 65,
             "text": "I've switched to yogurt and fruit for breakfast."},
        ],
        "question": {"subcategory": "food_preference", "day": 95,
                     "text": "What do I have for breakfast?",
                     "expected_answer": "Yogurt and fruit.",
                     "relevant_keys": ["t7_old", "t7_new"],
                     "answer_keys": ["t7_new"], "superseded_keys": ["t7_old"]},
    },
    {
        "category": "temporal_update",
        "memories": [
            {"key": "t8_old", "subcategory": "routine", "day": 25,
             "text": "I go to the gym on Mondays and Wednesdays."},
            {"key": "t8_new", "subcategory": "routine", "day": 68,
             "text": "I've changed my gym days to Tuesdays and Thursdays."},
        ],
        "question": {"subcategory": "routine", "day": 98,
                     "text": "Which days do I go to the gym?",
                     "expected_answer": "Tuesdays and Thursdays.",
                     "relevant_keys": ["t8_old", "t8_new"],
                     "answer_keys": ["t8_new"], "superseded_keys": ["t8_old"]},
    },
    {
        "category": "temporal_update",
        "memories": [
            {"key": "t9_old", "subcategory": "object_location", "day": 28,
             "text": "My parking spot is number 14."},
            {"key": "t9_new", "subcategory": "object_location", "day": 70,
             "text": "My parking spot has been reassigned to number 22."},
        ],
        "question": {"subcategory": "object_location", "day": 100,
                     "text": "Which parking spot is mine?",
                     "expected_answer": "Number 22.",
                     "relevant_keys": ["t9_old", "t9_new"],
                     "answer_keys": ["t9_new"], "superseded_keys": ["t9_old"]},
    },
    {
        "category": "temporal_update",
        "memories": [
            {"key": "t10_old", "subcategory": "appliance_preference", "day": 30,
             "text": "I keep my alarm volume on low."},
            {"key": "t10_new", "subcategory": "appliance_preference", "day": 72,
             "text": "I've turned my alarm volume up to high now."},
        ],
        "question": {"subcategory": "appliance_preference", "day": 102,
                     "text": "What volume is my alarm set to?",
                     "expected_answer": "High.",
                     "relevant_keys": ["t10_old", "t10_new"],
                     "answer_keys": ["t10_new"], "superseded_keys": ["t10_old"]},
    },
]

# Multi-hop: every question needs >= 2 memories combined. relevant_keys and
# answer_keys are the same set here (all listed memories are needed).
MULTI_HOP = [
    {
        "category": "multi_hop",
        "memories": [
            {"key": "mh1_a", "subcategory": "visitor_preference", "day": 33,
             "text": "My friend Priya is vegetarian."},
            {"key": "mh1_b", "subcategory": "visitor_preference", "day": 34,
             "text": "Priya is coming over for dinner on Saturday."},
        ],
        "question": {"subcategory": "visitor_preference", "day": 36,
                     "text": "Is there anything I should avoid serving Priya at dinner on Saturday?",
                     "expected_answer": "Meat \u2014 Priya is vegetarian.",
                     "relevant_keys": ["mh1_a", "mh1_b"], "answer_keys": ["mh1_a", "mh1_b"]},
    },
    {
        "category": "multi_hop",
        "memories": [
            {"key": "mh2_a", "subcategory": "member_allergy", "day": 35,
             "text": "My daughter's name is Zoe."},
            {"key": "mh2_b", "subcategory": "member_allergy", "day": 37,
             "text": "Zoe is lactose intolerant."},
        ],
        "question": {"subcategory": "member_allergy", "day": 41,
                     "text": "What should I avoid giving my daughter?",
                     "expected_answer": "Dairy \u2014 Zoe is lactose intolerant.",
                     "relevant_keys": ["mh2_a", "mh2_b"], "answer_keys": ["mh2_a", "mh2_b"]},
    },
    {
        "category": "multi_hop",
        "memories": [
            {"key": "mh3_a", "subcategory": "ownership_location", "day": 38,
             "text": "The blue suitcase is mine."},
            {"key": "mh3_b", "subcategory": "ownership_location", "day": 42,
             "text": "The blue suitcase is stored in the attic."},
        ],
        "question": {"subcategory": "ownership_location", "day": 50,
                     "text": "Where is my suitcase?",
                     "expected_answer": "In the attic.",
                     "relevant_keys": ["mh3_a", "mh3_b"], "answer_keys": ["mh3_a", "mh3_b"]},
    },
    {
        "category": "multi_hop",
        "memories": [
            {"key": "mh4_a", "subcategory": "visitor_event", "day": 44,
             "text": "My parents are visiting on the 20th."},
            {"key": "mh4_b", "subcategory": "visitor_event", "day": 46,
             "text": "My dad uses a wheelchair."},
        ],
        "question": {"subcategory": "visitor_event", "day": 52,
                     "text": "Do I need to prepare anything for accessibility when my parents visit on the 20th?",
                     "expected_answer": "Yes \u2014 step-free access, because your dad uses a wheelchair.",
                     "relevant_keys": ["mh4_a", "mh4_b"], "answer_keys": ["mh4_a", "mh4_b"]},
    },
    {
        "category": "multi_hop",
        "memories": [
            {"key": "mh5_a", "subcategory": "routine_preference", "day": 48,
             "text": "I go for a run every morning at 6."},
            {"key": "mh5_b", "subcategory": "routine_preference", "day": 49,
             "text": "After my morning run I like a protein smoothie."},
        ],
        "question": {"subcategory": "routine_preference", "day": 54,
                     "text": "What should be ready for me after my morning run?",
                     "expected_answer": "A protein smoothie.",
                     "relevant_keys": ["mh5_a", "mh5_b"], "answer_keys": ["mh5_a", "mh5_b"]},
    },
    {
        "category": "multi_hop",
        "memories": [
            {"key": "mh6_a", "subcategory": "visitor_preference", "day": 51,
             "text": "Sam is coming to the barbecue."},
            {"key": "mh6_b", "subcategory": "visitor_preference", "day": 53,
             "text": "Sam doesn't drink alcohol."},
        ],
        "question": {"subcategory": "visitor_preference", "day": 58,
                     "text": "What drink should I have ready for Sam at the barbecue?",
                     "expected_answer": "A non-alcoholic drink.",
                     "relevant_keys": ["mh6_a", "mh6_b"], "answer_keys": ["mh6_a", "mh6_b"]},
    },
    {
        "category": "multi_hop",
        "memories": [
            {"key": "mh7_a", "subcategory": "member_routine", "day": 56,
             "text": "My grandmother takes her heart medication at 8 PM."},
            {"key": "mh7_b", "subcategory": "member_routine", "day": 57,
             "text": "My grandmother is staying with us this week."},
        ],
        "question": {"subcategory": "member_routine", "day": 61,
                     "text": "Is there anything I should remember about my grandmother's evening routine this week?",
                     "expected_answer": "Her heart medication at 8 PM.",
                     "relevant_keys": ["mh7_a", "mh7_b"], "answer_keys": ["mh7_a", "mh7_b"]},
    },
    {
        "category": "multi_hop",
        "memories": [
            {"key": "mh8_a", "subcategory": "ownership_location", "day": 59,
             "text": "The good camera belongs to me."},
            {"key": "mh8_b", "subcategory": "ownership_location", "day": 63,
             "text": "I keep the good camera in the hallway closet."},
        ],
        "question": {"subcategory": "ownership_location", "day": 66,
                     "text": "Where is my camera?",
                     "expected_answer": "In the hallway closet.",
                     "relevant_keys": ["mh8_a", "mh8_b"], "answer_keys": ["mh8_a", "mh8_b"]},
    },
    {
        "category": "multi_hop",
        "memories": [
            {"key": "mh9_a", "subcategory": "routine_preference", "day": 64,
             "text": "Book club meets at my place on the first Monday of the month."},
            {"key": "mh9_b", "subcategory": "routine_preference", "day": 67,
             "text": "For book club I always serve tea."},
        ],
        "question": {"subcategory": "routine_preference", "day": 71,
                     "text": "What should I prepare to serve for the gathering on the first Monday?",
                     "expected_answer": "Tea.",
                     "relevant_keys": ["mh9_a", "mh9_b"], "answer_keys": ["mh9_a", "mh9_b"]},
    },
    {
        "category": "multi_hop",
        "memories": [
            {"key": "mh10_a", "subcategory": "visitor_preference", "day": 69,
             "text": "My brother Leo visits every Sunday."},
            {"key": "mh10_b", "subcategory": "visitor_preference", "day": 73,
             "text": "Leo is gluten-free."},
        ],
        "question": {"subcategory": "visitor_preference", "day": 78,
                     "text": "What kind of food should I make for my Sunday visitor?",
                     "expected_answer": "Gluten-free food.",
                     "relevant_keys": ["mh10_a", "mh10_b"], "answer_keys": ["mh10_a", "mh10_b"]},
    },
]

# each fact buried under filler_count unrelated memories, gap = age in days
RETENTION_BUCKETS = [
    {"bucket": "1_day", "gap_days": 1, "filler_count": 5, "design": "matched"},
    {"bucket": "7_days", "gap_days": 7, "filler_count": 10, "design": "matched"},
    {"bucket": "30_days", "gap_days": 30, "filler_count": 20, "design": "matched"},
    {"bucket": "90_days", "gap_days": 90, "filler_count": 35, "design": "matched"},
    {"bucket": "180_days", "gap_days": 180, "filler_count": 55, "design": "matched"},
]

# decouples age from clutter (2x2: age_level x clutter_level) so the two effects can be told apart
CROSSED_BUCKETS = [
    {"bucket": "crossed_young_sparse", "gap_days": 10, "filler_count": 8,
     "design": "crossed", "age_level": "low", "clutter_level": "low"},
    {"bucket": "crossed_young_dense", "gap_days": 10, "filler_count": 75,
     "design": "crossed", "age_level": "low", "clutter_level": "high"},
    {"bucket": "crossed_old_sparse", "gap_days": 120, "filler_count": 8,
     "design": "crossed", "age_level": "high", "clutter_level": "low"},
    {"bucket": "crossed_old_dense", "gap_days": 120, "filler_count": 75,
     "design": "crossed", "age_level": "high", "clutter_level": "high"},
]

# (subcategory, fact_text, question_text, expected_answer) per bucket
RETENTION_CASES = {
    "1_day": [
        ("object_location", "I put the spare batteries in the blue drawer under the television.",
         "Where did I leave those batteries?", "In the blue drawer under the television."),
        ("food_preference", "I'm cutting back on caffeine, so just one coffee in the mornings from now on.",
         "How many coffees should I have in the morning?", "Just one."),
        ("routine", "I'm heading to bed early tonight, around 9:30.",
         "What time did I say I was going to bed?", "Around 9:30."),
        ("appliance_setting", "Keep the living room thermostat at 21 degrees during the day.",
         "What should the living room thermostat be set to during the day?", "21 degrees."),
        ("household_preference", "Please keep the porch light on until midnight.",
         "Until what time should the porch light stay on?", "Until midnight."),
    ],
    "7_days": [
        ("family_detail", "My nephew Oliver is allergic to peanuts.",
         "Is there any allergy I should know about for my nephew?", "Yes, Oliver is allergic to peanuts."),
        ("object_location", "I hung the spare car key on the hook by the garage door.",
         "Where's the extra car key?", "On the hook by the garage door."),
        ("recurring_household", "The recycling gets picked up every other Wednesday.",
         "When does recycling collection happen?", "Every other Wednesday."),
        ("appliance_setting", "I switched the coffee maker to auto-brew at 6:45 AM.",
         "What time does the coffee maker start brewing automatically?", "6:45 AM."),
        ("food_preference", "I've started keeping oat milk instead of regular milk in the fridge.",
         "What kind of milk do I keep in the fridge now?", "Oat milk."),
    ],
    "30_days": [
        ("household_preference", "I'd rather the blinds stay closed in the afternoon to keep the room cool.",
         "Should the blinds be open or closed in the afternoons?", "Closed."),
        ("object_location", "I stored the holiday decorations in a box on the top shelf of the garage.",
         "Where did I put the holiday decorations?", "In a box on the top shelf of the garage."),
        ("routine", "I go for a swim every Tuesday and Thursday evening.",
         "Which days do I usually swim?", "Tuesdays and Thursdays."),
        ("family_detail", "My cousin Rachel is visiting from out of town in a few months and staying in the guest room.",
         "Who's staying in the guest room when they visit?", "My cousin Rachel."),
        ("recurring_household", "The gutters need to be cleaned out twice a year, spring and fall.",
         "How often do the gutters need cleaning?", "Twice a year, in spring and fall."),
    ],
    "90_days": [
        ("object_location", "I keep the extra house key inside the mailbox, taped under the lid.",
         "Where do I keep the extra house key?", "Inside the mailbox, taped under the lid."),
        ("appliance_setting", "I set the water heater to run on eco mode overnight.",
         "What mode does the water heater run on overnight?", "Eco mode."),
        ("food_preference", "I've started eating gluten-free, so avoid ordering regular pasta for me.",
         "What dietary restriction should you keep in mind when ordering food for me?", "Gluten-free."),
        ("family_detail", "My grandfather's favorite chair is the brown recliner in the study, so no one else should sit in it when he visits.",
         "Whose chair is the brown recliner in the study reserved for?", "My grandfather's."),
        ("recurring_household", "The water filter under the sink needs replacing about every four months.",
         "How often should the under-sink water filter be replaced?", "About every four months."),
    ],
    "180_days": [
        ("family_detail", "My daughter's godmother is Aunt Ruth, in case that ever comes up.",
         "Who is my daughter's godmother?", "Aunt Ruth."),
        ("object_location", "I put the passports in the fireproof safe in the closet.",
         "Where are the passports kept?", "In the fireproof safe in the closet."),
        ("household_preference", "I prefer the guest bed made up with the extra-firm pillows, not the soft ones.",
         "What kind of pillows should be on the guest bed?", "The extra-firm pillows."),
        ("routine", "I do a full pantry restock and inventory once every six months.",
         "How often do I restock and check the pantry?", "Once every six months."),
        ("recurring_household", "The septic tank needs to be serviced twice a year.",
         "How often does the septic tank need servicing?", "Twice a year."),
    ],
}

# 3 cases per crossed cell (12 total) — same (subcategory, fact, question,
# answer) shape as RETENTION_CASES, distinct wording/entities from every
# other scenario in the file.
CROSSED_CASES = {
    "crossed_young_sparse": [
        ("object_location", "I put the flashlight in the top drawer of the hallway cabinet.",
         "Where's the flashlight?", "In the top drawer of the hallway cabinet."),
        ("food_preference", "I've started adding cinnamon to my oatmeal every morning.",
         "What do I add to my oatmeal now?", "Cinnamon."),
        ("appliance_setting", "Set the humidifier to run at 45% in the nursery.",
         "What humidity level should the nursery humidifier run at?", "45%."),
    ],
    "crossed_young_dense": [
        ("household_preference", "I like the hallway nightlight left on but dimmed low.",
         "How should the hallway nightlight be set?", "On, but dimmed low."),
        ("routine", "I do my stretching routine right after waking up, before breakfast.",
         "When do I do my stretching?", "Right after waking up, before breakfast."),
        ("recurring_household", "The air conditioning filter gets checked every month.",
         "How often is the air conditioning filter checked?", "Every month."),
    ],
    "crossed_old_sparse": [
        ("family_detail", "My uncle Tom is the one who fixed the back fence last summer.",
         "Who fixed the back fence?", "My uncle Tom."),
        ("object_location", "I keep the spare lightbulbs on the shelf above the washing machine.",
         "Where are the spare lightbulbs?", "On the shelf above the washing machine."),
        ("food_preference", "I stopped eating red meat, so stick to chicken or fish for me.",
         "What kind of meat should be avoided for me now?", "Red meat."),
    ],
    "crossed_old_dense": [
        ("appliance_setting", "I set the fridge to keep the vegetable drawer humidity on high.",
         "What humidity setting is the fridge's vegetable drawer on?", "High."),
        ("household_preference", "I like fresh towels put out every Sunday, not daily.",
         "How often should fresh towels be put out?", "Every Sunday."),
        ("recurring_household", "The car gets its oil changed roughly every 5,000 miles.",
         "How often does the car need an oil change?", "Roughly every 5,000 miles."),
    ],
}

# (subcategory, text) distractor pool for retention filler
FILLER_TEMPLATES = [
    ("grocery_reminder", "Remind me to buy milk and eggs this week."),
    ("grocery_reminder", "We're almost out of coffee filters."),
    ("grocery_reminder", "Add paper towels to the shopping list."),
    ("grocery_reminder", "Don't forget to pick up dog food on the way home."),
    ("weather_question", "What's the weather like today?"),
    ("weather_question", "Will it rain this weekend?"),
    ("weather_question", "Is it going to be cold enough for a jacket tomorrow?"),
    ("weather_question", "How windy is it supposed to get tonight?"),
    ("cleaning_task", "Remind me to vacuum the living room on Saturday."),
    ("cleaning_task", "The bathroom needs a deep clean this week."),
    ("cleaning_task", "Can you schedule the robot vacuum for tomorrow morning?"),
    ("cleaning_task", "I need to wipe down the kitchen counters."),
    ("tv_movie", "What time does the new season start streaming?"),
    ("tv_movie", "Add that documentary to my watchlist."),
    ("tv_movie", "We should watch a movie tonight."),
    ("tv_movie", "Did you see the score from last night's game?"),
    ("appointment", "I have a dentist appointment next Tuesday."),
    ("appointment", "Remind me about my car service appointment."),
    ("appointment", "Schedule a reminder for my haircut on Friday."),
    ("appointment", "I need to call the vet to book a checkup."),
    ("work_schedule_chat", "I'm working from home on Thursday this week."),
    ("work_schedule_chat", "I have a work meeting at 10 AM tomorrow."),
    ("work_schedule_chat", "Remind me to submit my timesheet by Friday."),
    ("work_schedule_chat", "I'll be in the office later than usual today."),
    ("dinner_planning", "What should we make for dinner tonight?"),
    ("dinner_planning", "Let's order pizza on Friday."),
    ("dinner_planning", "I'm thinking of trying a new pasta recipe this weekend."),
    ("dinner_planning", "Can you suggest something quick for dinner?"),
    ("appliance_use", "Start the dishwasher after dinner."),
    ("appliance_use", "The washing machine finished its cycle."),
    ("appliance_use", "Set the oven to preheat to 400 degrees."),
    ("appliance_use", "Turn the thermostat down a couple degrees tonight."),
    ("household_chore", "I need to take out the recycling tonight."),
    ("household_chore", "Remind me to water the plants tomorrow."),
    ("household_chore", "The lawn needs mowing this weekend."),
    ("household_chore", "Change the air filter in the hallway."),
    ("small_talk", "That was a good podcast episode this morning."),
    ("small_talk", "Traffic was pretty bad on the way home today."),
    ("small_talk", "I finally finished that book I was reading."),
    ("small_talk", "It's been a long week."),
]


def _build_retention_scenarios():
    """Builds long_term_retention scenarios: fact + filler memories spread before the question."""
    rng = random.Random(config.SEED)
    shuffled_fillers = FILLER_TEMPLATES.copy()
    rng.shuffle(shuffled_fillers)
    filler_cursor = 0

    scenarios = []
    day_cursor = 200.0
    scenario_gap_days = 3

    all_buckets = RETENTION_BUCKETS + CROSSED_BUCKETS
    all_cases = {**RETENTION_CASES, **CROSSED_CASES}

    for bucket in all_buckets:
        for i, (subcategory, fact_text, question_text, expected_answer) in enumerate(
                all_cases[bucket["bucket"]]):
            fact_key = f"ret_{bucket['bucket']}_{i}_orig"
            fact_day = day_cursor
            question_day = day_cursor + bucket["gap_days"]

            memories = [{"key": fact_key, "subcategory": subcategory, "day": fact_day, "text": fact_text}]
            for j in range(bucket["filler_count"]):
                filler_subcategory, filler_text = shuffled_fillers[filler_cursor % len(shuffled_fillers)]
                filler_cursor += 1
                # spread strictly between fact_day and question_day, never touching either
                offset = bucket["gap_days"] * (j + 1) / (bucket["filler_count"] + 1)
                memories.append({
                    "key": f"{fact_key}_filler_{j}",
                    "subcategory": filler_subcategory,
                    "day": fact_day + offset,
                    "text": filler_text,
                })

            question = {
                "subcategory": subcategory,
                "day": question_day,
                "text": question_text,
                "expected_answer": expected_answer,
                "relevant_keys": [fact_key],
                "answer_keys": [fact_key],
                "retention_bucket": bucket["bucket"],
                "retention_design": bucket["design"],
            }
            if "age_level" in bucket:
                question["age_level"] = bucket["age_level"]
                question["clutter_level"] = bucket["clutter_level"]

            scenarios.append({
                "category": "long_term_retention",
                "memories": memories,
                "question": question,
            })

            day_cursor = question_day + scenario_gap_days

    return scenarios


ALL_SCENARIOS = SIMPLE_RECALL + TEMPORAL_UPDATE + MULTI_HOP + _build_retention_scenarios()


# ===========================================================================
# Compilation: scenarios -> concrete records with global ids/timestamps
# ===========================================================================

def _ts(base: datetime, day: float, minute: int) -> str:
    return (base + timedelta(days=day, minutes=minute)).isoformat()


def build_dataset():
    base = datetime.fromisoformat(config.BASE_DATE)
    counter = 0  # global, so memory and question timestamps never collide

    # --- 1. Collect memories, sort by time, assign m_### ids ---------------
    mem_entries = []
    for sc in ALL_SCENARIOS:
        for m in sc["memories"]:
            mem_entries.append({
                "_key": m["key"],
                "user_id": config.USER_ID,
                "timestamp": _ts(base, m["day"], counter),
                "type": "memory",
                "category": sc["category"],
                "subcategory": m["subcategory"],
                "text": m["text"],
            })
            counter += 1
    mem_entries.sort(key=lambda r: r["timestamp"])

    key_to_id = {}
    for i, m in enumerate(mem_entries, start=1):
        mid = f"m_{i:03d}"
        m["memory_id"] = mid
        key_to_id[m["_key"]] = mid

    # --- 2. Collect questions, sort by time, assign q_### ids --------------
    q_entries = []
    for sc in ALL_SCENARIOS:
        q = sc["question"]
        rec = {
            "user_id": config.USER_ID,
            "timestamp": _ts(base, q["day"], counter),
            "type": "question",
            "category": sc["category"],
            "subcategory": q["subcategory"],
            "text": q["text"],
            "expected_answer": q["expected_answer"],
            "relevant_memory_ids": [key_to_id[k] for k in q["relevant_keys"]],
            "answer_memory_ids": [key_to_id[k] for k in q["answer_keys"]],
        }
        if "superseded_keys" in q:
            rec["superseded_memory_ids"] = [key_to_id[k] for k in q["superseded_keys"]]
        if sc["category"] == "multi_hop":
            rec["n_hops"] = len(q["answer_keys"])
        if sc["category"] == "long_term_retention":
            rec["retention_bucket"] = q["retention_bucket"]
            rec["retention_design"] = q["retention_design"]
            rec["original_memory_id"] = key_to_id[q["relevant_keys"][0]]
            if "age_level" in q:
                rec["age_level"] = q["age_level"]
                rec["clutter_level"] = q["clutter_level"]
        q_entries.append(rec)
        counter += 1
    q_entries.sort(key=lambda r: r["timestamp"])

    for i, q in enumerate(q_entries, start=1):
        q["question_id"] = f"q_{i:03d}"

    # --- 2b. Retention fields computed from the real resolved timestamps ---
    ts_by_id = {m["memory_id"]: m["timestamp"] for m in mem_entries}
    for q in q_entries:
        if q["category"] != "long_term_retention":
            continue
        orig_ts = ts_by_id[q["original_memory_id"]]
        q["memory_age_days"] = (
            datetime.fromisoformat(q["timestamp"]) - datetime.fromisoformat(orig_ts)
        ).days
        q["intervening_memory_count"] = sum(
            1 for m in mem_entries if orig_ts < m["timestamp"] < q["timestamp"]
        )

    # --- 3. Assign session_id: one session per simulated calendar day ------
    for m in mem_entries:
        m.pop("_key", None)
    all_records = mem_entries + q_entries
    all_records.sort(key=lambda r: r["timestamp"])

    session, last_date = 0, None
    for r in all_records:
        date = r["timestamp"][:10]
        if date != last_date:
            session += 1
            last_date = date
        r["session_id"] = session

    return all_records


# ===========================================================================
# Validation + writing
# ===========================================================================

def _ordered(record):
    """Return the record with a stable, readable key order for JSONL output."""
    order = ["user_id", "session_id", "timestamp", "type", "memory_id",
             "question_id", "category", "subcategory", "text", "expected_answer",
             "relevant_memory_ids", "answer_memory_ids", "superseded_memory_ids",
             "n_hops", "original_memory_id", "retention_bucket", "retention_design",
             "age_level", "clutter_level", "memory_age_days", "intervening_memory_count"]
    return {k: record[k] for k in order if k in record}


def validate(records):
    """Sanity checks that make the dataset safe to rely on downstream."""
    mems = [r for r in records if r["type"] == "memory"]
    qs = [r for r in records if r["type"] == "question"]

    counts = {"simple_recall": 0, "temporal_update": 0, "multi_hop": 0, "long_term_retention": 0}
    for q in qs:
        counts[q["category"]] += 1
    assert counts == {"simple_recall": 10, "temporal_update": 10, "multi_hop": 10,
                       "long_term_retention": 37}, counts  # 25 matched + 12 crossed

    mem_ids = {m["memory_id"] for m in mems}
    assert len(mem_ids) == len(mems), "duplicate memory ids"

    ts_by_id = {m["memory_id"]: m["timestamp"] for m in mems}
    all_retention_buckets = RETENTION_BUCKETS + CROSSED_BUCKETS
    bucket_gap_days = {b["bucket"]: b["gap_days"] for b in all_retention_buckets}
    bucket_filler_counts = {b["bucket"]: b["filler_count"] for b in all_retention_buckets}
    bucket_design = {b["bucket"]: b["design"] for b in all_retention_buckets}

    for q in qs:
        # every referenced memory exists...
        for mid in q["relevant_memory_ids"]:
            assert mid in mem_ids, f"{q['question_id']} references missing {mid}"
            # ...and was stored strictly before the question was asked
            assert ts_by_id[mid] < q["timestamp"], (
                f"{q['question_id']} asked before memory {mid} existed")
        # multi-hop really needs >= 2 memories
        if q["category"] == "multi_hop":
            assert len(q["answer_memory_ids"]) >= 2, q["question_id"]
        # retention metadata is internally consistent with how it was built
        if q["category"] == "long_term_retention":
            bucket = q["retention_bucket"]
            assert bucket in bucket_gap_days, f"{q['question_id']} unknown bucket {bucket}"
            assert q["original_memory_id"] == q["relevant_memory_ids"][0], q["question_id"]
            assert q["memory_age_days"] == bucket_gap_days[bucket], (
                f"{q['question_id']} memory_age_days={q['memory_age_days']} "
                f"!= expected {bucket_gap_days[bucket]} for bucket {bucket}")
            assert q["intervening_memory_count"] == bucket_filler_counts[bucket], (
                f"{q['question_id']} intervening_memory_count="
                f"{q['intervening_memory_count']} != expected "
                f"{bucket_filler_counts[bucket]} filler memories for bucket {bucket}")
            assert q["retention_design"] == bucket_design[bucket], q["question_id"]
            if q["retention_design"] == "crossed":
                assert q["age_level"] in ("low", "high"), q["question_id"]
                assert q["clutter_level"] in ("low", "high"), q["question_id"]

    return counts, len(mems), len(qs)


def write_jsonl(records, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(_ordered(r), ensure_ascii=False) + "\n")


def main():
    records = build_dataset()
    counts, n_mem, n_q = validate(records)
    write_jsonl(records, config.DATASET_PATH)

    retention_qs = [r for r in records if r.get("category") == "long_term_retention" and r["type"] == "question"]
    by_bucket = {b["bucket"]: 0 for b in RETENTION_BUCKETS + CROSSED_BUCKETS}
    for q in retention_qs:
        by_bucket[q["retention_bucket"]] += 1

    print(f"Wrote {config.DATASET_PATH}")
    print(f"  memories : {n_mem}")
    print(f"  questions: {n_q}  ({counts})")
    print(f"  retention buckets, matched design (questions / age / filler count):")
    for b in RETENTION_BUCKETS:
        print(f"    {b['bucket']:>20}: {by_bucket[b['bucket']]} questions, "
              f"age={b['gap_days']}d, buried under {b['filler_count']} filler memories")
    print(f"  retention buckets, crossed design (age x clutter, decoupled):")
    for b in CROSSED_BUCKETS:
        print(f"    {b['bucket']:>20}: {by_bucket[b['bucket']]} questions, "
              f"age={b['gap_days']}d ({b['age_level']}), "
              f"buried under {b['filler_count']} filler memories ({b['clutter_level']})")
    print(f"  timeline : {records[0]['timestamp'][:10]} -> {records[-1]['timestamp'][:10]}")
    print(f"  sessions : {records[-1]['session_id']}")


if __name__ == "__main__":
    main()
