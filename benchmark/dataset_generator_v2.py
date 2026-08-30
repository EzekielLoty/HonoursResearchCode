"""Builds the v2 stress dataset (harder, fewer memories, tighter k): reuses v1's scenarios plus hard-negative distractors, 3-state temporal chains, multi-hop decoys.

Run: DATASET_VERSION=home_assistant_v2_stress RETRIEVAL_K=3 python -m benchmark.dataset_generator_v2
"""

import json
import random
from datetime import datetime, timedelta

try:
    from .. import config
    from . import dataset_generator as v1
except (ImportError, ValueError):
    import sys
    from pathlib import Path
    sys.path.append(str(Path(__file__).resolve().parents[1]))
    import config
    from benchmark import dataset_generator as v1


# new simple_recall: target + hard-negative distractor each
SIMPLE_RECALL_V2 = [
    {"category": "simple_recall",
     "memories": [
         {"key": "v2s1_fact", "subcategory": "object_location", "day": 100,
          "text": "I keep the umbrella in the closet by the front door."},
         {"key": "v2s1_hn", "subcategory": "object_location", "day": 101,
          "text": "I keep the raincoat in the closet by the back door."},
     ],
     "question": {"subcategory": "object_location", "day": 105,
                  "text": "Where's the umbrella?", "expected_answer": "In the closet by the front door.",
                  "relevant_keys": ["v2s1_fact"], "answer_keys": ["v2s1_fact"]}},
    {"category": "simple_recall",
     "memories": [
         {"key": "v2s2_fact", "subcategory": "food_preference", "day": 102,
          "text": "I like my steak cooked medium-rare."},
         {"key": "v2s2_hn", "subcategory": "food_preference", "day": 103,
          "text": "I like my burgers cooked well-done."},
     ],
     "question": {"subcategory": "food_preference", "day": 108,
                  "text": "How do I like my steak cooked?", "expected_answer": "Medium-rare.",
                  "relevant_keys": ["v2s2_fact"], "answer_keys": ["v2s2_fact"]}},
    {"category": "simple_recall",
     "memories": [
         {"key": "v2s3_fact", "subcategory": "appliance_setting", "day": 104,
          "text": "My preferred washing machine cycle is delicates."},
         {"key": "v2s3_hn", "subcategory": "appliance_setting", "day": 106,
          "text": "My preferred dryer cycle is low heat."},
     ],
     "question": {"subcategory": "appliance_setting", "day": 111,
                  "text": "Which washing machine cycle do I prefer?", "expected_answer": "Delicates.",
                  "relevant_keys": ["v2s3_fact"], "answer_keys": ["v2s3_fact"]}},
    {"category": "simple_recall",
     "memories": [
         {"key": "v2s4_fact", "subcategory": "family", "day": 107,
          "text": "My brother's name is Marcus."},
         {"key": "v2s4_hn", "subcategory": "family", "day": 109,
          "text": "My cousin's name is Marcus too, the one in Chicago."},
     ],
     "question": {"subcategory": "family", "day": 114,
                  "text": "What's my brother's name?", "expected_answer": "Marcus.",
                  "relevant_keys": ["v2s4_fact"], "answer_keys": ["v2s4_fact"]}},
    {"category": "simple_recall",
     "memories": [
         {"key": "v2s5_fact", "subcategory": "recurring_household", "day": 110,
          "text": "The pool gets cleaned every two weeks."},
         {"key": "v2s5_hn", "subcategory": "recurring_household", "day": 112,
          "text": "The hot tub filter gets cleaned every month."},
     ],
     "question": {"subcategory": "recurring_household", "day": 117,
                  "text": "How often does the pool get cleaned?", "expected_answer": "Every two weeks.",
                  "relevant_keys": ["v2s5_fact"], "answer_keys": ["v2s5_fact"]}},
    {"category": "simple_recall",
     "memories": [
         {"key": "v2s6_fact", "subcategory": "object_location", "day": 113,
          "text": "I store winter coats in the cedar chest in the bedroom."},
         {"key": "v2s6_hn", "subcategory": "object_location", "day": 115,
          "text": "I store summer clothes in the cedar chest in the attic."},
     ],
     "question": {"subcategory": "object_location", "day": 120,
                  "text": "Where are the winter coats stored?",
                  "expected_answer": "In the cedar chest in the bedroom.",
                  "relevant_keys": ["v2s6_fact"], "answer_keys": ["v2s6_fact"]}},
    {"category": "simple_recall",
     "memories": [
         {"key": "v2s7_fact", "subcategory": "food_preference", "day": 116,
          "text": "I prefer my eggs scrambled, not fried."},
         {"key": "v2s7_hn", "subcategory": "food_preference", "day": 118,
          "text": "I prefer pancakes over waffles for breakfast."},
     ],
     "question": {"subcategory": "food_preference", "day": 123,
                  "text": "How do I like my eggs?", "expected_answer": "Scrambled.",
                  "relevant_keys": ["v2s7_fact"], "answer_keys": ["v2s7_fact"]}},
    {"category": "simple_recall",
     "memories": [
         {"key": "v2s8_fact", "subcategory": "family", "day": 119,
          "text": "My niece's birthday is in June."},
         {"key": "v2s8_hn", "subcategory": "family", "day": 121,
          "text": "My nephew's birthday is in July."},
     ],
     "question": {"subcategory": "family", "day": 126,
                  "text": "When is my niece's birthday?", "expected_answer": "In June.",
                  "relevant_keys": ["v2s8_fact"], "answer_keys": ["v2s8_fact"]}},
]

# new temporal_update: 5 two-state, 3 three-state chains
TEMPORAL_UPDATE_V2 = [
    {"category": "temporal_update",
     "memories": [
         {"key": "v2t1_old", "subcategory": "phone_setting", "day": 122, "text": "I set my phone to silent mode at night."},
         {"key": "v2t1_new", "subcategory": "phone_setting", "day": 140, "text": "I now keep my phone on vibrate at night."},
     ],
     "question": {"subcategory": "phone_setting", "day": 150,
                  "text": "What mode is my phone on at night?", "expected_answer": "Vibrate.",
                  "relevant_keys": ["v2t1_old", "v2t1_new"], "answer_keys": ["v2t1_new"],
                  "superseded_keys": ["v2t1_old"]}},
    {"category": "temporal_update",
     "memories": [
         {"key": "v2t2_old", "subcategory": "commute", "day": 124, "text": "My commute takes the bus."},
         {"key": "v2t2_mid", "subcategory": "commute", "day": 138, "text": "I switched to biking to work."},
         {"key": "v2t2_new", "subcategory": "commute", "day": 155, "text": "I've started carpooling with a coworker now."},
     ],
     "question": {"subcategory": "commute", "day": 165,
                  "text": "How do I get to work these days?", "expected_answer": "Carpooling with a coworker.",
                  "relevant_keys": ["v2t2_old", "v2t2_mid", "v2t2_new"], "answer_keys": ["v2t2_new"],
                  "superseded_keys": ["v2t2_old", "v2t2_mid"]}},
    {"category": "temporal_update",
     "memories": [
         {"key": "v2t3_old", "subcategory": "room_temperature", "day": 126, "text": "I keep the AC at 72 in summer."},
         {"key": "v2t3_new", "subcategory": "room_temperature", "day": 142, "text": "I've lowered the AC to 70 in summer."},
     ],
     "question": {"subcategory": "room_temperature", "day": 152,
                  "text": "What's the AC set to in summer now?", "expected_answer": "70.",
                  "relevant_keys": ["v2t3_old", "v2t3_new"], "answer_keys": ["v2t3_new"],
                  "superseded_keys": ["v2t3_old"]}},
    {"category": "temporal_update",
     "memories": [
         {"key": "v2t4_old", "subcategory": "supplement", "day": 128, "text": "I take vitamin D in the morning."},
         {"key": "v2t4_mid", "subcategory": "supplement", "day": 144, "text": "I moved my vitamin D to the evening."},
         {"key": "v2t4_new", "subcategory": "supplement", "day": 158, "text": "I stopped taking vitamin D and switched to a multivitamin in the morning."},
     ],
     "question": {"subcategory": "supplement", "day": 168,
                  "text": "What supplement do I take in the morning now?", "expected_answer": "A multivitamin.",
                  "relevant_keys": ["v2t4_old", "v2t4_mid", "v2t4_new"], "answer_keys": ["v2t4_new"],
                  "superseded_keys": ["v2t4_old", "v2t4_mid"]}},
    {"category": "temporal_update",
     "memories": [
         {"key": "v2t5_old", "subcategory": "gym", "day": 130, "text": "My gym membership is at PowerFit."},
         {"key": "v2t5_new", "subcategory": "gym", "day": 146, "text": "I switched my gym membership to CoreStrength."},
     ],
     "question": {"subcategory": "gym", "day": 156,
                  "text": "Which gym am I a member of?", "expected_answer": "CoreStrength.",
                  "relevant_keys": ["v2t5_old", "v2t5_new"], "answer_keys": ["v2t5_new"],
                  "superseded_keys": ["v2t5_old"]}},
    {"category": "temporal_update",
     "memories": [
         {"key": "v2t6_old", "subcategory": "routine", "day": 132, "text": "I water the garden every morning."},
         {"key": "v2t6_new", "subcategory": "routine", "day": 148, "text": "I've moved garden watering to every evening."},
     ],
     "question": {"subcategory": "routine", "day": 158,
                  "text": "When do I water the garden now?", "expected_answer": "Every evening.",
                  "relevant_keys": ["v2t6_old", "v2t6_new"], "answer_keys": ["v2t6_new"],
                  "superseded_keys": ["v2t6_old"]}},
    {"category": "temporal_update",
     "memories": [
         {"key": "v2t7_old", "subcategory": "device_security", "day": 134, "text": "My laptop password hint is my dog's name."},
         {"key": "v2t7_mid", "subcategory": "device_security", "day": 150, "text": "I changed my laptop password hint to my street name."},
         {"key": "v2t7_new", "subcategory": "device_security", "day": 162, "text": "I've disabled the password hint entirely for security."},
     ],
     "question": {"subcategory": "device_security", "day": 172,
                  "text": "Do I still have a password hint set on my laptop?", "expected_answer": "No, I disabled it.",
                  "relevant_keys": ["v2t7_old", "v2t7_mid", "v2t7_new"], "answer_keys": ["v2t7_new"],
                  "superseded_keys": ["v2t7_old", "v2t7_mid"]}},
    {"category": "temporal_update",
     "memories": [
         {"key": "v2t8_old", "subcategory": "object_location", "day": 136, "text": "I keep spare cash in the desk drawer."},
         {"key": "v2t8_new", "subcategory": "object_location", "day": 152, "text": "I moved the spare cash to the safe."},
     ],
     "question": {"subcategory": "object_location", "day": 162,
                  "text": "Where do I keep spare cash now?", "expected_answer": "In the safe.",
                  "relevant_keys": ["v2t8_old", "v2t8_new"], "answer_keys": ["v2t8_new"],
                  "superseded_keys": ["v2t8_old"]}},
]

# new multi_hop: each with a decoy pair (similar person/event, different specifics)
MULTI_HOP_V2 = [
    {"category": "multi_hop",
     "memories": [
         {"key": "v2m1_a", "subcategory": "visitor_preference", "day": 140, "text": "My friend Noah is coming to the picnic on Sunday."},
         {"key": "v2m1_b", "subcategory": "visitor_preference", "day": 141, "text": "Noah has a peanut allergy."},
         {"key": "v2m1_decoy", "subcategory": "visitor_preference", "day": 142, "text": "My friend Owen is also vegetarian and coming to the barbecue next month."},
     ],
     "question": {"subcategory": "visitor_preference", "day": 145,
                  "text": "Is there anything I should avoid bringing to the picnic for Noah?",
                  "expected_answer": "Peanuts — Noah has a peanut allergy.",
                  "relevant_keys": ["v2m1_a", "v2m1_b"], "answer_keys": ["v2m1_a", "v2m1_b"]}},
    {"category": "multi_hop",
     "memories": [
         {"key": "v2m2_a", "subcategory": "visitor_event", "day": 143, "text": "My aunt Carol uses a cane to walk."},
         {"key": "v2m2_b", "subcategory": "visitor_event", "day": 144, "text": "Aunt Carol is coming to stay for the holidays."},
         {"key": "v2m2_decoy", "subcategory": "visitor_event", "day": 146, "text": "My uncle Dave is coming to stay too, but only for a weekend."},
     ],
     "question": {"subcategory": "visitor_event", "day": 149,
                  "text": "Should I prepare anything for Aunt Carol's visit over the holidays?",
                  "expected_answer": "Yes — she uses a cane, so make sure there's clear, step-free access.",
                  "relevant_keys": ["v2m2_a", "v2m2_b"], "answer_keys": ["v2m2_a", "v2m2_b"]}},
    {"category": "multi_hop",
     "memories": [
         {"key": "v2m3_a", "subcategory": "ownership_location", "day": 147, "text": "The gray backpack belongs to my son."},
         {"key": "v2m3_b", "subcategory": "ownership_location", "day": 148, "text": "I keep the gray backpack on the hook by the garage."},
         {"key": "v2m3_decoy", "subcategory": "ownership_location", "day": 150, "text": "The black backpack belongs to my daughter, and it's in her closet."},
     ],
     "question": {"subcategory": "ownership_location", "day": 153,
                  "text": "Where's my son's backpack?", "expected_answer": "On the hook by the garage.",
                  "relevant_keys": ["v2m3_a", "v2m3_b"], "answer_keys": ["v2m3_a", "v2m3_b"]}},
    {"category": "multi_hop",
     "memories": [
         {"key": "v2m4_a", "subcategory": "colleague_visit", "day": 151, "text": "My colleague Emma is stopping by to pick up documents Thursday."},
         {"key": "v2m4_b", "subcategory": "colleague_visit", "day": 152, "text": "Emma prefers I text her instead of calling."},
         {"key": "v2m4_decoy", "subcategory": "colleague_visit", "day": 153, "text": "My colleague Jake is also stopping by Thursday, but he prefers email."},
     ],
     "question": {"subcategory": "colleague_visit", "day": 156,
                  "text": "How should I let Emma know when the documents are ready Thursday?",
                  "expected_answer": "Text her, not call.",
                  "relevant_keys": ["v2m4_a", "v2m4_b"], "answer_keys": ["v2m4_a", "v2m4_b"]}},
    {"category": "multi_hop",
     "memories": [
         {"key": "v2m5_a", "subcategory": "routine_preference", "day": 154, "text": "I go to pottery class every other Saturday."},
         {"key": "v2m5_b", "subcategory": "routine_preference", "day": 155, "text": "After pottery class I usually grab coffee with Dana."},
         {"key": "v2m5_decoy", "subcategory": "routine_preference", "day": 156, "text": "I go to yoga every Saturday morning, before pottery class."},
     ],
     "question": {"subcategory": "routine_preference", "day": 159,
                  "text": "What do I usually do right after pottery class?",
                  "expected_answer": "Grab coffee with Dana.",
                  "relevant_keys": ["v2m5_a", "v2m5_b"], "answer_keys": ["v2m5_a", "v2m5_b"]}},
    {"category": "multi_hop",
     "memories": [
         {"key": "v2m6_a", "subcategory": "member_routine", "day": 157, "text": "My son plays soccer on Wednesdays."},
         {"key": "v2m6_b", "subcategory": "member_routine", "day": 158, "text": "After soccer practice he's always starving and wants a snack ready."},
         {"key": "v2m6_decoy", "subcategory": "member_routine", "day": 159, "text": "My daughter has piano lessons on Wednesdays too, right before soccer."},
     ],
     "question": {"subcategory": "member_routine", "day": 162,
                  "text": "What should be ready for my son after soccer on Wednesdays?",
                  "expected_answer": "A snack — he's always starving after practice.",
                  "relevant_keys": ["v2m6_a", "v2m6_b"], "answer_keys": ["v2m6_a", "v2m6_b"]}},
    {"category": "multi_hop",
     "memories": [
         {"key": "v2m7_a", "subcategory": "neighbor_item", "day": 160, "text": "The spare umbrella belongs to my neighbor Rita."},
         {"key": "v2m7_b", "subcategory": "neighbor_item", "day": 161, "text": "I keep it by the front door to return when I see her."},
         {"key": "v2m7_decoy", "subcategory": "neighbor_item", "day": 162, "text": "My other neighbor Sam borrowed a ladder last month and hasn't returned it."},
     ],
     "question": {"subcategory": "neighbor_item", "day": 165,
                  "text": "Where should I leave the spare umbrella for my neighbor?",
                  "expected_answer": "By the front door, to return when I see her.",
                  "relevant_keys": ["v2m7_a", "v2m7_b"], "answer_keys": ["v2m7_a", "v2m7_b"]}},
    {"category": "multi_hop",
     "memories": [
         {"key": "v2m8_a", "subcategory": "member_allergy", "day": 163, "text": "My mother-in-law is lactose intolerant."},
         {"key": "v2m8_b", "subcategory": "member_allergy", "day": 164, "text": "She's coming for brunch this Sunday."},
         {"key": "v2m8_decoy", "subcategory": "member_allergy", "day": 165, "text": "My father-in-law is coming too, and he doesn't eat spicy food."},
     ],
     "question": {"subcategory": "member_allergy", "day": 168,
                  "text": "What should I keep in mind about food for brunch this Sunday?",
                  "expected_answer": "Avoid dairy — my mother-in-law is lactose intolerant.",
                  "relevant_keys": ["v2m8_a", "v2m8_b"], "answer_keys": ["v2m8_a", "v2m8_b"]}},
]

# retention: v1's bucket/gap structure, fewer cases, smaller filler, +1 hard-negative filler each
RETENTION_BUCKETS_V2 = [
    {"bucket": "1_day", "gap_days": 1, "filler_count": 3, "design": "matched"},
    {"bucket": "7_days", "gap_days": 7, "filler_count": 4, "design": "matched"},
    {"bucket": "30_days", "gap_days": 30, "filler_count": 6, "design": "matched"},
    {"bucket": "90_days", "gap_days": 90, "filler_count": 9, "design": "matched"},
    {"bucket": "180_days", "gap_days": 180, "filler_count": 14, "design": "matched"},
]
CROSSED_BUCKETS_V2 = [
    {"bucket": "crossed_young_sparse", "gap_days": 10, "filler_count": 3,
     "design": "crossed", "age_level": "low", "clutter_level": "low"},
    {"bucket": "crossed_young_dense", "gap_days": 10, "filler_count": 10,
     "design": "crossed", "age_level": "low", "clutter_level": "high"},
    {"bucket": "crossed_old_sparse", "gap_days": 120, "filler_count": 3,
     "design": "crossed", "age_level": "high", "clutter_level": "low"},
    {"bucket": "crossed_old_dense", "gap_days": 120, "filler_count": 10,
     "design": "crossed", "age_level": "high", "clutter_level": "high"},
]

RETENTION_CASES_V2 = {b["bucket"]: v1.RETENTION_CASES[b["bucket"]][:3] for b in RETENTION_BUCKETS_V2}

_CROSSED_EXTRA = {
    "crossed_young_sparse": ("object_location", "I keep the phone charger on the nightstand.",
                              "Where's the phone charger?", "On the nightstand."),
    "crossed_young_dense": ("appliance_setting", "I like the bedroom fan set to low overnight.",
                             "What speed should the bedroom fan be at overnight?", "Low."),
    "crossed_old_sparse": ("family_detail", "My cousin Ellie is the one who taught me to bake.",
                            "Who taught me to bake?", "My cousin Ellie."),
    "crossed_old_dense": ("recurring_household", "The kids' vitamins get restocked every two months.",
                           "How often do the kids' vitamins get restocked?", "Every two months."),
}
CROSSED_CASES_V2 = {
    cell: v1.CROSSED_CASES[cell] + [_CROSSED_EXTRA[cell]]
    for cell in [b["bucket"] for b in CROSSED_BUCKETS_V2]
}

HARD_NEGATIVE_FILLERS = {
    "object_location": [("object_location", "I keep the spare charger cable in the kitchen junk drawer."),
                         ("object_location", "The extra remote batteries are in the hallway console table.")],
    "food_preference": [("food_preference", "I've started drinking green tea instead of black tea in the afternoons."),
                         ("food_preference", "I switched from white bread to sourdough for sandwiches.")],
    "routine": [("routine", "I do my grocery run every Sunday afternoon."),
                ("routine", "I check the mail right after getting home from work.")],
    "appliance_setting": [("appliance_setting", "I set the ceiling fan to run on medium during summer nights."),
                           ("appliance_setting", "The garage door opener is set to close automatically after 10 minutes.")],
    "household_preference": [("household_preference", "I like the curtains fully open in the mornings."),
                              ("household_preference", "I prefer the dishwasher run overnight instead of after dinner.")],
    "family_detail": [("family_detail", "My sister-in-law's favorite color is teal."),
                       ("family_detail", "My second cousin recently moved to Denver.")],
    "recurring_household": [("recurring_household", "The smoke detectors get tested every three months."),
                             ("recurring_household", "The mailbox lock gets a fresh coat of oil once a year.")],
}


def _build_retention_scenarios_v2():
    rng = random.Random(config.SEED)
    shuffled_fillers = v1.FILLER_TEMPLATES.copy()
    rng.shuffle(shuffled_fillers)
    filler_cursor = 0

    hn_cursor = {sc: 0 for sc in HARD_NEGATIVE_FILLERS}  # per-subcategory cursor

    scenarios = []
    day_cursor = 200.0
    scenario_gap_days = 3

    all_buckets = RETENTION_BUCKETS_V2 + CROSSED_BUCKETS_V2
    all_cases = {**RETENTION_CASES_V2, **CROSSED_CASES_V2}

    for bucket in all_buckets:
        for i, (subcategory, fact_text, question_text, expected_answer) in enumerate(all_cases[bucket["bucket"]]):
            fact_key = f"v2ret_{bucket['bucket']}_{i}_orig"
            fact_day = day_cursor
            question_day = day_cursor + bucket["gap_days"]

            memories = [{"key": fact_key, "subcategory": subcategory, "day": fact_day, "text": fact_text}]

            # generic filler, spread strictly between fact and question
            n_generic = bucket["filler_count"]
            n_slots = n_generic + 1  # +1 for the hard-negative filler, also spread in
            for j in range(n_generic):
                filler_subcategory, filler_text = shuffled_fillers[filler_cursor % len(shuffled_fillers)]
                filler_cursor += 1
                offset = bucket["gap_days"] * (j + 1) / (n_slots + 1)
                memories.append({
                    "key": f"{fact_key}_filler_{j}",
                    "subcategory": filler_subcategory,
                    "day": fact_day + offset,
                    "text": filler_text,
                })

            # one hard-negative filler matching this fact's own subcategory
            hn_pool = HARD_NEGATIVE_FILLERS.get(subcategory)
            if hn_pool:
                hn_subcategory, hn_text = hn_pool[hn_cursor[subcategory] % len(hn_pool)]
                hn_cursor[subcategory] += 1
                offset = bucket["gap_days"] * (n_generic + 1) / (n_slots + 1)
                memories.append({
                    "key": f"{fact_key}_hardneg",
                    "subcategory": hn_subcategory,
                    "day": fact_day + offset,
                    "text": hn_text,
                })

            question = {
                "subcategory": subcategory, "day": question_day, "text": question_text,
                "expected_answer": expected_answer, "relevant_keys": [fact_key], "answer_keys": [fact_key],
                "retention_bucket": bucket["bucket"], "retention_design": bucket["design"],
            }
            if "age_level" in bucket:
                question["age_level"] = bucket["age_level"]
                question["clutter_level"] = bucket["clutter_level"]

            scenarios.append({"category": "long_term_retention", "memories": memories, "question": question})
            day_cursor = question_day + scenario_gap_days

    return scenarios


ALL_SCENARIOS_V2 = (
    v1.SIMPLE_RECALL + SIMPLE_RECALL_V2
    + v1.TEMPORAL_UPDATE + TEMPORAL_UPDATE_V2
    + v1.MULTI_HOP + MULTI_HOP_V2
    + _build_retention_scenarios_v2()
)


# compilation reuses v1's build/validate/write machinery

def build_dataset():
    base = datetime.fromisoformat(config.BASE_DATE)
    counter = 0

    mem_entries = []
    for sc in ALL_SCENARIOS_V2:
        for m in sc["memories"]:
            mem_entries.append({
                "_key": m["key"], "user_id": config.USER_ID,
                "timestamp": v1._ts(base, m["day"], counter),
                "type": "memory", "category": sc["category"],
                "subcategory": m["subcategory"], "text": m["text"],
            })
            counter += 1
    mem_entries.sort(key=lambda r: r["timestamp"])

    key_to_id = {}
    for i, m in enumerate(mem_entries, start=1):
        mid = f"m_{i:03d}"
        m["memory_id"] = mid
        key_to_id[m["_key"]] = mid

    q_entries = []
    for sc in ALL_SCENARIOS_V2:
        q = sc["question"]
        rec = {
            "user_id": config.USER_ID, "timestamp": v1._ts(base, q["day"], counter),
            "type": "question", "category": sc["category"], "subcategory": q["subcategory"],
            "text": q["text"], "expected_answer": q["expected_answer"],
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

    ts_by_id = {m["memory_id"]: m["timestamp"] for m in mem_entries}
    for q in q_entries:
        if q["category"] != "long_term_retention":
            continue
        orig_ts = ts_by_id[q["original_memory_id"]]
        q["memory_age_days"] = (datetime.fromisoformat(q["timestamp"]) - datetime.fromisoformat(orig_ts)).days
        q["intervening_memory_count"] = sum(1 for m in mem_entries if orig_ts < m["timestamp"] < q["timestamp"])

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


def validate(records):
    mems = [r for r in records if r["type"] == "memory"]
    qs = [r for r in records if r["type"] == "question"]

    counts = {"simple_recall": 0, "temporal_update": 0, "multi_hop": 0, "long_term_retention": 0}
    for q in qs:
        counts[q["category"]] += 1
    expected_counts = {
        "simple_recall": len(v1.SIMPLE_RECALL) + len(SIMPLE_RECALL_V2),
        "temporal_update": len(v1.TEMPORAL_UPDATE) + len(TEMPORAL_UPDATE_V2),
        "multi_hop": len(v1.MULTI_HOP) + len(MULTI_HOP_V2),
        "long_term_retention": sum(len(RETENTION_CASES_V2[b["bucket"]]) for b in RETENTION_BUCKETS_V2)
        + sum(len(CROSSED_CASES_V2[b["bucket"]]) for b in CROSSED_BUCKETS_V2),
    }
    assert counts == expected_counts, (counts, expected_counts)

    mem_ids = {m["memory_id"] for m in mems}
    assert len(mem_ids) == len(mems), "duplicate memory ids"

    ts_by_id = {m["memory_id"]: m["timestamp"] for m in mems}
    all_buckets = RETENTION_BUCKETS_V2 + CROSSED_BUCKETS_V2
    bucket_gap_days = {b["bucket"]: b["gap_days"] for b in all_buckets}
    bucket_filler_counts = {b["bucket"]: b["filler_count"] + 1 for b in all_buckets}  # +1: hard-negative filler
    bucket_design = {b["bucket"]: b["design"] for b in all_buckets}

    for q in qs:
        for mid in q["relevant_memory_ids"]:
            assert mid in mem_ids, f"{q['question_id']} references missing {mid}"
            assert ts_by_id[mid] < q["timestamp"], f"{q['question_id']} asked before memory {mid} existed"
        if q["category"] == "multi_hop":
            assert len(q["answer_memory_ids"]) >= 2, q["question_id"]
        if q["category"] == "long_term_retention":
            bucket = q["retention_bucket"]
            assert bucket in bucket_gap_days, f"{q['question_id']} unknown bucket {bucket}"
            assert q["original_memory_id"] == q["relevant_memory_ids"][0], q["question_id"]
            assert q["memory_age_days"] == bucket_gap_days[bucket], (
                q["question_id"], q["memory_age_days"], bucket_gap_days[bucket])
            assert q["intervening_memory_count"] == bucket_filler_counts[bucket], (
                q["question_id"], q["intervening_memory_count"], bucket_filler_counts[bucket])
            assert q["retention_design"] == bucket_design[bucket], q["question_id"]
            if q["retention_design"] == "crossed":
                assert q["age_level"] in ("low", "high"), q["question_id"]
                assert q["clutter_level"] in ("low", "high"), q["question_id"]

    return counts, len(mems), len(qs)


def main():
    records = build_dataset()
    counts, n_mem, n_q = validate(records)
    v1.write_jsonl(records, config.DATASET_PATH)

    print(f"Wrote {config.DATASET_PATH}")
    print(f"  memories : {n_mem}")
    print(f"  questions: {n_q}  ({counts})")
    print(f"  timeline : {records[0]['timestamp'][:10]} -> {records[-1]['timestamp'][:10]}")
    print(f"  sessions : {records[-1]['session_id']}")


if __name__ == "__main__":
    main()
