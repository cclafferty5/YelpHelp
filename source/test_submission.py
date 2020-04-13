import json, sys
from csv import reader, writer
from utils import get_model

OUTPUT_FILE = "output.jsonl"
BEST_MODEL = get_model("models/final_model")

def usage():
    print("Usage: python3 test_submission.py validation_file")
    sys.exit(1)

def eval_review(review_text):
    return BEST_MODEL.review(review_text)

try:    
    _, validation_file = sys.argv
except:
    usage()

with open(validation_file) as valf:
    with open(OUTPUT_FILE, "w+") as outf:
        for line in valf:
            review = json.loads(line)
            output = eval_review(review)
            print(json.dumps({"review_id": review['review_id'],
                     "predicted_stars": eval_review(review['text'])}), file=outf)
