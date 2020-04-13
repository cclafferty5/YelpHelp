import json, sys
from models import BEST_MODEL_INFO
import tf
from utils import *

OUTPUT_FILE = "output.jsonl"
BEST_MODEL, BEST_MODEL_PATH = BEST_MODEL_INFO

def usage():
    print("Usage: python3 test_submission.py validation_file")
    sys.exit(1)

def eval_review(review_text):
    with Session() as sess:
        BEST_MODEL.saver.restore(sess, BEST_MODEL_PATH)
        feed_dict = {BEST_MODEL.inputs: build_batch_from_sample(review_text)}
        review = sess.run(BEST_MODEL.scores, feed_dict=feed_dict)[0]
        return review

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
