import json, sys
from models import BEST_MODEL
from preprocess import BEST_PREPROCESSOR
from utils import predict_test_set
from getopt import getopt

OUTPUT_FILE = "output.jsonl"

def usage():
    print("Usage: python test_submission.py test_file")
    sys.exit(1)

try:
    opts, args = getopt(sys.argv[1:], "", ["show-accuracy", "keep-texts"])
    assert len(args) == 1
    test_file = args[0]
    show_accuracy, keep_texts = False, False
    for opt, val in opts:
        if opt == "--show-accuracy":
            show_accuracy = True
        elif opt == "--keep-texts":
            keep_texts = True
except:
    usage()

with open(test_file) as tstf:
    test_set = [json.loads(line) for line in tstf]

predict_test_set(test_set, BEST_MODEL, BEST_PREPROCESSOR, show_accuracy=show_accuracy)

with open(OUTPUT_FILE, "w+") as out:
    for d in test_set:
        if not keep_texts:
            del d["text"]
        print(json.dumps(d), file=out)