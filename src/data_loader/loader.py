import json

class JSONLLoader:
    @staticmethod
    def load_texts(path, text_field="text"):
        texts = []
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                record = json.loads(line)
                if text_field in record:
                    texts.append(record[text_field])
        print(f"📄 Loaded {len(texts)} texts from {path}")
        return texts