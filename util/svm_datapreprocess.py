import json
import cPickle

def generate_rupen_docs(gsr_file, clean_str=False):
    docs = []
    vocab = defaultdict(float)
    type2id = {}
    pop2id = {}

    tid = 0
    pid = 0

    with open(gsr_file) as gf:
        for line in gf:
            event = json.loads(line)
            # check the data, remove those data without downloaded articles
            if len(event["downloaded_articles"]) == 0:
                continue
            articles = []
            articles_sens = []
            for url, value in event["downloaded_articles"].items():
                if not isinstance(value, dict):
                    continue
                tokens = value["original_text_basis_enrichment"]["tokens"]
                if len(tokens) > 0:
                    # compare the similarity of current articles with pervious
                    content = u' '.join([t['value'] for t in tokens])
                    dup = False
                    for article in articles:
                        if content[:100] == article[:100]:
                            dup = True
                    if not dup:
                        articles.append(content)
                        # construct the sentences
                        sens = []
                        sen = []
                        for token in tokens:
                            if token['POS'] == 'SENT':
                                sen.append(token['value'])
                                sens.append(sen)
                                sen = []
                            else:
                                sen.append(token['value'])
                        if len(sen) > 0:
                            sens.append(sen)
                        articles_sens.append(sens)

            # we construct each event for each individual articles
            for i, article in enumerate(articles):
                doc = {}
                eventType = event["eventType"]
                eventDate = event["eventDate"]
                population = event["population"]
                location = event["location"]
                if eventType not in type2id:
                    type2id[eventType] = tid
                    tid += 1
                if population not in pop2id:
                    pop2id[population] = pid
                    pid += 1
                
                doc["etype"] = type2id[eventType]
                doc["pop"] = pop2id[population]
                doc["location"] = location
                doc["eventDate"] = eventDate
                doc["content"] = article
                doc["sens"] = articles_sens[i]

                if clean_str:
                    content = clean_content(content)
                tokens = content.split()
                words = set(tokens)
                for w in words:
                    vocab[w] += 1
                    
                doc["tokens"] = tokens
                doc["length"] = len(tokens)
                doc["cv"] = np.random.randint(0, 10)
                docs.append(doc)
    return docs, vocab, type2id, pop2id

def dump_gsrs(gsr_file, clean_str=False):
    docs, vocab, type2id, pop2id = generate_rupen_docs(gsr_file, clean_str)
    outfile = '../data/svm_dataset'
    with open(outfile, 'wb') as otf:
        cPickle.dump(docs, otf)
        cPickle.dump(vocab, otf)
        cPickle.dump(type2id, otf)
        cPickle.dump(pop2id, otf)

if __name__ == "__main__":
    dump_gsrs('../data/all_gsr_events_BoT-March_2015.hdl-desc-dwnldarticles.translated.enr.json')

