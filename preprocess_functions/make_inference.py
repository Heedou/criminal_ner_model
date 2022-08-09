import numpy as np

def ner_inference(test_sentence, nr_model, tokenizer, max_len, ner_labels):

  tokenized_sentence = np.array([tokenizer.encode(test_sentence, max_length=max_len, truncation=True, padding='max_length')])
  tokenized_mask = np.array([[int(x!=1) for x in tokenized_sentence[0].tolist()]])
  ans = nr_model.predict([tokenized_sentence, tokenized_mask])
  ans = np.argmax(ans, axis=2)

  tokens = tokenizer.convert_ids_to_tokens(tokenized_sentence[0])
  new_tokens, new_labels = [], []
  for token, label_idx in zip(tokens, ans[0]):
    if (token.startswith("##")):
      new_labels.append(ner_labels[label_idx])
      new_tokens.append(token[2:])
    elif (token=='[CLS]'):
      pass
    elif (token=='[SEP]'):
      pass
    elif (token=='[PAD]'):
      pass
    elif (token != '[CLS]' or token != '[SEP]'):
      new_tokens.append(token)
      new_labels.append(ner_labels[label_idx])

  for token, label in zip(new_tokens, new_labels):
      print("{}\t{}".format(label, token))