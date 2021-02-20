from nltk import ngrams 
from collections import Counter 
import numpy as np 


translation = 'It is a guide to action which ensures that the military always obeys the commands of the party.'
references=[
    'It is a guide to action that ensures that the military will forever heed Party commands.',
    'It is the guiding principle which guarantees the military forces always being under the command of the Party.',
    'It is the practical guide for the army always to heed the directions of the party.'
]

def count_gram(sentences,ngram=1):
    return Counter(ngrams(sentences,ngram))

def count_clip_gram(result_translation,list_of_references,ngram=1):
    result=dict()
    translation_u=count_gram(result_translation,ngram=ngram)

    for sentence in list_of_references:
        sentence_gram=count_gram(sentence.split(),ngram=ngram)
        for i in sentence_gram:
            if i in result:
                result[i]=max(result[i],sentence_gram[i])
            else:
                result[i]=sentence_gram[i]
            
    return {
        k: min(result.get(k,0),translation_u.get(k,0)) for k in translation_u
    }
 
def modified_precision(result_translation,list_references,ngram=1):
    count=count_gram(result_translation)
    count_clip=count_clip_gram(result_translation,list_references,ngram=ngram)
    return 1.0*sum(count_clip.values())/max(sum(count.values()),1)

def get_closest_reference(result_translation,list_of_references):
    len_trans=len(result_translation)
    index=np.argmin([abs(len(reference)-len_trans) for reference in list_of_references])
    return len(list_of_references[index].split())

def brevity_penalty(result_translation,list_of_references):
    c=len(result_translation)
    r=get_closest_reference(result_translation,list_of_references)
    if c>r:
        return 1
    return np.exp(1-r*1.0/c)

def final_bleu_score(result_translation,list_of_references):
    bp=brevity_penalty(result_translation,list_of_references)
    modified_precision_score=[
        modified_precision(result_translation,list_of_references,ngram=i) \
             for i in range(1,4,1) 
    ]
    score=np.sum(
        [
            0.25*np.log(modified_precision_score[i]) if modified_precision_score[i]>0 else 0 
            for i in range(len(modified_precision_score))
        ]

    )
    return bp*np.exp(score)

bleu_score_modified_precision=modified_precision(translation.split(),references)
print("BLEU Score Modified Precision :",bleu_score_modified_precision)
bleu_final=final_bleu_score(translation.split(),references)
print("BLEU Score Final :",bleu_final)