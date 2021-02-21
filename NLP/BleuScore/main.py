from nltk import ngrams 
from collections import Counter 
import numpy as np 
import nltk.translate .bleu_score as bleu
import math 
# translation = 'It is a guide to action which ensures that the military always obeys the commands of the party.'
# references=[
#     'It is a guide to action that ensures that the military will forever heed Party commands.',
#     'It is the guiding principle which guarantees the military forces always being under the command of the Party.',
#     'It is the practical guide for the army always to heed the directions of the party.'
# ]


def count_gram(sentences,ngram=1):
    return Counter(ngrams(sentences,ngram))

def count_clip_gram(result_translation,list_of_references,ngram=1):
    result=dict()
    translation_u=count_gram(result_translation,ngram=ngram)

    for sentence in list_of_references:
        sentence_gram=count_gram(sentence.split(),ngram=ngram)
        for i in translation_u:
            result[i]=max(result.get(i,0),sentence_gram[i])

    return {
        words:min(count,result[words]) for words,count in translation_u.items()
    }
 
def modified_precision(result_translation,list_references,ngram=1):
    count=count_gram(result_translation,ngram=ngram)
    count_clip=count_clip_gram(result_translation,list_references,ngram=ngram)
    return max(1.0*sum(count_clip.values())/max(sum(count.values()),1),1e-10)#smooth values

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
        modified_precision(result_translation,list_of_references,ngram=i) for i in range(1,5,1) 
    ]
  
    score=np.sum(
        [
            0.25*math.log(modified_precision_score[i]) for i in range(len(modified_precision_score))
        ]

    )

    return bp*np.exp(score)

if __name__=="__main__":
    references=['The cat is on the mat.',
            'There is a cat on the mat.'
        ]

    translation_1='the the the mat on the the.'
    translation_2='The cat is on the mat.'
    bleu_score_modified_precision=modified_precision(translation_1.split(),references)
    print("BLEU Score Modified Precision :",bleu_score_modified_precision)
    bleu_final=final_bleu_score(translation_2.split(),references)
    print("BLEU Score Final :",bleu_final)

    #using library
    references_translation=[reference.split() for reference in references]
    print("Bleu score:",bleu.sentence_bleu(references_translation,translation_2.split()))