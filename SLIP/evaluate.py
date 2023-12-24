import torch
from torch import nn
import collections
from tqdm import tqdm
from predict import predict
from data.data_manager import manage_data

from pycocoevalcap.bleu.bleu import Bleu
from pycocoevalcap.meteor.meteor import Meteor
from pycocoevalcap.rouge.rouge import Rouge
from pycocoevalcap.cider.cider import Cider


class Scorer():
    def __init__(self, ref, gt):
        self.ref = ref
        self.gt = gt
        print('setting up scorers...')
        self.scorers = [
            (Bleu(4), ["Bleu_1", "Bleu_2", "Bleu_3", "Bleu_4"]),
            (Meteor(), "METEOR"),
            (Rouge(), "ROUGE_L"),
            (Cider(), "CIDEr"),
        ]
    
    def compute_scores(self):
        total_scores = {}
        for scorer, method in self.scorers:
            print('computing %s score...' % (scorer.method()))
            score, scores = scorer.compute_score(self.gt, self.ref)
            if type(method) == list:
                for sc, scs, m in zip(score, scores, method):
                    print("%s: %0.3f" % (m, sc))
                total_scores["Bleu"] = score
            else:
                print("%s: %0.3f" % (method, score))
                total_scores[method] = score
        
        print('*****DONE*****')
        for key, value in total_scores.items():
            print('{}:{}'.format(key, value))
        return total_scores

def evaluate(model: nn.Module, valid_dataseet, device=torch.device("cuda:0"), im_show=False, 
             method = "greedy", max_len=50, steps=20, prompt=" ", **kwargs):
    model.to(device)
    model.eval()
    ref_dict = collections.defaultdict(list)
    hyp_dict = collections.defaultdict(list)
    with torch.no_grad():
        for image_path, captions in tqdm(valid_dataseet, desc="Validating..."):
            ref_dict[image_path].extend(list(captions))  # 加入参考答案
            predictions = predict(model, image_path, steps=steps, max_len=max_len, device=device, 
                                  im_show=im_show, se_print=False, prompt = prompt, method=method, **kwargs)
            hyp_dict[image_path].append(predictions)
            
        scorer = Scorer(hyp_dict, ref_dict)
        scorer.compute_scores()
        print("=========================================")
        

if __name__=="__main__":
    from model.EncoderDecoderModel import EncoderDecoder
    model = EncoderDecoder()
    model.load_state_dict(torch.load("model/checkpoints.pth"))
    valid_dataset = manage_data(root="data/dataset/COCO", max_words=50, dataset="val", version="2017")
    device = torch.device("cuda:0")
    # evaluate(model, valid_dataset, device=device, method="greedy")
    evaluate(model, valid_dataset, device=device, method="beam_search", beam_width=5)
    # evaluate(model, valid_dataset, device=device, method="sampling")
    # evaluate(model, valid_dataset, device=device, method="temperature_sampling", temperature=0.5)
    # evaluate(model, valid_dataset, device=device, method="topk_sampling", top_k=10)