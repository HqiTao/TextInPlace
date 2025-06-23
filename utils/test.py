import faiss
import logging
import numpy as np
from tqdm import tqdm
import string

import torch
from torch.utils.data import DataLoader
from torch.utils.data.dataset import Subset
from utils import visualizations, rerank
from openai import OpenAI


voc = list(string.printable[:-6])


def test(args, eval_ds, model , pca = None):
    """Compute features of the given dataset and compute the recalls."""
    
    model = model.eval()
    with torch.no_grad():
        logging.debug("Extracting database features for evaluation/testing")
        # For database use "hard_resize", although it usually has no effect because database images have same resolution
        database_subset_ds = Subset(eval_ds, list(range(eval_ds.database_num)))
        database_dataloader = DataLoader(dataset=database_subset_ds, num_workers=args.num_workers,
                                         batch_size=args.infer_batch_size, pin_memory=True)
        all_features = np.empty((len(eval_ds), args.features_dim), dtype="float32")

        # database_descriptors_dir = os.path.join(eval_ds.dataset_folder, f"database_{args.aggregation}.npy")
        # database_features = np.load(database_descriptors_dir)
        for inputs, indices in tqdm(database_dataloader, ncols=100):
            features = model(inputs.to("cuda"))
            features = features.cpu().numpy()
            if pca is not None:
                features = pca.transform(features)
            all_features[indices.numpy(), :] = features
        
        # print(model.all_time / eval_ds.database_num)
        
        logging.debug("Extracting queries features for evaluation/testing")
        # queries_infer_batch_size = args.infer_batch_size
        queries_infer_batch_size = 1
        queries_subset_ds = Subset(eval_ds, list(range(eval_ds.database_num, eval_ds.database_num+eval_ds.queries_num)))
        queries_dataloader = DataLoader(dataset=queries_subset_ds, num_workers=args.num_workers,
                                        batch_size=queries_infer_batch_size, pin_memory=True)
        for inputs, indices in tqdm(queries_dataloader, ncols=100):
            features = model(inputs.to("cuda"))
            features = features.cpu().numpy()
            if pca is not None:
                features = pca.transform(features)
            
            all_features[indices.numpy(), :] = features
    
    queries_features = all_features[eval_ds.database_num:]
    database_features = all_features[:eval_ds.database_num]
    
    faiss_index = faiss.IndexFlatL2(args.features_dim)
    faiss_index.add(database_features)
    del database_features, all_features

    logging.debug("Calculating recalls")
    distances, predictions = faiss_index.search(queries_features, max(args.recall_values))
    
    if args.dataset_name == "msls_challenge":
        fp = open("msls_challenge.txt", "w")
        for query in range(eval_ds.queries_num):
            query_path = eval_ds.queries_paths[query]
            fp.write(query_path.split("@")[-1][:-4]+' ')
            for i in range(20):
                pred_path = eval_ds.database_paths[predictions[query,i]]
                fp.write(pred_path.split("@")[-1][:-4]+' ')
            fp.write("\n")
        fp.write("\n")

    #### For each query, check if the predictions are correct
    positives_per_query = eval_ds.get_positives()
    # args.recall_values by default is [1, 5, 10, 20]
    recalls = np.zeros(len(args.recall_values))
    for query_index, pred in enumerate(predictions):
        for i, n in enumerate(args.recall_values):
            if np.any(np.in1d(pred[:n], positives_per_query[query_index])):
                recalls[i:] += 1
                break
    # Divide by the number of queries*100, so the recalls are in percentages
    recalls = recalls / eval_ds.queries_num * 100
    recalls_str = ", ".join([f"R@{val}: {rec:.1f}" for val, rec in zip(args.recall_values, recalls)])

    # Save visualizations of predictions
    if args.num_preds_to_save != 0:
        logging.info("Saving final predictions")
        # For each query save num_preds_to_save predictions
        visualizations.save_preds(predictions[:, :args.num_preds_to_save], eval_ds,
                                args.save_dir, args.save_only_wrong_preds)


    return recalls, recalls_str


def test_text_rerank(args, eval_ds, model, use_llm=False):
    """Compute features of the given dataset and compute the recalls."""
    
    model = model.eval()
    with torch.no_grad():
        logging.debug("Extracting database features for evaluation/testing")
        # For database use "hard_resize", although it usually has no effect because database images have same resolution
        database_subset_ds = Subset(eval_ds, list(range(eval_ds.database_num)))
        database_dataloader = DataLoader(dataset=database_subset_ds, num_workers=args.num_workers,
                                         batch_size=args.infer_batch_size, pin_memory=True)
        all_features = np.empty((len(eval_ds), args.features_dim), dtype="float32")

        database_text = []
        for inputs, indices in tqdm(database_dataloader, ncols=100):
            predictions, frozen_features = model(inputs.to("cuda"))

            recs = []

            if len(predictions) == 0:
                text = [[] for _ in range(inputs.shape[0])]
            else:
                assert len(predictions) == inputs.shape[0], f'{predictions:{len(predictions)}}'
                for pred in predictions:
                    instances = pred["instances"].to("cpu")
                    rec_strings = []
                    for rec, score in zip(instances.recs, instances.rec_scores):
                        rec_string = rec_decode(rec)
                        rec_strings.append(rec_string)
                    recs.append(rec_strings)
                assert len(predictions) == len(recs), f'{recs:{len(recs)}}'
                text = recs

            features = model.vpr_branch(frozen_features)

            database_text = [*database_text, *text]
            features = features.cpu().numpy()
            all_features[indices.numpy(), :] = features
        assert len(database_text) == eval_ds.database_num, \
            f"database_text: {len(database_text)} | eval_ds: {eval_ds.database_num}"
        for text, path in zip(database_text, eval_ds.database_paths):
            if len(text) != 0:
                logging.debug(f"text: {text}, path: {path}")
        
        logging.debug("Extracting queries features for evaluation/testing")
        # queries_infer_batch_size = args.infer_batch_size
        queries_infer_batch_size = 1
        query_text = []
        queries_subset_ds = Subset(eval_ds, list(range(eval_ds.database_num, eval_ds.database_num+eval_ds.queries_num)))
        queries_dataloader = DataLoader(dataset=queries_subset_ds, num_workers=args.num_workers,
                                        batch_size=queries_infer_batch_size, pin_memory=True)
        for inputs, indices in tqdm(queries_dataloader, ncols=100):
            inputs = inputs.to("cuda")
            batch_size, height, width = inputs.shape[0], inputs.shape[-2], inputs.shape[-1]
            batched_inputs = [{'image': inputs[i], 'height': height, 'width': width} for i in range(batch_size)]
            recs = []

            predictions, frozen_features = model.backbone.textmodel(batched_inputs)

            if len(predictions) == 0:
                text = [[] for _ in range(inputs.shape[0])]
            else:
                assert len(predictions) == inputs.shape[0], f'{predictions:{len(predictions)}}'
                for pred in predictions:
                    instances = pred["instances"].to("cpu")
                    rec_strings = []
                    for rec in instances.recs:
                        rec_string = rec_decode(rec)
                        rec_strings.append(rec_string)
                    recs.append(rec_strings)
                assert len(predictions) == len(recs), f'{recs:{len(recs)}}'
                text = recs

            features = model.vpr_branch(frozen_features)
            features = features.cpu().numpy()
            query_text = [*query_text, *text]
            all_features[indices.numpy(), :] = features
            
        assert len(query_text) == eval_ds.queries_num
        for text, path in zip(query_text, eval_ds.queries_paths):
            if len(text) != 0:
                logging.debug(f"text: {text}, path: {path}")

    queries_features = all_features[eval_ds.database_num:]
    database_features = all_features[:eval_ds.database_num]
    
    faiss_index = faiss.IndexFlatL2(args.features_dim)
    faiss_index.add(database_features)
    del database_features, all_features
    
    logging.debug("Calculating recalls")
    distances, predictions = faiss_index.search(queries_features, 100)

    #### For each query, check if the predictions are correct
    positives_per_query = eval_ds.get_positives()
    # args.recall_values by default is [1, 5, 10]
    recalls = np.zeros(len(args.recall_values))
    for query_index, pred in enumerate(predictions):
        for i, n in enumerate(args.recall_values):
            if np.any(np.in1d(pred[:n], positives_per_query[query_index])):
                recalls[i:] += 1
                break
    # Divide by the number of queries*100, so the recalls are in percentages
    recalls = recalls / eval_ds.queries_num * 100
    recalls_str = ", ".join([f"R@{val}: {rec:.1f}" for val, rec in zip(args.recall_values, recalls)])

    ranked = []
    with tqdm(total=len(predictions)) as pbar:
        pbar.set_description('Re-ranking by scene text')
        for q_idx, pred in enumerate(predictions):
            if use_llm:
                api_key = "<Your API Key>" # Please set your api key here.
                base_url = "<Base URL>" # Please set your base url here.
                client = OpenAI(api_key=api_key, base_url=base_url)
                q_text = rerank.generate_reranking(query_text[q_idx], client)
                print(q_text)
                re_pred = _text_rerank(pred, q_text, database_text)
            else:
                re_pred = _text_rerank(pred, query_text[q_idx], database_text)
            if not (re_pred==pred).all():
                ranked.append(dict(id=q_idx, pred=re_pred))
            pbar.update(1)

    logging.info(f"{len(ranked)} queries reranked")

    for i in range(len(ranked)):
        predictions[ranked[i]["id"]] = ranked[i]["pred"]

    # args.recall_values by default is [1, 5, 10]
    recalls_rerank = np.zeros(len(args.recall_values))
    for query_index, pred in enumerate(predictions):
        for i, n in enumerate(args.recall_values):
            if np.any(np.in1d(pred[:n], positives_per_query[query_index])):
                recalls_rerank[i:] += 1
                break
    # Divide by the number of queries*100, so the recalls are in percentages
    recalls_rerank = recalls_rerank / eval_ds.queries_num * 100
    recalls_str_rerank = ", ".join([f"R@{val}: {rec:.1f}" for val, rec in zip(args.recall_values, recalls_rerank)])

    # Save visualizations of predictions
    if args.num_preds_to_save != 0:
        logging.info("Saving final predictions")
        # For each query save num_preds_to_save predictions
        visualizations.save_preds(predictions[:, :args.num_preds_to_save], eval_ds,
                                args.save_dir, args.save_only_wrong_preds)

    return recalls, recalls_str, recalls_rerank, recalls_str_rerank


# scene text based re-ranking
def _text_rerank(pred, q_words, db_words):
    # q_words = words[db_num+q_idx]
    if len(q_words) == 0:
        return pred
    
    q_words = [s for s in q_words if any(char.isdigit() for char in s)]
    
    preds_with_scores = []

    for ref_index in pred:
        r_words = db_words[ref_index]
        r_words = [s for s in r_words if any(char.isdigit() for char in s)]
        if len(r_words) != 0:
            common_strings = set(q_words) & set(r_words)
            numerator = sum(len(s) for s in common_strings) if common_strings else 0
            denominator = sum(len(s) for s in q_words)
            score = numerator / denominator if denominator != 0 else 0
        else:
            score = 0

        preds_with_scores.append((ref_index, score))

    preds_with_scores.sort(key=lambda a: a[1], reverse=True)

    r_predictions, _ = zip(*preds_with_scores)

    return r_predictions


def rec_decode(rec):
    s = ''
    for c in rec:
        c = int(c)
        if c < len(voc):
            s += voc[c]
        elif c == len(voc):
            return s
        else:
            s += u''
    return s
